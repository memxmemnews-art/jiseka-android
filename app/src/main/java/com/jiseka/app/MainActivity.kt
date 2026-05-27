package com.jiseka.app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PointF
import android.os.Build
import android.os.Bundle
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

@OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {

    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null
    private var resultActionLayout: LinearLayout? = null
    private var btnCapture: Button? = null
    private var progressBar: ProgressBar? = null

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    
    private lateinit var precomputeExecutor: ThreadPoolExecutor
    private lateinit var maskExecutor: ThreadPoolExecutor

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null 
    private var displayedBitmap: Bitmap? = null    

    private val captureSessionId = AtomicInteger(0)

    @Volatile private var precalculatedCandidates: List<CandidatePolygon> = emptyList()
    @Volatile private var currentlyHoveredBitmapPolygon: List<ImmutablePoint>? = null

    private val viewMatrix = Matrix()
    private val inverseMatrix = Matrix()
    private var isMatrixReady = false

    // 오프셋 연산 완전 제거! Matrix mapPoints 다이렉트 처리용 재사용 버퍼
    private val matrixMappingBuffer = FloatArray(2)
    private val nativeGuidePassingBuffer = Array(4) { PointF() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) Log.e("CAMERA_DEBUG", "OpenCV initialization failed.")

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        // 🌟 프리뷰와 캡처 화면 1:1 완벽 거울 모드 (FIT_CENTER)
        viewFinder?.scaleType = PreviewView.ScaleType.FIT_CENTER
        
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeBackgroundView?.scaleType = ImageView.ScaleType.FIT_CENTER
        
        nativeGuideView = findViewById(R.id.nativeGuideView)
        resultActionLayout = findViewById(R.id.resultActionLayout)
        btnCapture = findViewById(R.id.btnCapture)
        progressBar = findViewById(R.id.progressBar)

        cameraExecutor = Executors.newSingleThreadExecutor()
        precomputeExecutor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue(1), ThreadPoolExecutor.DiscardOldestPolicy())
        maskExecutor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue(1), ThreadPoolExecutor.AbortPolicy())

        setupUIListeners()
        resetToLiveMode()

        if (allPermissionsGranted()) viewFinder?.post { startCamera() }
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    private fun setupUIListeners() {
        btnCapture?.setOnClickListener { takePhoto() }
        findViewById<Button>(R.id.btnRetry).setOnClickListener { resetToLiveMode() }
        findViewById<Button>(R.id.btnSave).setOnClickListener {
            displayedBitmap?.let { bmp -> saveBitmapToGallery(bmp) } 
                ?: Toast.makeText(this, "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
        }

        nativeGuideView?.onCrosshairMoveListener = { uiPoint -> handleCrosshairMove(uiPoint) }
        
        // 🌟 다단계 Progressive Refinement 트리거 수신
        nativeGuideView?.onDwellTriggeredListener = { currentLevel ->
            val anchorSnapshot = currentlyHoveredBitmapPolygon?.toList()
            val currentSession = captureSessionId.get()
            val targetLevel = currentLevel + 1
            
            if (anchorSnapshot != null) {
                val msg = if (targetLevel == 1) "🔍 1차 정밀 밀착 중..." else "🔬 2차 초정밀 밀착 중..."
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                
                precomputeExecutor.execute {
                    if (captureSessionId.get() != currentSession) return@execute
                    val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
                    
                    if (safeBitmap != null) {
                        val tightenedPolygon = PlateDetectionEngine.refineAnchoredPolygon(safeBitmap, anchorSnapshot, targetLevel)
                        safeBitmap.recycle()

                        runOnUiThread {
                            if (tightenedPolygon != null && tightenedPolygon != anchorSnapshot && captureSessionId.get() == currentSession) {
                                currentlyHoveredBitmapPolygon = tightenedPolygon
                                
                                val uiTightPolygon = tightenedPolygon.map { pt ->
                                    matrixMappingBuffer[0] = pt.x; matrixMappingBuffer[1] = pt.y
                                    viewMatrix.mapPoints(matrixMappingBuffer)
                                    PointF(matrixMappingBuffer[0], matrixMappingBuffer[1])
                                }.toTypedArray()
                                
                                nativeGuideView?.setHoveredPolygon(uiTightPolygon)
                                nativeGuideView?.notifyRefinementCompleted(true) // 성공! 다음 레벨 타이머 작동
                            } else {
                                nativeGuideView?.notifyRefinementCompleted(false) // 실패 시 정지
                            }
                        }
                    }
                }
            }
        }

        nativeGuideView?.onCrosshairDropListener = {
            val targetSnapshot = currentlyHoveredBitmapPolygon?.toList() 
            if (targetSnapshot != null) {
                triggerInstantMasking(targetSnapshot)
            }
        }
    }

    private fun resetToLiveMode() {
        captureSessionId.incrementAndGet()
        btnCapture?.isEnabled = true
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle(); displayedBitmap = null

        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        
        precalculatedCandidates = emptyList()
        currentlyHoveredBitmapPolygon = null
        isMatrixReady = false

        viewFinder?.visibility = View.VISIBLE
        btnCapture?.visibility = View.VISIBLE
        nativeGuideView?.visibility = View.GONE
        nativeBackgroundView?.visibility = View.GONE
        resultActionLayout?.visibility = View.GONE
        progressBar?.visibility = View.GONE
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // 🌟 4:3 족쇄 해제! 기기 Native Output 스트림 활용
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) { Log.e("CAMERA_DEBUG", "Camera binding failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        btnCapture?.isEnabled = false
        progressBar?.visibility = View.VISIBLE
        btnCapture?.visibility = View.GONE

        val currentSessionId = captureSessionId.incrementAndGet()
        val currentCapture = imageCapture ?: run { resetToLiveMode(); return }

        currentCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    try {
                        val rawBitmap = imageProxy.toUprightBitmap()
                        val maxDim = 1920f
                        val scale = minOf(1f, maxDim / maxOf(rawBitmap.width, rawBitmap.height))
                        val resizedBitmap = if (scale < 1f) {
                            Bitmap.createScaledBitmap(rawBitmap, (rawBitmap.width * scale).toInt(), (rawBitmap.height * scale).toInt(), true)
                        } else rawBitmap
                        val uprightBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
                        
                        if (resizedBitmap !== rawBitmap) resizedBitmap.recycle()
                        if (rawBitmap !== uprightBitmap) rawBitmap.recycle() 

                        synchronized(bitmapLock) {
                            lastCapturedBitmap?.recycle()
                            lastCapturedBitmap = uprightBitmap
                        }

                        runOnUiThread {
                            if (isFinishing || isDestroyed || captureSessionId.get() != currentSessionId) return@runOnUiThread
                            viewFinder?.visibility = View.GONE
                            nativeBackgroundView?.setImageBitmap(uprightBitmap)
                            nativeBackgroundView?.visibility = View.VISIBLE
                            progressBar?.visibility = View.GONE 
                            
                            setupMatrixAndPrecalculate(currentSessionId)
                        }
                    } catch (t: Throwable) { runOnUiThread { resetToLiveMode() }
                    } finally { imageProxy.close() }
                }
                override fun onError(exception: ImageCaptureException) { runOnUiThread { resetToLiveMode() } }
            }
        )
    }

    private fun setupMatrixAndPrecalculate(sessionId: Int) {
        val bgView = nativeBackgroundView ?: return

        bgView.viewTreeObserver.addOnPreDrawListener(object : ViewTreeObserver.OnPreDrawListener {
            override fun onPreDraw(): Boolean {
                bgView.viewTreeObserver.removeOnPreDrawListener(this)

                // 🌟 안드로이드 imageMatrix를 다이렉트로 흡수하여 모든 복잡한 오프셋 패딩 계산 우회
                viewMatrix.set(bgView.imageMatrix)
                isMatrixReady = viewMatrix.invert(inverseMatrix)
                
                nativeGuideView?.visibility = View.VISIBLE
                
                precomputeExecutor.execute {
                    if (captureSessionId.get() != sessionId) return@execute
                    val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
                    if (safeBitmap != null) {
                        val candidates = PlateDetectionEngine.precalculateGeometryCandidates(safeBitmap)
                        safeBitmap.recycle()
                        if (captureSessionId.get() == sessionId) precalculatedCandidates = candidates
                    }
                }
                return true 
            }
        })
        bgView.invalidate() 
    }

    private fun handleCrosshairMove(uiPoint: PointF) {
        if (!isMatrixReady || precalculatedCandidates.isEmpty()) return

        // 오프셋 없이 다이렉트 역행렬 투영
        matrixMappingBuffer[0] = uiPoint.x; matrixMappingBuffer[1] = uiPoint.y
        inverseMatrix.mapPoints(matrixMappingBuffer)
        val bitmapX = matrixMappingBuffer[0]; val bitmapY = matrixMappingBuffer[1]

        var bestPolygon: List<ImmutablePoint>? = null
        var minArea = Float.MAX_VALUE

        for (candidate in precalculatedCandidates) {
            if (candidate.bounds.contains(bitmapX, bitmapY)) {
                if (isPointInPolygon(bitmapX, bitmapY, candidate.points)) {
                    val area = candidate.bounds.width() * candidate.bounds.height()
                    if (area < minArea) {
                        minArea = area; bestPolygon = candidate.points
                    }
                }
            }
        }

        currentlyHoveredBitmapPolygon = bestPolygon

        if (bestPolygon != null) {
            for (i in 0 until 4) {
                val pt = bestPolygon[i]
                matrixMappingBuffer[0] = pt.x; matrixMappingBuffer[1] = pt.y
                viewMatrix.mapPoints(matrixMappingBuffer)
                nativeGuidePassingBuffer[i].set(matrixMappingBuffer[0], matrixMappingBuffer[1])
            }
            nativeGuideView?.setHoveredPolygon(nativeGuidePassingBuffer)
        } else {
            nativeGuideView?.setHoveredPolygon(null)
        }
    }

    private fun triggerInstantMasking(targetCandidate: List<ImmutablePoint>) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            runOnUiThread { triggerInstantMasking(targetCandidate) }
            return
        }

        val currentSessionId = captureSessionId.get()
        progressBar?.visibility = View.VISIBLE
        nativeGuideView?.visibility = View.GONE

        maskExecutor.execute {
            if (captureSessionId.get() != currentSessionId) return@execute

            val safeTargetBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
            if (safeTargetBitmap != null) {
                try {
                    val orderedCorners = orderCorners(targetCandidate)
                    val resultMat = Mat()
                    Utils.bitmapToMat(safeTargetBitmap, resultMat)
                    
                    applyMaskToMat(resultMat, orderedCorners)

                    val resultBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(resultMat, resultBitmap)
                    resultMat.release()

                    runOnUiThread {
                        if (isFinishing || isDestroyed || captureSessionId.get() != currentSessionId) { 
                            resultBitmap.recycle(); return@runOnUiThread 
                        }

                        val oldBitmap = displayedBitmap
                        nativeBackgroundView?.setImageBitmap(resultBitmap)
                        displayedBitmap = resultBitmap
                        
                        oldBitmap?.let { bmp -> nativeBackgroundView?.post { if (!bmp.isRecycled) bmp.recycle() } }
                        
                        progressBar?.visibility = View.GONE
                        resultActionLayout?.visibility = View.VISIBLE
                    }
                } finally { safeTargetBitmap.recycle() }
            }
        }
    }

    private fun isPointInPolygon(px: Float, py: Float, polygon: List<ImmutablePoint>): Boolean {
        var result = false; var j = polygon.size - 1
        for (i in polygon.indices) {
            if ((polygon[i].y > py) != (polygon[j].y > py) &&
                (px < (polygon[j].x - polygon[i].x) * (py - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)
            ) result = !result; j = i
        }
        return result
    }

    private fun orderCorners(corners: List<ImmutablePoint>): List<PointF> {
        if (corners.size != 4) return corners.map { PointF(it.x, it.y) }
        val cx = corners.map { it.x }.average().toFloat()
        val cy = corners.map { it.y }.average().toFloat()
        return corners.map { PointF(it.x, it.y) }.sortedBy { Math.atan2((it.y - cy).toDouble(), (it.x - cx).toDouble()) }
    }

    private fun applyMaskToMat(mat: Mat, corners: List<PointF>) {
        if (corners.size != 4) return
        var maskMat: Mat? = null; var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null; var coloredMask: Mat? = null; var alphaMat: Mat? = null
        val matChannels = ArrayList<Mat>(); val coloredChannels = ArrayList<Mat>()
        
        try {
            val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }
            val maskColor = Scalar(0.0, 255.0, 0.0, 255.0) 

            maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
            contour = org.opencv.core.MatOfPoint(*pts.toTypedArray())
            Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))

            blurredMask = Mat()
            Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)

            coloredMask = Mat(mat.size(), mat.type(), maskColor)
            alphaMat = Mat()
            blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)

            Core.split(mat, matChannels); Core.split(coloredMask, coloredChannels)

            for (i in 0 until 3) {
                var mcF: Mat? = null; var ccF: Mat? = null; var blendedF: Mat? = null
                var invAlpha: Mat? = null; var scalarMat: Mat? = null
                try {
                    val mc = matChannels[i]; val cc = coloredChannels[i]
                    mcF = Mat(); ccF = Mat(); mc.convertTo(mcF, CvType.CV_32F); cc.convertTo(ccF, CvType.CV_32F)
                    blendedF = Mat(); Core.multiply(ccF, alphaMat, ccF)
                    invAlpha = Mat(); scalarMat = Mat(alphaMat.size(), alphaMat.type(), Scalar(1.0))
                    Core.subtract(scalarMat, alphaMat, invAlpha)
                    Core.multiply(mcF, invAlpha, mcF); Core.add(ccF, mcF, blendedF)
                    blendedF.convertTo(matChannels[i], CvType.CV_8U)
                } finally {
                    mcF?.release(); ccF?.release(); blendedF?.release()
                    invAlpha?.release(); scalarMat?.release()
                }
            }
            Core.merge(matChannels, mat)
        } finally {
            maskMat?.release(); contour?.release(); blurredMask?.release()
            coloredMask?.release(); alphaMat?.release()
            matChannels.forEach { it.release() }; coloredChannels.forEach { it.release() }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        try {
            val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename); put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/JiSeKa"); put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
            contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
            }
            Toast.makeText(this, "💾 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
            resetToLiveMode() 
        } catch (e: Exception) { Toast.makeText(this, "저장 실패", Toast.LENGTH_SHORT).show() }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) viewFinder?.post { startCamera() } else finish()
        }
    }
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle(); displayedBitmap = null
        cameraExecutor.shutdownNow(); precomputeExecutor.shutdownNow(); maskExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
