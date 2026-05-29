package com.jiseka.app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
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
import androidx.camera.core.UseCaseGroup
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
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max

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
    
    @Volatile private var isRefining = false
    @Volatile private var pendingMaskRequest = false

    @Volatile private var precalculatedCandidates: List<CandidatePolygon> = emptyList()
    @Volatile private var currentlyHoveredBitmapPolygon: List<ImmutablePoint>? = null

    private val viewMatrix = Matrix()
    private val inverseMatrix = Matrix()
    private var isMatrixReady = false

    private val matrixMappingBuffer = FloatArray(2)
    private val nativeGuidePassingBuffer = Array(4) { PointF() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) Log.e("CAMERA_DEBUG", "OpenCV initialization failed.")

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        viewFinder?.scaleType = PreviewView.ScaleType.FILL_CENTER
        
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeBackgroundView?.scaleType = ImageView.ScaleType.CENTER_CROP
        
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
        
        nativeGuideView?.onDwellTriggeredListener = { currentLevel ->
             
            val anchorSnapshot = currentlyHoveredBitmapPolygon?.toList()
            val currentSession = captureSessionId.get()
            val targetLevel = currentLevel + 1
            
            if (anchorSnapshot != null) {
                isRefining = true 
                
                val msg = if (targetLevel == 1) "🔍 1차 정밀 밀착 중..." else "🔬 2차 초정밀 밀착 중..."
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                
                precomputeExecutor.execute {
                   if (captureSessionId.get() != currentSession) { 
                        isRefining = false
                        return@execute 
                    }
                    
                    val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
                    
                    if (safeBitmap != null) {
                        val tightenedPolygon = PlateDetectionEngine.refineAnchoredPolygon(safeBitmap, anchorSnapshot, targetLevel)
                        safeBitmap.recycle()

                        runOnUiThread {
                            if (tightenedPolygon != null && tightenedPolygon != anchorSnapshot && captureSessionId.get() == currentSession) {
                               currentlyHoveredBitmapPolygon = tightenedPolygon
                                val xCoords = tightenedPolygon.map { it.x }
                                val yCoords = tightenedPolygon.map { it.y }
                                
                                val newBounds = RectF(
                                    xCoords.minOrNull() ?: 0f,
                                    yCoords.minOrNull() ?: 0f,
                                    xCoords.maxOrNull() ?: 0f,
                                    yCoords.maxOrNull() ?: 0f
                               )
         
                                precalculatedCandidates = precalculatedCandidates.map { 
                                    if (it.points == anchorSnapshot) CandidatePolygon(tightenedPolygon, newBounds) else it 
                                }

                                val uiTightPolygon = tightenedPolygon.map { pt ->
                                    matrixMappingBuffer[0] = pt.x
                                    matrixMappingBuffer[1] = pt.y
                                    viewMatrix.mapPoints(matrixMappingBuffer)
                                    PointF(matrixMappingBuffer[0], matrixMappingBuffer[1])
                                }.toTypedArray()
                              
                                nativeGuideView?.setHoveredPolygon(uiTightPolygon)
                                nativeGuideView?.notifyRefinementCompleted(true)
                            } else {
                                nativeGuideView?.notifyRefinementCompleted(false)
                            }

                            isRefining = false
                            if (pendingMaskRequest) {
                                pendingMaskRequest = false
                                val target = currentlyHoveredBitmapPolygon?.toList()
                                if (target != null) triggerInstantMasking(target)
                            }
                        }
                    } else {
                        runOnUiThread { isRefining = false }
                    }
                }
            }
        }

        nativeGuideView?.onCrosshairDropListener = {
            if (isRefining) {
                pendingMaskRequest = true
            } else {
                val targetSnapshot = currentlyHoveredBitmapPolygon?.toList() 
                if (targetSnapshot != null) triggerInstantMasking(targetSnapshot)
            }
        }
    }

    private fun resetToLiveMode() {
        captureSessionId.incrementAndGet()
        isRefining = false
        pendingMaskRequest = false
        
        btnCapture?.isEnabled = true
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle()
        displayedBitmap = null

        synchronized(bitmapLock) { 
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null 
        }
        
        precalculatedCandidates = emptyList()
        currentlyHoveredBitmapPolygon = null
        isMatrixReady = false

        nativeGuideView?.resetState()
        
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
            
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
            
            try {
                cameraProvider.unbindAll()
                val viewPort = viewFinder?.viewPort
                if (viewPort != null) {
                    val useCaseGroup = UseCaseGroup.Builder()
                        .addUseCase(preview)
                        .addUseCase(imageCapture!!)
                        .setViewPort(viewPort)
                        .build()
                    cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, useCaseGroup)
                } else {
                    cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
                }
            } catch (e: Exception) { Log.e("CAMERA_DEBUG", "Camera binding failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        btnCapture?.isEnabled = false
        progressBar?.visibility = View.VISIBLE
        btnCapture?.visibility = View.GONE
         
        val currentSessionId = captureSessionId.incrementAndGet()
        imageCapture?.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(imageProxy: ImageProxy) {
                try {
                    val uprightBitmap = imageProxy.toUprightBitmapInternal()
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
                } catch (t: Throwable) { 
                    runOnUiThread { resetToLiveMode() } 
                } finally { 
                    imageProxy.close() 
                }
            }
            override fun onError(exception: ImageCaptureException) { 
                runOnUiThread { resetToLiveMode() } 
            }
        })
    }

    private fun ImageProxy.toUprightBitmapInternal(): Bitmap {
        val cropRect = this.cropRect

        val croppedOriginalBitmap = if (this.format == ImageFormat.YUV_420_888) {
            yuv420ToCroppedBitmap(this, cropRect)
        } else {
            val buffer = planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            
            val fullBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: throw IllegalStateException("Bitmap decode failed: JPEG 데이터를 디코딩할 수 없습니다.")
                
            val cropped = Bitmap.createBitmap(fullBitmap, cropRect.left, cropRect.top, cropRect.width(), cropRect.height())
            if (cropped != fullBitmap) fullBitmap.recycle()
            cropped
         }

        val rotateMatrix = Matrix().apply {
            postRotate(imageInfo.rotationDegrees.toFloat())
        }

        val rotatedBitmap = Bitmap.createBitmap(
            croppedOriginalBitmap,
            0, 0,
            croppedOriginalBitmap.width, 
            croppedOriginalBitmap.height,
            rotateMatrix, true
        )

        if (rotatedBitmap != croppedOriginalBitmap) {
            croppedOriginalBitmap.recycle()
        }

        val finalBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)
        if (finalBitmap != rotatedBitmap) {
             rotatedBitmap.recycle()
        }

        return finalBitmap
    }

    private fun yuv420ToCroppedBitmap(image: ImageProxy, cropRect: Rect): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
         
        yuvImage.compressToJpeg(cropRect, 100, out)
        
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: throw IllegalStateException("Bitmap decode failed: YUV 데이터를 비트맵으로 변환할 수 없습니다.")
    }

    private fun setupMatrixAndPrecalculate(sessionId: Int) {
        val bgView = nativeBackgroundView ?: return
        val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap } ?: return
        
        bgView.viewTreeObserver.addOnPreDrawListener(object : ViewTreeObserver.OnPreDrawListener {
            override fun onPreDraw(): Boolean {
                bgView.viewTreeObserver.removeOnPreDrawListener(this)
                
                val viewW = bgView.width.toFloat()
                val viewH = bgView.height.toFloat()
                val imgW = safeBitmap.width.toFloat()
                val imgH = safeBitmap.height.toFloat()
                
                val scale = max(viewW / imgW, viewH / imgH)
                val offsetX = (viewW - (imgW * scale)) / 2f
                val offsetY = (viewH - (imgH * scale)) / 2f
                
                viewMatrix.reset()
                viewMatrix.postScale(scale, scale)
                viewMatrix.postTranslate(offsetX, offsetY)
         
                isMatrixReady = viewMatrix.invert(inverseMatrix)
                
                nativeGuideView?.visibility = View.VISIBLE
                
                precomputeExecutor.execute {
                    if (captureSessionId.get() != sessionId) return@execute
                    val bitmapCopy = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
                    if (bitmapCopy != null) {
                        val candidates = PlateDetectionEngine.precalculateGeometryCandidates(bitmapCopy)
                        bitmapCopy.recycle()
                        if (captureSessionId.get() == sessionId) precalculatedCandidates = candidates
                    }
                }
                return true 
            }
        })
        bgView.invalidate() 
    }

    /**
     * 십자선 이동 시 매칭 시스템 검증 로그 통합 버전
     */
    private fun handleCrosshairMove(uiPoint: PointF) {
        val candidatesSnapshot = precalculatedCandidates.toList()
        
        // 🌟 [로그 1] 엔진이 찾은 원본 후보군 총 개수 출력
        Log.d("DEBUG", "candidate count = ${candidatesSnapshot.size}")
        
        if (!isMatrixReady) {
            Log.d("DEBUG", "[Skip] 역연산 매트릭스(inverseMatrix) 준비 해제 상태")
            return
        }
        if (candidatesSnapshot.isEmpty()) {
            // 후보군 자체가 0개면 뒤쪽 연산을 수행할 필요 없이 조기 종료
            return
        }
        
        matrixMappingBuffer[0] = uiPoint.x
        matrixMappingBuffer[1] = uiPoint.y
        inverseMatrix.mapPoints(matrixMappingBuffer)
        
        val bitmapX = matrixMappingBuffer[0]
        val bitmapY = matrixMappingBuffer[1]
        
        // 🌟 [로그 2] 변환된 비트맵 기준 실제 매핑 픽셀 좌표 출력
        Log.d("DEBUG", "bitmap point = $bitmapX, $bitmapY")
        
        var bestPolygon: List<ImmutablePoint>? = null
        var minArea = Float.MAX_VALUE

        for (candidate in candidatesSnapshot) {
            // 🌟 [로그 4] 순회 중인 개별 후보 다각형의 바운딩 박스 출력
            Log.d("DEBUG", "candidate bounds = ${candidate.bounds}")
            
            if (candidate.bounds.contains(bitmapX, bitmapY)) {
                // Bounds 필터 합격 시 다각형 내부 매칭 여부 체크 시작
                if (isPointInPolygon(bitmapX, bitmapY, candidate.points)) {
                    val pts = candidate.points
                    val pointArray = Array(4) { idx -> Point(pts[idx].x.toDouble(), pts[idx].y.toDouble()) }
                    val matOfPoint = MatOfPoint(*pointArray)
                  
                    val actualArea = Imgproc.contourArea(matOfPoint).toFloat()
                    matOfPoint.release() 
                    
                    if (actualArea < minArea) { 
                        minArea = actualArea
                        bestPolygon = pts.toList() 
                    }
                } else {
                    Log.d("DEBUG", "-> [Fail] Bounds 내부엔 들어왔으나 Point-In-Polygon 수학 공식 검사 탈락")
                }
            }
        }
        
        // 🌟 [로그 3] 최종적으로 매칭에 성공한 최적 다각형 확보 여부 출력
        Log.d("DEBUG", "best polygon found = ${bestPolygon != null}")
        
        currentlyHoveredBitmapPolygon = bestPolygon
         
        if (bestPolygon != null) {
            for (i in 0 until 4) {
                val pt = bestPolygon[i]
                matrixMappingBuffer[0] = pt.x
                matrixMappingBuffer[1] = pt.y
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
                            resultBitmap.recycle()
                            return@runOnUiThread 
                        }
           
                        val oldBitmap = displayedBitmap
                        nativeBackgroundView?.setImageBitmap(resultBitmap)
                        displayedBitmap = resultBitmap
                         
                        oldBitmap?.let { bmp -> 
                            nativeBackgroundView?.post { if (!bmp.isRecycled) bmp.recycle() } 
                        }
                        
                        progressBar?.visibility = View.GONE
                        resultActionLayout?.visibility = View.VISIBLE
                    }
                } finally { 
                    safeTargetBitmap.recycle() 
                }
            }
        }
    }

    private fun isPointInPolygon(px: Float, py: Float, polygon: List<ImmutablePoint>): Boolean {
        var result = false
        var j = polygon.size - 1
        for (i in polygon.indices) {
            if ((polygon[i].y > py) != (polygon[j].y > py) && (px < (polygon[j].x - polygon[i].x) * (py - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) result = !result
            j = i
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
         
        var maskMat: Mat? = null
        var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null
        var coloredMask: Mat? = null
        var alphaMat: Mat? = null
        
        val matChannels = ArrayList<Mat>()
        val coloredChannels = ArrayList<Mat>()
        
        try {
            val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }
            maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
            contour = org.opencv.core.MatOfPoint(*pts.toTypedArray())
             
            Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))
            
            blurredMask = Mat()
            Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)
            
            coloredMask = Mat(mat.size(), mat.type(), Scalar(0.0, 255.0, 0.0, 255.0))
            alphaMat = Mat()
             
            blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)
            
            Core.split(mat, matChannels)
            Core.split(coloredMask, coloredChannels)
            
            for (i in 0 until 3) {
                var mcF: Mat? = null
                var ccF: Mat? = null
                var blendedF: Mat? = null
                var invAlpha: Mat? = null
                var scalarMat: Mat? = null
                try {
                    mcF = Mat()
                    ccF = Mat()
                    matChannels[i].convertTo(mcF, CvType.CV_32F)
                     
                    coloredChannels[i].convertTo(ccF, CvType.CV_32F)
                    
                    blendedF = Mat()
                    Core.multiply(ccF, alphaMat, ccF)
                    
                     
                    invAlpha = Mat()
                    scalarMat = Mat(alphaMat.size(), alphaMat.type(), Scalar(1.0))
                    
                    Core.subtract(scalarMat, alphaMat, invAlpha)
                    Core.multiply(mcF, invAlpha, mcF)
                     
                    Core.add(ccF, mcF, blendedF)
                    blendedF.convertTo(matChannels[i], CvType.CV_8U)
                } finally { 
                    mcF?.release()
                    ccF?.release()
                    blendedF?.release()
                    invAlpha?.release()
                    scalarMat?.release()
                }
            }
             
            Core.merge(matChannels, mat)
        } finally {
            maskMat?.release()
            contour?.release()
            blurredMask?.release()
            coloredMask?.release()
            alphaMat?.release()
            matChannels.forEach { it.release() }
            coloredChannels.forEach { it.release() }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        try {
            val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
            val values = ContentValues().apply { 
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg") 
             }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
            contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            Toast.makeText(this, "💾 저장 완료", Toast.LENGTH_SHORT).show()
            resetToLiveMode()
        } catch (e: Exception) { 
            Toast.makeText(this, "저장 실패", Toast.LENGTH_SHORT).show() 
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && allPermissionsGranted()) viewFinder?.post { startCamera() }
    }
    
    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA).all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }
    
    override fun onDestroy() {
        synchronized(bitmapLock) { 
             lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null 
        }
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle()
        displayedBitmap = null
        cameraExecutor.shutdownNow()
        precomputeExecutor.shutdownNow()
        maskExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object { 
         private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) 
    }
}
