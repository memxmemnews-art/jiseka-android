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
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
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

@OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {

    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null
    private var modeSelectionLayout: LinearLayout? = null
    private var resultActionLayout: LinearLayout? = null
    private var btnCapture: Button? = null
    private var progressBar: ProgressBar? = null

    private var btnModePassenger: Button? = null
    private var btnModeFront: Button? = null
    private var btnModeDriver: Button? = null

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ThreadPoolExecutor

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null 
    private var displayedBitmap: Bitmap? = null    

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("CAMERA_DEBUG", "OpenCV initialization failed.")
        }

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
     
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeBackgroundView?.scaleType = ImageView.ScaleType.CENTER_CROP
        
        nativeGuideView = findViewById(R.id.nativeGuideView)
        modeSelectionLayout = findViewById(R.id.modeSelectionLayout)
        resultActionLayout = findViewById(R.id.resultActionLayout)
        btnCapture = findViewById(R.id.btnCapture)
        progressBar = findViewById(R.id.progressBar)
   
        btnModePassenger = findViewById(R.id.btnModePassenger)
        btnModeFront = findViewById(R.id.btnModeFront)
        btnModeDriver = findViewById(R.id.btnModeDriver)

        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = ThreadPoolExecutor(
            1, 1, 0L, TimeUnit.MILLISECONDS,
            ArrayBlockingQueue(1), ThreadPoolExecutor.DiscardOldestPolicy()
        )

        setupUIListeners()
        resetToLiveMode()

        if (allPermissionsGranted()) {
            viewFinder?.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun setupUIListeners() {
        btnModePassenger?.setOnClickListener { setMode("PASSENGER", it as Button) }
        btnModeFront?.setOnClickListener { setMode("FRONT", it as Button) }
        btnModeDriver?.setOnClickListener { setMode("DRIVER", it as Button) }

        btnCapture?.setOnClickListener { takePhoto() }
        
        findViewById<Button>(R.id.btnRetry).setOnClickListener { resetToLiveMode() }
        findViewById<Button>(R.id.btnSave).setOnClickListener {
            displayedBitmap?.let { bmp -> saveBitmapToGallery(bmp) } 
                ?: Toast.makeText(this, "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
        }

        nativeGuideView?.onGuideDropListener = { mode ->
            triggerAnalysis(mode)
        }
    }

    private fun setMode(mode: String, clickedButton: Button) {
        val defaultBg = Color.parseColor("#80000000")
        val activeBg = Color.parseColor("#00FF00")
        val defaultText = Color.WHITE
        val activeText = Color.BLACK

        btnModePassenger?.apply { setBackgroundColor(defaultBg); setTextColor(defaultText) }
        btnModeFront?.apply { setBackgroundColor(defaultBg); setTextColor(defaultText) }
        btnModeDriver?.apply { setBackgroundColor(defaultBg); setTextColor(defaultText) }

        clickedButton.setBackgroundColor(activeBg)
        clickedButton.setTextColor(activeText)

        nativeGuideView?.setMode(mode)
        
        synchronized(bitmapLock) {
            lastCapturedBitmap?.let {
                nativeBackgroundView?.setImageBitmap(it)
            }
        }
    }

    private fun resetToLiveMode() {
        btnCapture?.isEnabled = true

        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle()
        displayedBitmap = null

        synchronized(bitmapLock) {
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null
        }

        viewFinder?.visibility = View.VISIBLE
        btnCapture?.visibility = View.VISIBLE
        
        btnModeFront?.let { setMode("FRONT", it) }
        
        nativeGuideView?.visibility = View.GONE
        modeSelectionLayout?.visibility = View.GONE
        nativeBackgroundView?.visibility = View.GONE
        resultActionLayout?.visibility = View.GONE
        progressBar?.visibility = View.GONE
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("CAMERA_DEBUG", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        btnCapture?.isEnabled = false
        progressBar?.visibility = View.VISIBLE
        btnCapture?.visibility = View.GONE

        val currentCapture = imageCapture ?: run {
            Toast.makeText(this, "카메라 초기화 실패", Toast.LENGTH_SHORT).show()
            resetToLiveMode()
            return
        }

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
                            if (isFinishing || isDestroyed) return@runOnUiThread

                            viewFinder?.visibility = View.GONE
                            nativeBackgroundView?.setImageBitmap(uprightBitmap)
                            nativeBackgroundView?.visibility = View.VISIBLE
                            
                            nativeGuideView?.visibility = View.VISIBLE
                            modeSelectionLayout?.visibility = View.VISIBLE
                            
                            resultActionLayout?.visibility = View.GONE
                            progressBar?.visibility = View.GONE
                        }
                    } catch (t: Throwable) {
                        runOnUiThread {
                            if (isFinishing || isDestroyed) return@runOnUiThread
                            Toast.makeText(this@MainActivity, "이미지 처리 오류", Toast.LENGTH_SHORT).show()
                            resetToLiveMode()
                        }
                    } finally {
                        imageProxy.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        if (isFinishing || isDestroyed) return@runOnUiThread
                        Toast.makeText(this@MainActivity, "카메라 캡처 실패", Toast.LENGTH_SHORT).show()
                        resetToLiveMode()
                    }
                }
            }
        )
    }

    private fun calculatePolygonArea(pts: List<PointF>): Float {
        if (pts.size < 3) return 0f
        var area = 0f
        var j = pts.size - 1
        for (i in pts.indices) {
            area += (pts[j].x * pts[i].y) - (pts[i].x * pts[j].y)
            j = i
        }
        return Math.abs(area / 2.0f)
    }

    private fun triggerAnalysis(mode: String) {
        // 🛠️ [치명적 문제 4 방어] UI 스레드 런타임 상태 완벽 고정 유도
        if (Looper.myLooper() != Looper.getMainLooper()) {
            runOnUiThread { triggerAnalysis(mode) }
            return
        }

        val safeTargetBitmap = synchronized(bitmapLock) { 
            lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
        } ?: return

        val guideView = nativeGuideView
        val bgView = nativeBackgroundView

        if (guideView == null || bgView == null || bgView.width <= 1 || bgView.drawable == null) {
            safeTargetBitmap.recycle()
            return
        }

        // 🛠️ [치명적 문제 1 & 4 스냅숏화] 매트릭스, 패딩, 계층 오프셋 정보를 UI 스레드 컨텍스트에서 원자적으로 캡처
        val viewMatrix = Matrix(bgView.imageMatrix)
        val inverseMatrix = Matrix()
        val invertSuccess = viewMatrix.invert(inverseMatrix)
        if (!invertSuccess) {
            Log.e("CAMERA_DEBUG", "Matrix inversion failed (singular matrix).")
            safeTargetBitmap.recycle()
            return
        }

        // 🛠️ [치명적 문제 2 방어] ImageView 내부 패딩 정보 독립 확보
        val padLeft = bgView.paddingLeft
        val padTop = bgView.paddingTop

        // 🛠️ [치명적 문제 1 해결] 공통 부모 계층 구조의 미세 오차 무력화를 위해 스크린 절대 물리 좌표 기준 오프셋 수립
        val guideLoc = IntArray(2)
        val bgLoc = IntArray(2)
        guideView.getLocationOnScreen(guideLoc)
        bgView.getLocationOnScreen(bgLoc)
        val offsetX = guideLoc[0] - bgLoc[0]
        val offsetY = guideLoc[1] - bgLoc[1]

        val uiCorners = guideView.getCorners()

        if (!isFinishing && !isDestroyed) {
            progressBar?.visibility = View.VISIBLE
            resultActionLayout?.visibility = View.GONE 
        }

        analysisExecutor.execute {
            try {
                // 🛠️ [치명적 문제 1 & 2 병합 정합] GuideView 로컬 ➔ ImageView 로컬 이동 후, 패딩을 차감하여 순수 Drawable 차원으로 투영
                var bmpCorners = uiCorners.map { pt ->
                    val bgLocalX = pt.x + offsetX
                    val bgLocalY = pt.y + offsetY
                    val drawableX = bgLocalX - padLeft
                    val drawableY = bgLocalY - padTop
                    
                    val pts = floatArrayOf(drawableX, drawableY)
                    inverseMatrix.mapPoints(pts)
                    
                    val safeX = pts[0].coerceIn(0f, (safeTargetBitmap.width - 1).toFloat())
                    val safeY = pts[1].coerceIn(0f, (safeTargetBitmap.height - 1).toFloat())
                    PointF(safeX, safeY)
                }
                
                val originalArea = calculatePolygonArea(bmpCorners)

                val padding = 0 
                val minX = maxOf(0, bmpCorners.minOf { it.x }.toInt() - padding)
                val minY = maxOf(0, bmpCorners.minOf { it.y }.toInt() - padding)
                val maxX = minOf(safeTargetBitmap.width, bmpCorners.maxOf { it.x }.toInt() + padding)
                val maxY = minOf(safeTargetBitmap.height, bmpCorners.maxOf { it.y }.toInt() + padding)
                
                val cropWidth = maxX - minX
                val cropHeight = maxY - minY

                // 🛠️ [치명적 문제 5 방어] 다운스트림 OpenCV 컨투어 붕괴 및 크래시 방지 최소 안전 한계 도출 (50px)
                if (cropWidth >= 50 && cropHeight >= 50) {
                    val croppedBitmap = Bitmap.createBitmap(safeTargetBitmap, minX, minY, cropWidth, cropHeight)

                    try {
                        val detected = PlateDetectionEngine.findPlateCorners(croppedBitmap)
                  
                        if (detected != null && detected.size == 4) {
                            val detectedArea = calculatePolygonArea(detected)
                            
                            // 실무 타겟 정합 범위 압축 (0.6 ~ 1.4)
                            if (detectedArea >= originalArea * 0.6f && detectedArea <= originalArea * 1.4f) {
                                bmpCorners = detected.map { PointF(it.x + minX, it.y + minY) }
                                
                                // 역변환 정합 파이프라인 (Drawable ➔ ImageView ➔ GuideView 순방향 복원)
                                val newUiCorners = bmpCorners.map { pt ->
                                    val pts = floatArrayOf(pt.x, pt.y)
                                    viewMatrix.mapPoints(pts)
                                    val uiX = pts[0] + padLeft - offsetX
                                    val uiY = pts[1] + padTop - offsetY
                                    PointF(uiX, uiY)
                                }
                                
                                runOnUiThread {
                                    if (isFinishing || isDestroyed) return@runOnUiThread
                                    nativeGuideView?.setCorners(newUiCorners)
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("CAMERA_DEBUG", "ML Kit / Engine 인식 실패", e)
                    } finally {
                        croppedBitmap.recycle()
                    }
                }

                val orderedCorners = orderCorners(bmpCorners)

                val resultMat = Mat()
                Utils.bitmapToMat(safeTargetBitmap, resultMat)
                
                applyMaskToMat(resultMat, orderedCorners)

                val resultBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(resultMat, resultBitmap)
                resultMat.release()

                runOnUiThread {
                    if (isFinishing || isDestroyed) {
                        resultBitmap.recycle()
                        return@runOnUiThread
                    }

                    // 🛠️ [치명적 문제 6 방어] 하드웨어 가속 렌더큐 레이스 컨디션 해결을 위한 지연 소멸 아키텍처
                    val oldBitmap = displayedBitmap
                    nativeBackgroundView?.setImageBitmap(resultBitmap)
                    displayedBitmap = resultBitmap
                    
                    // 현재 프레임 그리기가 끝나고 메시지 큐의 다음 턴에 안전하게 소멸 지시
                    oldBitmap?.let { bmp ->
                        nativeBackgroundView?.post {
                            if (!bmp.isRecycled) bmp.recycle()
                        }
                    }
                    
                    progressBar?.visibility = View.GONE
                    resultActionLayout?.visibility = View.VISIBLE
                }
            } catch (t: Throwable) {
                Log.e("CAMERA_DEBUG", "Analysis Error", t)
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    Toast.makeText(this@MainActivity, "가림막 처리 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                    progressBar?.visibility = View.GONE
                    resultActionLayout?.visibility = View.VISIBLE 
                }
            } finally {
                safeTargetBitmap.recycle()
            }
        }
    }

    private fun orderCorners(corners: List<PointF>): List<PointF> {
        if (corners.size != 4) return corners
        val cx = corners.map { it.x }.average().toFloat()
        val cy = corners.map { it.y }.average().toFloat()
        
        return corners.sortedBy { Math.atan2((it.y - cy).toDouble(), (it.x - cx).toDouble()) }
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
                    mcF = Mat(); ccF = Mat()
                    mc.convertTo(mcF, CvType.CV_32F); cc.convertTo(ccF, CvType.CV_32F)

                    blendedF = Mat()
                    Core.multiply(ccF, alphaMat, ccF)

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
            maskMat?.release(); contour?.release()
            blurredMask?.release(); coloredMask?.release(); alphaMat?.release()
            matChannels.forEach { it.release() }; coloredChannels.forEach { it.release() }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        try {
            val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/JiSeKa")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
            contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
            }

            Toast.makeText(this, "💾 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
            resetToLiveMode() 
        } catch (e: Exception) {
            Toast.makeText(this, "이미지 저장에 실패했습니다.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) viewFinder?.post { startCamera() } else finish()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle(); displayedBitmap = null
        cameraExecutor.shutdownNow(); analysisExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
