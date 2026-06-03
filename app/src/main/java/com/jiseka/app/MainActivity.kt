package com.jiseka.app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.RectF
import android.os.Bundle
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.OrientationEventListener
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
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
import org.opencv.core.MatOfPoint2f
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
import kotlin.math.max

@OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {

    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null
    private var resultActionLayout: LinearLayout? = null
    private var btnCapture: Button? = null
    private var btnRetry: Button? = null
    private var btnSave: Button? = null
    private var progressBar: ProgressBar? = null
    private var guideText: TextView? = null

    private var orientationEventListener: OrientationEventListener? = null
    private var currentLogicalRotation = 0f
    private var accumulatedRotation = 0f

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    
    private lateinit var precomputeExecutor: ThreadPoolExecutor
    private lateinit var maskExecutor: ThreadPoolExecutor

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null 
    private var displayedBitmap: Bitmap? = null    

    // 🌟 텍스처를 메모리에 한 번만 캐싱하여 재사용 및 I/O 병목 방지
    private var cachedTextureMat: Mat? = null

    private val captureSessionId = AtomicInteger(0)
    
    @Volatile private var isRefining = false
    @Volatile private var pendingMaskRequest = false

    @Volatile private var precalculatedCandidates: List<CandidatePolygon> = emptyList()
    @Volatile private var currentlyHoveredBitmapPolygon: List<ImmutablePoint>? = null

    private val viewMatrix = Matrix()
    private val inverseMatrix = Matrix()
    private var isMatrixReady = false

    private val matrixMappingBuffer = FloatArray(2)

    private val uiHandler = android.os.Handler(Looper.getMainLooper())
    private val hideGuideTextRunnable = Runnable {
        guideText?.animate()?.alpha(0f)?.setDuration(300)?.withEndAction {
            guideText?.visibility = View.GONE
        }?.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        if (!OpenCVLoader.initDebug()) {
            Log.e("CAMERA_DEBUG", "OpenCV initialization failed in MainActivity.")
            Toast.makeText(this, "엔진 초기화에 실패했습니다. 앱을 다시 실행해주세요.", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        // 🌟 앱 시작 시 텍스처 파일 1회 로드 및 예외 처리 (Crash 방지)
        loadTextureSafely()

        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        viewFinder?.scaleType = PreviewView.ScaleType.FILL_CENTER
        
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeBackgroundView?.scaleType = ImageView.ScaleType.CENTER_CROP
        
        nativeGuideView = findViewById(R.id.nativeGuideView)
        resultActionLayout = findViewById(R.id.resultActionLayout)
        btnCapture = findViewById(R.id.btnCapture)
        btnRetry = findViewById(R.id.btnRetry)
        btnSave = findViewById(R.id.btnSave)
        progressBar = findViewById(R.id.progressBar)
        guideText = findViewById(R.id.guideText)

        cameraExecutor = Executors.newSingleThreadExecutor()
         
        precomputeExecutor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue(1), ThreadPoolExecutor.DiscardOldestPolicy())
        maskExecutor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue(1), ThreadPoolExecutor.AbortPolicy())

        setupUIListeners()
        setupOrientationListener()
    
        resetToLiveMode()

        if (allPermissionsGranted()) viewFinder?.post { startCamera() }
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    // 🌟 안전한 텍스처 로드 함수
    private fun loadTextureSafely() {
        val rawBitmap = BitmapFactory.decodeResource(resources, R.drawable.plate_texture)
       
        if (rawBitmap != null) {
            cachedTextureMat = Mat()
            Utils.bitmapToMat(rawBitmap, cachedTextureMat!!)
            
            // 캐싱할 때부터 안전한 3채널(RGB)로 변환해 둠
            if (cachedTextureMat!!.channels() == 4) {
                Imgproc.cvtColor(cachedTextureMat!!, cachedTextureMat!!, Imgproc.COLOR_RGBA2RGB)
            }
            rawBitmap.recycle()
        } else {
            Log.e("MASK_INIT", "plate_texture 로드 실패. 파일명 오타 또는 누락 확인 요망.")
            Toast.makeText(this, "경고: 텍스처를 찾을 수 없어 기본 색상으로 대체됩니다.", Toast.LENGTH_LONG).show()
            // Null 처리 대신 기본 어두운 회색 텍스처를 생성하여 메모리에 올림 (Crash 절대 방어)
            cachedTextureMat = Mat(100, 300, CvType.CV_8UC3, Scalar(70.0, 70.0, 70.0))
        }
    }

    private fun setupUIListeners() {
        btnCapture?.setOnClickListener { takePhoto() }
        btnRetry?.setOnClickListener { resetToLiveMode() }
        btnSave?.setOnClickListener {
            displayedBitmap?.let { bmp -> saveBitmapToGallery(bmp) } 
            ?: Toast.makeText(this, "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
        }

        nativeGuideView?.onCrosshairMoveListener = { uiPoint -> handleCrosshairMove(uiPoint) }
        
        nativeGuideView?.onDwellTriggeredListener = { uiPoint ->
            val currentSession = captureSessionId.get()
            isRefining = true 
     
            Toast.makeText(this, "🔍 정밀 검사 중...", Toast.LENGTH_SHORT).show()
            
            precomputeExecutor.execute {
                if (captureSessionId.get() != currentSession) { 
                    isRefining = false
                    return@execute 
                }
                
                val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
       
                if (safeBitmap != null) {
                    val bitmapCoords = FloatArray(2)
                    bitmapCoords[0] = uiPoint.x
                    bitmapCoords[1] = uiPoint.y
    
                    inverseMatrix.mapPoints(bitmapCoords)
         
                    val bitmapX = bitmapCoords[0]
                    val bitmapY = bitmapCoords[1]

                    val tightenedPolygon = PlateDetectionEngine.rescuePlateFromCrosshair(safeBitmap, bitmapX, bitmapY)
    
                    safeBitmap.recycle()

                    runOnUiThread {
                        if (isFinishing || isDestroyed) return@runOnUiThread
                        
                        if (tightenedPolygon != null && tightenedPolygon.isNotEmpty() && captureSessionId.get() == currentSession) {
                            currentlyHoveredBitmapPolygon = tightenedPolygon
           
                            val xCoords = tightenedPolygon.map { it.x }
                            val yCoords = tightenedPolygon.map { it.y }
               
                            val newBounds = RectF(
                                xCoords.minOrNull() ?: 0f, yCoords.minOrNull() ?: 0f,
                                xCoords.maxOrNull() ?: 0f, yCoords.maxOrNull() ?: 0f
                            )
                            
                            precalculatedCandidates = listOf(CandidatePolygon(tightenedPolygon, newBounds))

                            val uiTightPolygon = tightenedPolygon.map { pt ->
                                val renderCoords = FloatArray(2)
                                renderCoords[0] = pt.x
                                renderCoords[1] = pt.y
                                viewMatrix.mapPoints(renderCoords)
                                PointF(renderCoords[0], renderCoords[1])
                            }.toTypedArray()
                            
                            nativeGuideView?.setHoveredPolygon(uiTightPolygon)
                            nativeGuideView?.notifyRefinementCompleted(true)
                        } else {
                            nativeGuideView?.setHoveredPolygon(null)
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

        nativeGuideView?.onCrosshairDropListener = {
            if (isRefining) {
                pendingMaskRequest = true
            } else {
                val targetSnapshot = currentlyHoveredBitmapPolygon?.toList() 
                if (targetSnapshot != null) triggerInstantMasking(targetSnapshot)
            }
        }
    }

    private fun setupOrientationListener() {
        orientationEventListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == ORIENTATION_UNKNOWN) return

                val targetRotation = when (orientation) {
                    in 45..134 -> 270f
                    in 135..224 -> 180f
                    in 225..314 -> 90f
                    else -> 0f
                }

                if (targetRotation != currentLogicalRotation) {
                    var diff = targetRotation - currentLogicalRotation
                    
                    if (diff > 180f) diff -= 360f
                    if (diff < -180f) diff += 360f

                    accumulatedRotation += diff
                    currentLogicalRotation = targetRotation

                    nativeGuideView?.currentDeviceRotation = targetRotation

                    val uiElements = listOf(guideText, btnCapture, btnRetry, btnSave)
                    
                    uiElements.forEach { view ->
                        view?.animate()?.rotation(accumulatedRotation)?.setDuration(200)?.start()
                    }
                }
            }
        }

        if (orientationEventListener?.canDetectOrientation() == true) {
            orientationEventListener?.enable()
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
        
        uiHandler.removeCallbacks(hideGuideTextRunnable) 
        guideText?.alpha = 1f
        guideText?.visibility = View.GONE
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
 
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
                imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
     
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
            } catch (e: Exception) { 
                Log.e("CAMERA_DEBUG", "Camera binding failed", e) 
                runOnUiThread {
                    Toast.makeText(this, "카메라를 실행할 수 없습니다.", Toast.LENGTH_LONG).show()
                    btnCapture?.isEnabled = false
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        if (imageCapture == null) {
            Toast.makeText(this, "카메라 초기화에 실패했거나 준비되지 않았습니다.", Toast.LENGTH_SHORT).show()
            return
        }

        btnCapture?.isEnabled = false
        progressBar?.visibility = View.VISIBLE
        btnCapture?.visibility = View.GONE
 
        val currentSessionId = captureSessionId.incrementAndGet()
        
        try {
            imageCapture?.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    try {
                        val uprightBitmap = imageProxy.toUprightBitmap()
                 
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
                        Log.e("CAMERA_DEBUG", "Error processing captured image", t)
                        runOnUiThread { 
                            resetToLiveMode() 
                            Toast.makeText(this@MainActivity, "이미지 처리 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                        } 
                    } finally { 
                        imageProxy.close() 
                    }
                }
                override fun onError(exception: ImageCaptureException) { 
                    Log.e("CAMERA_DEBUG", "Camera capture failed", exception)
                    runOnUiThread { 
                        resetToLiveMode() 
                        Toast.makeText(this@MainActivity, "촬영에 실패했습니다.", Toast.LENGTH_SHORT).show()
                    } 
                }
            })
        } catch (e: Exception) {
            Log.e("CAMERA_DEBUG", "takePicture call threw an exception", e)
            resetToLiveMode()
            Toast.makeText(this, "카메라 모듈 오류로 촬영할 수 없습니다.", Toast.LENGTH_SHORT).show()
        }
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
                
                guideText?.text = "십자선을 번호판 위로 옮겨주세요"
                guideText?.paint?.isFakeBoldText = true 
                guideText?.textSize = 20f 
                guideText?.alpha = 1f
                guideText?.visibility = View.VISIBLE
                
                uiHandler.removeCallbacks(hideGuideTextRunnable) 
                uiHandler.postDelayed(hideGuideTextRunnable, 2000) 
                
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

    private fun handleCrosshairMove(uiPoint: PointF) {
        val candidatesSnapshot = precalculatedCandidates.toList()
        
        if (!isMatrixReady) return
        if (candidatesSnapshot.isEmpty()) return
        
        matrixMappingBuffer[0] = uiPoint.x
        matrixMappingBuffer[1] = uiPoint.y
        inverseMatrix.mapPoints(matrixMappingBuffer)
     
        val bitmapX = matrixMappingBuffer[0]
        val bitmapY = matrixMappingBuffer[1]
        
        var bestPolygon: List<ImmutablePoint>? = null
        var maxScore = -1.0
        
        val safeBitmapArea = synchronized(bitmapLock) { lastCapturedBitmap?.let { it.width * it.height.toDouble() } ?: 1.0 }

        for (candidate in candidatesSnapshot) {
            if (candidate.bounds.contains(bitmapX, bitmapY)) {
                if (isPointInPolygon(bitmapX, bitmapY, candidate.points)) {
                    val currentScore = PlateDetectionEngine.calculatePolygonScore(candidate.points, safeBitmapArea)
                    
                    if (currentScore > maxScore) {
                        maxScore = currentScore
                        bestPolygon = candidate.points.toList() 
                    }
                }
            }
        }
        
        currentlyHoveredBitmapPolygon = bestPolygon
         
        if (bestPolygon != null) {
            val mappedPoints = Array(bestPolygon.size) { PointF() }
            for (i in bestPolygon.indices) {
                val pt = bestPolygon[i]
                matrixMappingBuffer[0] = pt.x
                matrixMappingBuffer[1] = pt.y
                viewMatrix.mapPoints(matrixMappingBuffer)
                mappedPoints[i] = PointF(matrixMappingBuffer[0], matrixMappingBuffer[1])
            }
            nativeGuideView?.setHoveredPolygon(mappedPoints)
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
        guideText?.visibility = View.GONE 
       
        try {
            maskExecutor.execute {
                if (captureSessionId.get() != currentSessionId) {
                    runOnUiThread { 
                        progressBar?.visibility = View.GONE
                        nativeGuideView?.visibility = View.VISIBLE 
                    }
                    return@execute
                }
                
                val safeTargetBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
                
                if (safeTargetBitmap != null) {
                    try {
                        val orderedCorners = orderCorners(targetCandidate)
                        val resultMat = Mat()
                        Utils.bitmapToMat(safeTargetBitmap, resultMat)

                        // 🌟 메모리에 올려둔 텍스처를 사용하여 마스킹 로직 실행 (I/O 비용 제로)
                        cachedTextureMat?.let { texture ->
                            applyMaskToMat(resultMat, orderedCorners, texture)
                        }

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
                    } catch (e: Exception) {
                        Log.e("CAMERA_DEBUG", "Masking failed", e)
                        runOnUiThread { 
                            progressBar?.visibility = View.GONE
                            nativeGuideView?.visibility = View.VISIBLE 
                        }
                    } finally { 
                        safeTargetBitmap.recycle() 
                    }
                } else {
                    runOnUiThread { 
                        progressBar?.visibility = View.GONE
                        nativeGuideView?.visibility = View.VISIBLE 
                        Toast.makeText(this@MainActivity, "이미지 데이터를 불러올 수 없습니다.", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            Log.w("CAMERA_DEBUG", "Mask execution rejected due to rapid interactions")
            progressBar?.visibility = View.GONE
            nativeGuideView?.visibility = View.VISIBLE
        } catch (e: Exception) {
            Log.e("CAMERA_DEBUG", "Failed to execute masking task", e)
            progressBar?.visibility = View.GONE
            nativeGuideView?.visibility = View.VISIBLE
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
        if (corners.isEmpty()) return emptyList()
        return corners.map { PointF(it.x, it.y) }
    }

    // 🌟 투시 변환(Warp) + 알파/색상 채널 안정성 확보 로직 (LAB 밝기 이식 제거됨)
    private fun applyMaskToMat(mat: Mat, corners: List<PointF>, textureInput: Mat) {
        if (corners.isEmpty()) return
         
        var maskMat: Mat? = null; var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null; var alphaMat: Mat? = null
        var warpedTexture: Mat? = null; var perspectiveMat: Mat? = null
        var preparedTexture: Mat? = null
        
        val matChannels = ArrayList<Mat>()
        val textureChannels = ArrayList<Mat>()

        // 🌟 Alpha 버그 원천 차단: 원본 mat이 4채널(RGBA)인지 기록해두고 RGB로 강제 변환
        var originalWasRgba = false
        if (mat.channels() == 4) {
            originalWasRgba = true
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        }
        
        try {
            val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }
            
            // 1. 다각형 마스크 생성
            maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
            contour = org.opencv.core.MatOfPoint(*pts.toTypedArray())
            Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))
            
            blurredMask = Mat()
            Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)
            
            alphaMat = Mat()
            blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)

            // 2. 4개 대표 코너 추출 (minAreaRect)
            val hullMat2f = MatOfPoint2f(*pts.toTypedArray())
            val minRect = Imgproc.minAreaRect(hullMat2f)
            val rectPts = arrayOf(Point(), Point(), Point(), Point())
            minRect.points(rectPts)
            hullMat2f.release()

            val sortedByY = rectPts.sortedBy { it.y }
            val topTwo = sortedByY.take(2).sortedBy { it.x }
            val bottomTwo = sortedByY.takeLast(2).sortedBy { it.x }
            
            val dstPts = MatOfPoint2f(topTwo[0], topTwo[1], bottomTwo[1], bottomTwo[0])
            val srcPts = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(textureInput.cols().toDouble(), 0.0),
                Point(textureInput.cols().toDouble(), textureInput.rows().toDouble()),
                Point(0.0, textureInput.rows().toDouble())
            )

            // 3. warpPerspective 적용
            perspectiveMat = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            warpedTexture = Mat(mat.size(), mat.type()) 
            preparedTexture = Mat()
            
            // 캐싱된 텍스처도 안전하게 3채널(RGB)로 맞춰서 변환
            if (textureInput.channels() != mat.channels()) {
                if (mat.channels() == 3 && textureInput.channels() == 4) 
                    Imgproc.cvtColor(textureInput, preparedTexture, Imgproc.COLOR_RGBA2RGB)
                else textureInput.copyTo(preparedTexture)
            } else {
                textureInput.copyTo(preparedTexture)
            }

            Imgproc.warpPerspective(preparedTexture, warpedTexture, perspectiveMat, mat.size(), Imgproc.INTER_LINEAR)

            // 4. (삭제됨) LAB 밝기 이식 로직 제거 완료

            // 5. 최종 마스킹 (다각형 크롭 + 3채널 기반 알파 블렌딩)
            Core.split(mat, matChannels)
            Core.split(warpedTexture, textureChannels)
            
            for (i in 0 until 3) {
                var mcF: Mat? = null; var ccF: Mat? = null; var blendedF: Mat? = null
                var invAlpha: Mat? = null; var scalarMat: Mat? = null
                try {
                    mcF = Mat()
                    ccF = Mat()
                    matChannels[i].convertTo(mcF, CvType.CV_32F)
                    textureChannels[i].convertTo(ccF, CvType.CV_32F)
                    
                    blendedF = Mat()
                    Core.multiply(ccF, alphaMat, ccF)
                    invAlpha = Mat()
                    scalarMat = Mat(alphaMat.size(), alphaMat.type(), Scalar(1.0))
                    
                    Core.subtract(scalarMat, alphaMat, invAlpha)
                    Core.multiply(mcF, invAlpha, mcF)
                     
                    Core.add(ccF, mcF, blendedF)
                    blendedF.convertTo(matChannels[i], CvType.CV_8U)
                } finally { 
                    mcF?.release(); ccF?.release(); blendedF?.release()
                    invAlpha?.release(); scalarMat?.release()
                }
            }
            Core.merge(matChannels, mat)

        } finally {
            maskMat?.release(); contour?.release(); blurredMask?.release()
            alphaMat?.release(); warpedTexture?.release(); perspectiveMat?.release()
            preparedTexture?.release()
            matChannels.forEach { it.release() }; textureChannels.forEach { it.release() }
            
            // 🌟 모든 작업이 완료된 후 원본이 RGBA였다면 다시 RGBA로 복구하여 반환 (에지 그림자 버그 해결)
            if (originalWasRgba) {
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2RGBA)
            }
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
    
    override fun onDestroy() {
        orientationEventListener?.disable()
        uiHandler.removeCallbacksAndMessages(null)
        
        synchronized(bitmapLock) { 
             lastCapturedBitmap?.recycle()
             lastCapturedBitmap = null 
        }
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle()
        displayedBitmap = null
        
        // 🌟 캐싱된 텍스처 메모리 해제
        cachedTextureMat?.release()
        cachedTextureMat = null

        cameraExecutor.shutdownNow()
        precomputeExecutor.shutdownNow()
        maskExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object { 
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) 
    }
}
