package com.jiseka.app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.OrientationEventListener
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Button
import android.widget.FrameLayout
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
import androidx.lifecycle.lifecycleScope 
import kotlinx.coroutines.Dispatchers 
import kotlinx.coroutines.launch 
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

// 💡 MediaPipe 패키지 import
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Detection
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
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

    private var debugLatch: CountDownLatch? = null
    private var btnDebugNext: Button? = null
    private var debugHudContainer: LinearLayout? = null
    private var debugHudTitle: TextView? = null
    private var debugHudLogs: TextView? = null

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

    private var cachedTextureMat: Mat? = null
    private val captureSessionId = AtomicInteger(0)

    private val viewMatrix = Matrix()
    private val inverseMatrix = Matrix()
    private var isMatrixReady = false

    private val uiHandler = android.os.Handler(Looper.getMainLooper())
    private val hideGuideTextRunnable = Runnable {
        guideText?.animate()?.alpha(0f)?.setDuration(300)?.withEndAction {
            guideText?.visibility = View.GONE
        }?.start()
    }

    // ⭐️ 커스텀 번호판 탐지 AI 모델 (MediaPipe ObjectDetector)
    private var objectDetector: ObjectDetector? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "엔진 초기화에 실패했습니다. 앱을 다시 실행해주세요.", Toast.LENGTH_LONG).show()
            finish()
            return
        }

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

        initCustomAIModel()
        setupDebugUI() 
        setupUIListeners()
        setupOrientationListener()
        resetToLiveMode()

        if (allPermissionsGranted()) viewFinder?.post { startCamera() }
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    private fun initCustomAIModel() {
        try {
            // 💡 MediaPipe 규격에 맞춘 모델 초기화
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("plate_detector.tflite")
                .build()

            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setMaxResults(3)
                .setScoreThreshold(0.5f)
                .setRunningMode(RunningMode.IMAGE)
                .build()
                
            objectDetector = ObjectDetector.createFromOptions(this, options)
        } catch (e: Exception) {
            Log.e("AI_DEBUG", "AI 모델 초기화 실패", e)
            Toast.makeText(this, "AI 모델을 불러오지 못했습니다.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupDebugUI() {
        btnDebugNext = Button(this).apply {
            text = "다음 단계 확인 ⏭️"
            textSize = 20f
            setBackgroundColor(Color.parseColor("#FF3333"))
            setTextColor(Color.WHITE)
            visibility = View.GONE
            setOnClickListener {
                debugLatch?.countDown() 
            }
        }
        val btnParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            bottomMargin = 50 
        }
        addContentView(btnDebugNext, btnParams)

        debugHudContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#CC000000")) 
            setPadding(40, 40, 40, 40)
            visibility = View.GONE
        }
        
        debugHudTitle = TextView(this).apply {
            setTextColor(Color.YELLOW)
            textSize = 22f
            paint.isFakeBoldText = true
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }
        
        debugHudLogs = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 16f
            setLineSpacing(0f, 1.2f)
            setPadding(0, 20, 0, 0)
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }

        debugHudContainer?.addView(debugHudTitle)
        debugHudContainer?.addView(debugHudLogs)

        val hudParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.CENTER_HORIZONTAL or Gravity.TOP
            topMargin = 100 
        }
        addContentView(debugHudContainer, hudParams)
    }

    private fun loadTextureSafely() {
        val rawBitmap = BitmapFactory.decodeResource(resources, R.drawable.plate_texture)
        if (rawBitmap != null) {
            cachedTextureMat = Mat()
            Utils.bitmapToMat(rawBitmap, cachedTextureMat!!)
            if (cachedTextureMat!!.channels() == 4) {
                Imgproc.cvtColor(cachedTextureMat!!, cachedTextureMat!!, Imgproc.COLOR_RGBA2RGB)
            }
            rawBitmap.recycle()
        } else {
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

        nativeGuideView?.onTouchPointListener = touchDrop@{ uiPoint ->
            if (!isMatrixReady) return@touchDrop

            val currentSession = captureSessionId.get()
            progressBar?.visibility = View.VISIBLE
            nativeGuideView?.visibility = View.GONE
            guideText?.visibility = View.GONE
            
            precomputeExecutor.execute {
                if (captureSessionId.get() != currentSession) return@execute 
                
                val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
       
                if (safeBitmap != null) {
                    val touchCoords = FloatArray(2).apply { 
                        this[0] = uiPoint.x
                        this[1] = uiPoint.y 
                    }
                    inverseMatrix.mapPoints(touchCoords)
                    val debugInterceptor = createDebugInterceptor()

                    runAIPipeline(safeBitmap, touchCoords[0], touchCoords[1], currentSession, debugInterceptor)
                } else {
                    runOnUiThread { progressBar?.visibility = View.GONE }
                }
            }
        }
    }

    // ⭐️ 터치된 지점 주변만 크롭한 뒤, "UX 최우선(터치 포함 여부)"으로 번호판을 찾는 함수
    private fun runAIPipeline(
        safeBitmap: Bitmap, touchX: Float, touchY: Float,
        currentSession: Int, debugInterceptor: PlateDetectionEngine.DetectionDebugListener
    ) {
        if (objectDetector == null) {
            fallbackToManualMode(currentSession, "AI 모델이 로드되지 않았습니다.")
            safeBitmap.recycle()
            return
        }

        // 1. 터치 위치 주변을 크롭 (가로 25%, 세로 10%)
        val localCrop = PlateDetectionEngine.prepareWideCrop(safeBitmap, touchX, touchY)
        
        // 2. 크롭된 이미지를 MediaPipe MPImage로 변환
        val mpImage = BitmapImageBuilder(localCrop.croppedBitmap).build()
        
        // 3. AI 모델 추론 실행 (작은 영역 안에서 번호판 탐색)
        val results = objectDetector?.detect(mpImage)
        val detections = results?.detections() ?: emptyList()

        // 크롭된 영역 기준의 터치 좌표 (크롭 영역의 정중앙)
        val localTouchX = localCrop.croppedBitmap.width / 2f
        val localTouchY = localCrop.croppedBitmap.height / 2f

        var bestDetection: Detection? = null

        if (detections.isNotEmpty()) {
            // [1순위 필터링] 터치점(크롭의 중앙)이 바운딩 박스 '내부'에 포함되는 것들 찾기
            val containingDetections = detections.filter { 
                it.boundingBox().contains(localTouchX, localTouchY) 
            }

            if (containingDetections.isNotEmpty()) {
                // 포함된 박스가 1개 이상이면, 그 중에서 가장 신뢰도가 높은 것을 선택
                bestDetection = containingDetections.maxByOrNull { it.categories().firstOrNull()?.score() ?: 0f }
            } else {
                // [2순위 필터링] 포함된 박스가 하나도 없다면, 중심점이 터치점과 가장 '가까운' 박스 탐색
                var minDistance = Float.MAX_VALUE
                // 최대 허용 거리 제한 (크롭 영역 짧은 변의 30%를 넘어가면 엉뚱한 물체로 간주)
                val maxAllowedDistance = Math.min(localCrop.croppedBitmap.width, localCrop.croppedBitmap.height) * 0.3f
                
                for (detection in detections) {
                    val cx = detection.boundingBox().centerX()
                    val cy = detection.boundingBox().centerY()
                    val dist = Math.hypot((cx - localTouchX).toDouble(), (cy - localTouchY).toDouble()).toFloat()
                    
                    if (dist < minDistance && dist < maxAllowedDistance) {
                        minDistance = dist
                        bestDetection = detection
                    }
                }
            }
        }

        if (bestDetection != null) {
            val localBox = bestDetection.boundingBox()

            // 4. 로컬 이미지 기준 Bounding Box를 전체 원본 화면(Global) 좌표로 복원
            val globalLineBox = android.graphics.Rect(
                localCrop.offsetX + localBox.left.toInt(),
                localCrop.offsetY + localBox.top.toInt(),
                localCrop.offsetX + localBox.right.toInt(),
                localCrop.offsetY + localBox.bottom.toInt()
            )
            
            localCrop.croppedBitmap.recycle()
            
            // 5. 글로벌 Box를 바탕으로 정밀 외곽선(Wireframe) 추출 단계로 이동
            buildFinalWireframe(safeBitmap, globalLineBox, currentSession, debugInterceptor)
        } else {
            // 터치 영역 주변에서 번호판을 못 찾았을 경우
            val debugBmp = localCrop.croppedBitmap.copy(Bitmap.Config.ARGB_8888, true)
            debugInterceptor.pauseAndShowStep(
                "디버그 1단계: [FAIL] AI 모델 탐색 실패", debugBmp,
                "[FAIL] 터치 영역 내 번호판 없음",
                listOf(
                    "-> 원인: 터치된 구역 안에서 AI가 번호판을 감지하지 못했습니다.",
                    "-> 조치: 번호판 안쪽을 정확히 다시 터치해주세요."
                )
            )
            localCrop.croppedBitmap.recycle()
            safeBitmap.recycle()
            fallbackToManualMode(currentSession, "해당 위치 주변에서 번호판을 찾지 못했습니다.")
        }
    }

    private fun buildFinalWireframe(
        safeBitmap: Bitmap, aiGlobalBox: android.graphics.Rect,
        currentSession: Int, debugInterceptor: PlateDetectionEngine.DetectionDebugListener
    ) {
        lifecycleScope.launch(Dispatchers.Default) {
            if (captureSessionId.get() != currentSession) {
                safeBitmap.recycle()
                return@launch
            }

            val targetPolygon = PlateDetectionEngine.processWithMLKitResult(
                safeBitmap, aiGlobalBox, debugInterceptor
            )

            runOnUiThread {
                if (isFinishing || isDestroyed || captureSessionId.get() != currentSession) {
                    safeBitmap.recycle()
                    return@runOnUiThread
                }

                if (targetPolygon != null && targetPolygon.isNotEmpty()) {
                    triggerInstantMasking(targetPolygon)
                } else {
                    fallbackToManualMode(currentSession, "번호판 기하학 조립에 실패했습니다.")
                }
                safeBitmap.recycle() 
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
        
        debugLatch?.countDown() 
        btnDebugNext?.visibility = View.GONE
        debugHudContainer?.visibility = View.GONE
       
        btnCapture?.isEnabled = true
        nativeBackgroundView?.setImageDrawable(null)
    
        displayedBitmap?.recycle()
        displayedBitmap = null

        synchronized(bitmapLock) { 
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null 
        }
        
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
                        val rawBitmap = imageProxy.toBitmap()
                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees.toFloat()
                        val matrix = Matrix().apply { postRotate(rotationDegrees) }
                        val uprightBitmap = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
  
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

    private fun fallbackToManualMode(sessionId: Int, message: String) {
        runOnUiThread {
            if (captureSessionId.get() != sessionId) return@runOnUiThread
            
            progressBar?.visibility = View.GONE
            nativeGuideView?.visibility = View.VISIBLE
            Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
        }
    }

    private fun createDebugInterceptor(): PlateDetectionEngine.DetectionDebugListener {
        return object : PlateDetectionEngine.DetectionDebugListener {
            override fun pauseAndShowStep(stageName: String, debugBitmap: Bitmap, title: String, logs: List<String>) {
                debugLatch = CountDownLatch(1)
                
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    
                    nativeBackgroundView?.setImageBitmap(debugBitmap)
                    
                    debugHudTitle?.text = title
                    val logText = StringBuilder()
                    for (log in logs) {
                        logText.append(log).append("\n")
                    }
                    debugHudLogs?.text = logText.toString()
                    debugHudContainer?.visibility = View.VISIBLE
                    
                    btnDebugNext?.visibility = View.VISIBLE 
                    progressBar?.visibility = View.GONE 
                }
                
                debugLatch?.await() 
                
                runOnUiThread { 
                    btnDebugNext?.visibility = View.GONE 
                    progressBar?.visibility = View.VISIBLE 
                    debugHudContainer?.visibility = View.GONE
                }
            }
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
                
                guideText?.text = "번호판을 터치해주세요"
                guideText?.paint?.isFakeBoldText = true 
                guideText?.textSize = 20f 
                guideText?.alpha = 1f
                guideText?.visibility = View.VISIBLE
                
                uiHandler.removeCallbacks(hideGuideTextRunnable) 
                uiHandler.postDelayed(hideGuideTextRunnable, 3500) 
                return true 
            }
        })
        bgView.invalidate() 
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
                        val resultMat = Mat()
                        Utils.bitmapToMat(safeTargetBitmap, resultMat)

                        cachedTextureMat?.let { texture ->
                            applyMaskToMat(resultMat, targetCandidate, texture)
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
                                uiHandler.postDelayed({ 
                                    if (!bmp.isRecycled) bmp.recycle() 
                                }, 500)
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
        } catch (e: Exception) {
            Log.e("CAMERA_DEBUG", "Failed to execute masking task", e)
            progressBar?.visibility = View.GONE
            nativeGuideView?.visibility = View.VISIBLE
        }
    }

    private fun applyMaskToMat(mat: Mat, corners: List<ImmutablePoint>, textureInput: Mat) {
        if (corners.size != 4) return

        var maskMat: Mat? = null
        var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null
        var alphaMat: Mat? = null
        var preparedTexture: Mat? = null
        var warpedTexture: Mat? = null

        var originalWasRgba = false
        if (mat.channels() == 4) {
            originalWasRgba = true
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        }

        try {
            val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }

            maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
            contour = org.opencv.core.MatOfPoint(*pts.toTypedArray())
            Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))

            blurredMask = Mat()
            Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)

            alphaMat = Mat()
            blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)

            preparedTexture = Mat()
            if (textureInput.channels() != mat.channels()) {
                if (mat.channels() == 3 && textureInput.channels() == 4) {
                    Imgproc.cvtColor(textureInput, preparedTexture, Imgproc.COLOR_RGBA2RGB)
                } else {
                    textureInput.copyTo(preparedTexture)
                }
            } else {
                textureInput.copyTo(preparedTexture)
            }

            warpedTexture = Mat.zeros(mat.size(), mat.type())
            val srcPts = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(preparedTexture.cols().toDouble(), 0.0),
                Point(preparedTexture.cols().toDouble(), preparedTexture.rows().toDouble()),
                Point(0.0, preparedTexture.rows().toDouble())
            )
            
            val dstPts = MatOfPoint2f(*pts.toTypedArray())

            val perspectiveMat = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(preparedTexture, warpedTexture, perspectiveMat, mat.size(), Imgproc.INTER_LINEAR)

            srcPts.release()
            dstPts.release()
            perspectiveMat.release()

            val matChannels = ArrayList<Mat>()
            val textureChannels = ArrayList<Mat>()
            Core.split(mat, matChannels)
            Core.split(warpedTexture, textureChannels)

            for (i in 0 until 3) {
                var mcF: Mat? = null; var ccF: Mat? = null; var blendedF: Mat? = null
                var invAlpha: Mat? = null; var scalarMat: Mat? = null
                try {
                    mcF = Mat(); ccF = Mat()
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
            matChannels.forEach { it.release() }
            textureChannels.forEach { it.release() }

        } finally {
            maskMat?.release()
            contour?.release()
            blurredMask?.release()
            alphaMat?.release()
            preparedTexture?.release()
            warpedTexture?.release()

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
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) viewFinder?.post { startCamera() }
    }
    
    override fun onDestroy() {
        orientationEventListener?.disable()
        uiHandler.removeCallbacksAndMessages(null)
        
        debugLatch?.countDown() 

        synchronized(bitmapLock) { 
             lastCapturedBitmap?.recycle()
             lastCapturedBitmap = null 
        }
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle()
        displayedBitmap = null
        
        cachedTextureMat?.release()
        cachedTextureMat = null

        objectDetector?.close() // 모델 메모리 해제

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
