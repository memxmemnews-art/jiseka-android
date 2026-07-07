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
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
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
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resume 
import kotlin.coroutines.suspendCoroutine 
import kotlin.math.max
import kotlin.math.min

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

    // 💡 화면 고정용 디버그 HUD UI 요소
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
    
    private var isZoomed = false

    private val uiHandler = android.os.Handler(Looper.getMainLooper())
    private val hideGuideTextRunnable = Runnable {
        guideText?.animate()?.alpha(0f)?.setDuration(300)?.withEndAction {
            guideText?.visibility = View.GONE
        }?.start()
    }

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

        setupDebugUI() 
        setupUIListeners()
        setupOrientationListener()
        resetToLiveMode()

        if (allPermissionsGranted()) viewFinder?.post { startCamera() }
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
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

            if (!isZoomed) {
                isZoomed = true
                nativeBackgroundView?.pivotX = uiPoint.x
                nativeBackgroundView?.pivotY = uiPoint.y
                nativeBackgroundView?.animate()
                    ?.scaleX(3f)
                    ?.scaleY(3f)
                    ?.setDuration(300)
                    ?.withEndAction {
                        guideText?.text = "번호판 글자를 터치해주세요"
                        guideText?.alpha = 1f
                        guideText?.visibility = View.VISIBLE
                        uiHandler.removeCallbacks(hideGuideTextRunnable)
                        uiHandler.postDelayed(hideGuideTextRunnable, 3500)
                    }
                    ?.start()
                return@touchDrop 
            }

            val currentSession = captureSessionId.get()
            progressBar?.visibility = View.VISIBLE
            nativeGuideView?.visibility = View.GONE
            guideText?.visibility = View.GONE
            
            precomputeExecutor.execute {
                if (captureSessionId.get() != currentSession) return@execute 
                
                val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
       
                if (safeBitmap != null) {
                    val pX = nativeBackgroundView?.pivotX ?: 0f
                    val pY = nativeBackgroundView?.pivotY ?: 0f
                    val currentScale = nativeBackgroundView?.scaleX ?: 1f
                    
                    val unscaledX = pX + (uiPoint.x - pX) / currentScale
                    val unscaledY = pY + (uiPoint.y - pY) / currentScale

                    val touchCoords = FloatArray(2).apply { 
                        this[0] = unscaledX
                        this[1] = unscaledY 
                    }
                    inverseMatrix.mapPoints(touchCoords)
                    val debugInterceptor = createDebugInterceptor()

                    runMLKitPipeline(safeBitmap, touchCoords[0], touchCoords[1], currentSession, debugInterceptor)

                } else {
                    runOnUiThread { progressBar?.visibility = View.GONE }
                }
            }
        }
    }

    private fun runMLKitPipeline(
        safeBitmap: Bitmap, touchX: Float, touchY: Float,
        currentSession: Int, debugInterceptor: PlateDetectionEngine.DetectionDebugListener
    ) {
        val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())

        val smartCrop = PlateDetectionEngine.prepareSeedCrop(safeBitmap, touchX, touchY, debugInterceptor)
        
        if (smartCrop != null) {
            val image1 = InputImage.fromBitmap(smartCrop.croppedBitmap, 0)
            recognizer.process(image1)
                .addOnSuccessListener { visionText1 ->
                    
                    val localTouchX = touchX - smartCrop.offsetX
                    val localTouchY = touchY - smartCrop.offsetY
                    var closestSymbolBox1 = findClosestSymbolBox(visionText1, localTouchX, localTouchY)

                    if (closestSymbolBox1 != null) {
                        buildFinalWireframe(safeBitmap, smartCrop.offsetX, smartCrop.offsetY, closestSymbolBox1, currentSession, debugInterceptor)
                    } else {
                        runFallbackDumbCrop(recognizer, safeBitmap, touchX, touchY, currentSession, debugInterceptor)
                    }
                }
                .addOnFailureListener {
                    runFallbackDumbCrop(recognizer, safeBitmap, touchX, touchY, currentSession, debugInterceptor)
                }
        } else {
            runFallbackDumbCrop(recognizer, safeBitmap, touchX, touchY, currentSession, debugInterceptor)
        }
    }

    private fun runFallbackDumbCrop(
        recognizer: com.google.mlkit.vision.text.TextRecognizer,
        safeBitmap: Bitmap, touchX: Float, touchY: Float,
        currentSession: Int, debugInterceptor: PlateDetectionEngine.DetectionDebugListener
    ) {
        val dumbCrop = PlateDetectionEngine.prepareDumbCrop(safeBitmap, touchX, touchY, debugInterceptor)
        val image2 = InputImage.fromBitmap(dumbCrop.croppedBitmap, 0)

        recognizer.process(image2)
            .addOnSuccessListener { visionText2 ->
                val localTouchX = touchX - dumbCrop.offsetX
                val localTouchY = touchY - dumbCrop.offsetY
                val closestSymbolBox2 = findClosestSymbolBox(visionText2, localTouchX, localTouchY)
                
                if (closestSymbolBox2 != null) {
                    buildFinalWireframe(safeBitmap, dumbCrop.offsetX, dumbCrop.offsetY, closestSymbolBox2, currentSession, debugInterceptor)
                } else {
                    safeBitmap.recycle()
                    fallbackToManualMode(currentSession, "해당 영역에서 번호판 글자를 찾지 못했습니다. 다시 시도해주세요.")
                }
            }
            .addOnFailureListener {
                safeBitmap.recycle()
                fallbackToManualMode(currentSession, "텍스트 인식 엔진에 문제가 발생했습니다.")
            }
    }

    private fun findClosestSymbolBox(visionText: com.google.mlkit.vision.text.Text, localTouchX: Float, localTouchY: Float): android.graphics.Rect? {
        var closestSymbolBox: android.graphics.Rect? = null
        var minDistance = Float.MAX_VALUE

        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                for (element in line.elements) {
                    for (symbol in element.symbols) {
                        val box = symbol.boundingBox
                        if (box != null) {
                            if (box.contains(localTouchX.toInt(), localTouchY.toInt())) {
                                closestSymbolBox = box
                                minDistance = 0f
                                break
                            }
                            val cx = box.exactCenterX()
                            val cy = box.exactCenterY()
                            val dist = Math.hypot((cx - localTouchX).toDouble(), (cy - localTouchY).toDouble()).toFloat()
                            if (dist < minDistance) {
                                minDistance = dist
                                closestSymbolBox = box
                            }
                        }
                    }
                    if (minDistance == 0f) break
                }
                if (minDistance == 0f) break
            }
            if (minDistance == 0f) break
        }
        return closestSymbolBox
    }

    private fun buildFinalWireframe(
        safeBitmap: Bitmap, offsetX: Int, offsetY: Int, mlKitBox: android.graphics.Rect,
        currentSession: Int, debugInterceptor: PlateDetectionEngine.DetectionDebugListener
    ) {
        lifecycleScope.launch(Dispatchers.Default) {
            if (captureSessionId.get() != currentSession) {
                safeBitmap.recycle()
                return@launch
            }

            val mlKitScanner = object : PlateDetectionEngine.MLKitScanner {
                override suspend fun scanCharacters(bitmap: Bitmap): List<android.graphics.Rect> = suspendCoroutine { continuation ->
                    val image = InputImage.fromBitmap(bitmap, 0)
                    val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
                    
                    recognizer.process(image)
                        .addOnSuccessListener { visionText ->
                            val boxes = mutableListOf<android.graphics.Rect>()
                            
                            for (block in visionText.textBlocks) {
                                for (line in block.lines) {
                                    for (element in line.elements) {
                                        for (symbol in element.symbols) {
                                            symbol.boundingBox?.let { boxes.add(it) }
                                        }
                                    }
                                }
                            }
                            continuation.resume(boxes)
                        }
                        .addOnFailureListener {
                            continuation.resume(emptyList())
                        }
                }
            }

            val targetPolygon = PlateDetectionEngine.processWithMLKitResult(
                safeBitmap, offsetX, offsetY, mlKitBox, mlKitScanner, debugInterceptor
            )

            runOnUiThread {
                if (isFinishing || isDestroyed || captureSessionId.get() != currentSession) {
                    safeBitmap.recycle()
                    return@runOnUiThread
                }

                if (targetPolygon != null && targetPolygon.isNotEmpty()) {
                    triggerInstantMasking(targetPolygon)
                } else {
                    fallbackToManualMode(currentSession, "번호판 세부 조립에 실패했습니다.")
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
        
        isZoomed = false
        nativeBackgroundView?.scaleX = 1f
        nativeBackgroundView?.scaleY = 1f
        nativeBackgroundView?.translationX = 0f
        nativeBackgroundView?.translationY = 0f
        
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
            
            nativeBackgroundView?.animate()?.scaleX(1f)?.scaleY(1f)?.setDuration(300)?.start()
            isZoomed = false
            
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
                            
                            nativeBackgroundView?.animate()?.scaleX(1f)?.scaleY(1f)?.setDuration(300)?.start()
                            isZoomed = false

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
        if (requestCode == 1001 && allPermissionsGranted()) viewFinder?.post { startCamera() }
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
