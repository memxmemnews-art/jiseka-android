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

// 💡 순수 TFLite Interpreter 엔진
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

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

    // ⭐️ 순수 TFLite Interpreter 변수 및 동적 텐서 매핑 인덱스
    private var tflite: Interpreter? = null
    private var inputWidth = 256
    private var inputHeight = 256
    private var isQuantized = false
    
    private var outIdxBoxes = -1
    private var outIdxScores = -1
    private var outIdxClasses = -1
    private var outIdxNum = -1

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
            Log.d("AI_DEBUG", "--- RetinaNet 커스텀 Interpreter 어댑터 로드 ---")

            val assetFileDescriptor = assets.openFd("plate_detector.tflite")
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val mappedByteBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFileDescriptor.startOffset,
                assetFileDescriptor.declaredLength
            )

            val options = Interpreter.Options().apply { setNumThreads(4) }
            tflite = Interpreter(mappedByteBuffer, options)

            // 1. 입력 텐서 분석
            val inputTensor = tflite!!.getInputTensor(0)
            val shape = inputTensor.shape()
            inputHeight = shape[1] // 256
            inputWidth = shape[2] // 256
            isQuantized = (inputTensor.dataType() == DataType.UINT8)
            Log.d("AI_DEBUG", "Input: ${inputWidth}x${inputHeight}, Type: ${inputTensor.dataType()}")

            // 2. 14개 출력 텐서 중 '최종 4개' 동적 스캔 및 매핑
            for (i in 0 until tflite!!.outputTensorCount) {
                val outTensor = tflite!!.getOutputTensor(i)
                val outShape = outTensor.shape()
                val outType = outTensor.dataType()
                
                // Boxes: [1, 100, 4] FLOAT32
                if (outShape.contentEquals(intArrayOf(1, 100, 4)) && outType == DataType.FLOAT32) {
                    outIdxBoxes = i
                } 
                // Scores: [1, 100] FLOAT32
                else if (outShape.contentEquals(intArrayOf(1, 100)) && outType == DataType.FLOAT32) {
                    outIdxScores = i
                }
                // Classes: [1, 100] INT32 (일부 모델은 Float32일 수도 있으나 명시된 조건 우선)
                else if (outShape.contentEquals(intArrayOf(1, 100)) && (outType == DataType.INT32 || outType == DataType.FLOAT32)) {
                    // 동일한 [1, 100] 배열이 두 개일 수 있으므로 Scores가 아니면 Classes로 할당
                    if (outIdxScores != i) outIdxClasses = i
                }
                // Num: [1] INT32 또는 FLOAT32
                else if (outShape.contentEquals(intArrayOf(1))) {
                    outIdxNum = i
                }
            }

            Log.d("AI_DEBUG", "텐서 매핑 결과 - Boxes:$outIdxBoxes, Scores:$outIdxScores, Classes:$outIdxClasses, Num:$outIdxNum")

            if (outIdxBoxes == -1 || outIdxScores == -1) {
                throw Exception("모델 구조 분석 실패: 최종 Detection 텐서(Box, Score)를 찾을 수 없습니다.")
            }

            Log.d("AI_DEBUG", "--- 어댑터 로드 성공 ---")
            
        } catch (e: Throwable) {
            tflite = null
            Log.e("AI_DEBUG", "Interpreter 초기화 실패", e)
            runOnUiThread {
                android.app.AlertDialog.Builder(this)
                    .setTitle("AI 로드 에러")
                    .setMessage(e.message)
                    .setPositiveButton("확인", null)
                    .setCancelable(false)
                    .show()
            }
        }
    }

    // 💡 Float32 입력을 위한 변환
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val bytesPerChannel = if (isQuantized) 1 else 4
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputWidth * inputHeight * 3 * bytesPerChannel)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        
        var pixel = 0
        for (i in 0 until inputHeight) {
            for (j in 0 until inputWidth) {
                val valInt = intValues[pixel++]
                val r = (valInt shr 16) and 0xFF
                val g = (valInt shr 8) and 0xFF
                val b = valInt and 0xFF

                if (isQuantized) {
                    byteBuffer.put(r.toByte()); byteBuffer.put(g.toByte()); byteBuffer.put(b.toByte())
                } else {
                    // RetinaNet 기본 정규화 (0.0 ~ 1.0)
                    byteBuffer.putFloat(r / 255.0f)
                    byteBuffer.putFloat(g / 255.0f)
                    byteBuffer.putFloat(b / 255.0f)
                }
            }
        }
        return byteBuffer
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

    private fun runAIPipeline(
        safeBitmap: Bitmap, touchX: Float, touchY: Float,
        currentSession: Int, debugInterceptor: PlateDetectionEngine.DetectionDebugListener
    ) {
        if (tflite == null) {
            fallbackToManualMode(currentSession, "AI 모델이 로드되지 않았습니다.")
            safeBitmap.recycle()
            return
        }

        val localCrop = PlateDetectionEngine.prepareWideCrop(safeBitmap, touchX, touchY)
        val resizedBitmap = Bitmap.createScaledBitmap(localCrop.croppedBitmap, inputWidth, inputHeight, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
        
        // 💡 14개 출력 중 매핑된 '4개 텐서 배열'을 담을 빈 그릇 준비
        val outBoxes = Array(1) { Array(100) { FloatArray(4) } }
        val outScores = Array(1) { FloatArray(100) }
        
        // Classes와 Num은 DataType에 따라 Int 또는 Float 배열이 필요할 수 있으나,
        // 현재 추론(Score > 0.4 판별)에는 Box와 Score만 사용하므로 해당 배열만 매핑하여 안전하게 뽑아냅니다.
        val outputs = mutableMapOf<Int, Any>()
        outputs[outIdxBoxes] = outBoxes
        outputs[outIdxScores] = outScores
        // (필요 시 classes, num 배열도 추가 가능하지만, Box와 Score로 필터링은 충분합니다)
        
        Log.d("AI_DEBUG", "AI 추론(runForMultipleInputsOutputs) 시작")

        try {
            tflite?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        } catch (e: Throwable) {
            Log.e("AI_DEBUG", "AI 연산 실패", e)
            runOnUiThread { Toast.makeText(this@MainActivity, "AI 연산 오류 발생:\n${e.message}", Toast.LENGTH_LONG).show() }
            localCrop.croppedBitmap.recycle()
            safeBitmap.recycle()
            fallbackToManualMode(currentSession, "AI 추론 중 오류가 발생했습니다.")
            return
        }

        val boxes = outBoxes[0]
        val scores = outScores[0]

        val localTouchX = localCrop.croppedBitmap.width / 2f
        val localTouchY = localCrop.croppedBitmap.height / 2f

        var bestBoxRect: android.graphics.RectF? = null
        var maxScore = -1f
        var minDistance = Float.MAX_VALUE
        val maxAllowedDistance = Math.min(localCrop.croppedBitmap.width, localCrop.croppedBitmap.height) * 0.3f
        
        // 100개의 후보 중 터치 위치 기반 최적의 박스 스캔
        for (i in 0 until 100) {
            val score = scores[i]
            if (score < 0.4f) continue // 정확도 낮은 후보 즉시 필터링
            
            // 모델의 출력 형태 [ymin, xmin, ymax, xmax] (정규화된 0.0 ~ 1.0 비율)
            val ymin = boxes[i][0] * localCrop.croppedBitmap.height
            val xmin = boxes[i][1] * localCrop.croppedBitmap.width
            val ymax = boxes[i][2] * localCrop.croppedBitmap.height
            val xmax = boxes[i][3] * localCrop.croppedBitmap.width
            
            val rect = android.graphics.RectF(xmin, ymin, xmax, ymax)

            if (rect.contains(localTouchX, localTouchY)) {
                if (score > maxScore) {
                    maxScore = score
                    bestBoxRect = rect
                }
            } else if (bestBoxRect == null) {
                val cx = rect.centerX()
                val cy = rect.centerY()
                val dist = Math.hypot((cx - localTouchX).toDouble(), (cy - localTouchY).toDouble()).toFloat()
                if (dist < minDistance && dist < maxAllowedDistance) {
                    minDistance = dist
                    bestBoxRect = rect
                }
            }
        }

        if (bestBoxRect != null) {
            Log.d("AI_DEBUG", "최적 번호판 박스 발견 - Score: $maxScore, Box: $bestBoxRect")
            
            val globalLineBox = android.graphics.Rect(
                localCrop.offsetX + bestBoxRect.left.toInt(),
                localCrop.offsetY + bestBoxRect.top.toInt(),
                localCrop.offsetX + bestBoxRect.right.toInt(),
                localCrop.offsetY + bestBoxRect.bottom.toInt()
            )
            
            localCrop.croppedBitmap.recycle()
            buildFinalWireframe(safeBitmap, globalLineBox, currentSession, debugInterceptor)
        } else {
            val debugBmp = localCrop.croppedBitmap.copy(Bitmap.Config.ARGB_8888, true)
            debugInterceptor.pauseAndShowStep(
                "디버그 1단계: [FAIL] AI 모델 탐색 실패", debugBmp,
                "[FAIL] 터치 영역 내 번호판 없음",
                listOf(
                    "-> 원인: 터치된 구역 안에서 AI가 유효한 번호판 박스(스코어 > 0.4)를 찾지 못했습니다.",
                    "-> 조치: 번호판 중앙을 다시 터치해주세요."
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

        // 💡 원시 엔진 메모리 해제
        tflite?.close()
        tflite = null

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
