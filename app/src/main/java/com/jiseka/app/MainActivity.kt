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
import android.widget.ScrollView
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

    // ⭐️ AI 디버그 패널 UI 변수
    private var debugLatch: CountDownLatch? = null
    private var btnDebugNext: Button? = null
    private var debugHudContainer: LinearLayout? = null
    private var debugHudTitle: TextView? = null
    private var debugHudLogs: TextView? = null
    private var debugImageView: ImageView? = null
    private val debugLogBuilder = java.lang.StringBuilder()

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

    // ============================================================
    // TFLite - plate_detector.tflite 실제 구조에 맞춘 설정
    // ============================================================
    private var tflite: Interpreter? = null
    private var inputWidth = 256
    private var inputHeight = 256
    private var inputType: DataType = DataType.FLOAT32

    private var outIdxBoxes = -1
    private var outIdxScores = -1
    private var outIdxClasses = -1
    private var outIdxNum = -1
    private var maxDetections = 100

    private val tfliteLock = Any()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        checkPreviousCrash() // ⭐️ 크래시 기록 확인

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

    // ============================================================
    // ⭐️ SharedPreferences 크래시 추적 로직
    // ============================================================
    private fun checkPreviousCrash() {
        val prefs = getSharedPreferences("JisekaCrashLog", MODE_PRIVATE)
        val lastStage = prefs.getString("LAST_STAGE", "NONE")
        
        if (lastStage != "NONE" && lastStage != "7_SUCCESS") {
            android.app.AlertDialog.Builder(this)
                .setTitle("🚨 지난 실행 강제 종료 감지")
                .setMessage("이전 실행에서 AI 연산 도중 앱이 종료되었습니다.\n\n마지막 기록 단계:\n$lastStage\n\n이 지점에서 TFLite Native Crash가 발생했을 확률이 99%입니다.")
                .setPositiveButton("확인", null)
                .show()
        }
        
        // 확인 후 초기화
        prefs.edit().putString("LAST_STAGE", "NONE").commit()
    }

    private fun saveCrashStage(stage: String) {
        // commit()을 사용해 동기식으로 확실히 기록한 뒤 다음 코드로 넘어감
        getSharedPreferences("JisekaCrashLog", MODE_PRIVATE)
            .edit()
            .putString("LAST_STAGE", stage)
            .commit()
    }

    // ============================================================
    // 모델 초기화
    // ============================================================
    private fun initCustomAIModel() {
        try {
            Log.d("AI_DEBUG", "========================================")
            Log.d("AI_DEBUG", "TFLite 모델 초기화 시작")

            val assetFileDescriptor = assets.openFd("plate_detector.tflite")
            val mappedByteBuffer = FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
                val fileChannel = inputStream.channel
                fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
            }
            assetFileDescriptor.close()

            val options = Interpreter.Options().apply {
                // ⭐️ [크래시 디버그용] 스레드 1개로 제한
                setNumThreads(1)
            }

            val interpreter = Interpreter(mappedByteBuffer, options)

            // 1. INPUT 검사
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val actualInputType = inputTensor.dataType()

            Log.d("AI_DEBUG", "INPUT shape=${inputShape.contentToString()}, type=$actualInputType")

            if (!inputShape.contentEquals(intArrayOf(1, 256, 256, 3))) {
                interpreter.close()
                throw IllegalStateException("예상하지 못한 입력 shape: ${inputShape.contentToString()}")
            }
            if (actualInputType != DataType.FLOAT32) {
                interpreter.close()
                throw IllegalStateException("예상하지 못한 입력 타입: $actualInputType (FLOAT32 필요)")
            }

            inputHeight = inputShape[1]
            inputWidth = inputShape[2]
            inputType = actualInputType

            // 2. OUTPUT 검사
            outIdxBoxes = -1
            outIdxScores = -1
            outIdxClasses = -1
            outIdxNum = -1

            Log.d("AI_DEBUG", "OUTPUT Tensor count=${interpreter.outputTensorCount}")

            for (i in 0 until interpreter.outputTensorCount) {
                val tensor = interpreter.getOutputTensor(i)
                val shape = tensor.shape()
                val type = tensor.dataType()

                Log.d("AI_DEBUG", "OUTPUT[$i] shape=${shape.contentToString()}, type=$type, name=${tensor.name()}")

                if (shape.size == 3 && shape[0] == 1 && shape[2] == 4 && type == DataType.FLOAT32) {
                    outIdxBoxes = i
                    maxDetections = shape[1]
                } else if (shape.contentEquals(intArrayOf(1, maxDetections)) && type == DataType.FLOAT32) {
                    outIdxScores = i
                } else if (shape.contentEquals(intArrayOf(1, maxDetections)) && type == DataType.INT32) {
                    outIdxClasses = i
                } else if (shape.contentEquals(intArrayOf(1)) && type == DataType.INT32) {
                    outIdxNum = i
                }
            }

            if (outIdxBoxes == -1) { interpreter.close(); throw java.lang.IllegalStateException("detection_boxes Tensor를 찾지 못했습니다.") }
            if (outIdxScores == -1) { interpreter.close(); throw java.lang.IllegalStateException("detection_scores Tensor를 찾지 못했습니다.") }
            if (outIdxClasses == -1) { interpreter.close(); throw java.lang.IllegalStateException("detection_classes Tensor를 찾지 못했습니다.") }
            if (outIdxNum == -1) { interpreter.close(); throw java.lang.IllegalStateException("num_detections Tensor를 찾지 못했습니다.") }

            synchronized(tfliteLock) {
                tflite?.close()
                tflite = interpreter
            }

            Log.d("AI_DEBUG", "========================================")
            Log.d("AI_DEBUG", "TFLite 초기화 성공 (Single Thread)")
            Log.d("AI_DEBUG", "boxes index = $outIdxBoxes, scores index = $outIdxScores")
            Log.d("AI_DEBUG", "classes index = $outIdxClasses, num index = $outIdxNum")
            Log.d("AI_DEBUG", "maxDetections = $maxDetections")
            Log.d("AI_DEBUG", "========================================")

        } catch (e: Throwable) {
            synchronized(tfliteLock) {
                tflite?.close()
                tflite = null
            }
            Log.e("AI_DEBUG", "TFLite 초기화 실패", e)
            runOnUiThread {
                android.app.AlertDialog.Builder(this)
                    .setTitle("AI 로드 에러")
                    .setMessage("plate_detector.tflite 초기화에 실패했습니다.\n\n${e.message}")
                    .setPositiveButton("확인", null)
                    .setCancelable(false)
                    .show()
            }
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
            throw IllegalArgumentException("입력 Bitmap 크기 오류: ${bitmap.width}x${bitmap.height}, 필요: ${inputWidth}x${inputHeight}")
        }

        val bufferSize = inputWidth * inputHeight * 3 * 4   // FLOAT32 = 4 bytes
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputWidth * inputHeight)

        bitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        var pixelIndex = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = intValues[pixelIndex++]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                byteBuffer.putFloat(r / 255.0f)
                byteBuffer.putFloat(g / 255.0f)
                byteBuffer.putFloat(b / 255.0f)
            }
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    // ============================================================
    // ⭐️ 진화된 디버그 UI 생성
    // ============================================================
    private fun setupDebugUI() {
        debugHudContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#EE000000")) 
            setPadding(40, 40, 40, 40)
            visibility = View.GONE
        }
        
        debugHudTitle = TextView(this).apply {
            text = "━━━━━━━━━━━━━━━━━━━━━━\nAI DEBUG MODE\n━━━━━━━━━━━━━━━━━━━━━━"
            setTextColor(Color.CYAN)
            textSize = 18f
            paint.isFakeBoldText = true
        }
        
        debugImageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(400, 400).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                topMargin = 20
                bottomMargin = 20
            }
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.DKGRAY)
        }

        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 
                0, 
                1.0f
            )
        }

        debugHudLogs = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 14f
            setLineSpacing(0f, 1.2f)
        }
        scrollView.addView(debugHudLogs)

        btnDebugNext = Button(this).apply {
            text = "▶ TFLite 실행"
            textSize = 18f
            setBackgroundColor(Color.parseColor("#FF3333"))
            setTextColor(Color.WHITE)
            setPadding(0, 20, 0, 20)
            visibility = View.GONE
            setOnClickListener { debugLatch?.countDown() }
        }

        debugHudContainer?.addView(debugHudTitle)
        debugHudContainer?.addView(debugImageView)
        debugHudContainer?.addView(scrollView)
        debugHudContainer?.addView(btnDebugNext)

        val hudParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        addContentView(debugHudContainer, hudParams)
    }

    // 디버그 진행 헬퍼
    private fun updateDebugStep(logText: String, image: Bitmap? = null, waitForClick: Boolean = false) {
        debugLogBuilder.append("\n").append(logText)
        val currentLog = debugLogBuilder.toString()
        
        if (waitForClick) {
            debugLatch = CountDownLatch(1)
        }

        runOnUiThread {
            debugHudContainer?.visibility = View.VISIBLE
            debugHudLogs?.text = currentLog
            image?.let { debugImageView?.setImageBitmap(it) }
            
            if (waitForClick) {
                btnDebugNext?.text = "▶ TFLite 실행 (터치 대기중)"
                btnDebugNext?.visibility = View.VISIBLE
            } else {
                btnDebugNext?.visibility = View.GONE
            }
        }

        if (waitForClick) {
            debugLatch?.await() // 클릭할 때까지 백그라운드 스레드 일시정지
            runOnUiThread { btnDebugNext?.visibility = View.GONE }
        }
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
                        this[0] = uiPoint.x; this[1] = uiPoint.y 
                    }
                    inverseMatrix.mapPoints(touchCoords)
                    val debugInterceptor = createDebugInterceptor()

                    // 초기화
                    debugLogBuilder.clear()
                    debugLogBuilder.append("[1] 터치 감지됨: X=${touchCoords[0].toInt()}, Y=${touchCoords[1].toInt()}")

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

        saveCrashStage("1_CROP_START")
        val localCrop = PlateDetectionEngine.prepareWideCrop(safeBitmap, touchX, touchY)
        updateDebugStep("[2] Crop 성공: ${localCrop.croppedBitmap.width} x ${localCrop.croppedBitmap.height}", localCrop.croppedBitmap)

        var resizedBitmap: Bitmap? = null

        try {
            if (localCrop.croppedBitmap.isRecycled) {
                throw IllegalStateException("AI 입력 Crop Bitmap이 이미 recycle되었습니다.")
            }

            saveCrashStage("2_RESIZE")
            resizedBitmap = Bitmap.createScaledBitmap(localCrop.croppedBitmap, inputWidth, inputHeight, true)
            updateDebugStep("[3] Resize 성공: 256x256", resizedBitmap)

            saveCrashStage("3_INPUT_BUFFER")
            val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
            updateDebugStep("[4] FLOAT32 입력 생성: ${inputBuffer.capacity()} bytes")

            // ------------------------------------------------------------
            // ⭐️ 4. 출력 배열 생성 (Direct ByteBuffer 대신 일반 배열 사용)
            // ------------------------------------------------------------
            saveCrashStage("4_OUTPUT_ARRAY_PREP")
            
            val outBoxes = Array(1) { Array(maxDetections) { FloatArray(4) } }
            val outScores = Array(1) { FloatArray(maxDetections) }
            val outClasses = Array(1) { IntArray(maxDetections) }
            val outNum = IntArray(1)

            val outputs = HashMap<Int, Any>()
            outputs[outIdxBoxes] = outBoxes
            outputs[outIdxScores] = outScores
            outputs[outIdxClasses] = outClasses
            outputs[outIdxNum] = outNum

            updateDebugStep("[5] Output 배열 매핑 완료")

            // ------------------------------------------------------------
            // ⭐️ 5. 실행 전 정지 (UI 버튼 클릭 대기)
            // ------------------------------------------------------------
            saveCrashStage("5_BEFORE_TFLITE_RUN")
            updateDebugStep("[6] TFLite 실행 준비 완료.\n버튼을 누르면 추론을 시작합니다.", waitForClick = true)
            

            // ------------------------------------------------------------
            // ⭐️ 6. TFLite 실행 (여기서 죽으면 이 단계가 저장됨)
            // ------------------------------------------------------------
            saveCrashStage("6_TFLITE_RUNNING")
            Log.e("AI_CRASH_TEST", "F: TFLite run 직전")

            synchronized(tfliteLock) {
                val interpreter = tflite ?: throw IllegalStateException("TFLite Interpreter가 없습니다.")
                interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            }

            Log.e("AI_CRASH_TEST", "G: TFLite run 완료")
            saveCrashStage("7_SUCCESS")
            updateDebugStep("[7] TFLite 실행 성공!")

            // ------------------------------------------------------------
            // 7. 결과 해석
            // ------------------------------------------------------------
            val reportedNum = outNum[0]
            val detectionCount = reportedNum.coerceIn(0, maxDetections)
            Log.d("AI_DEBUG", "num_detections=$reportedNum (사용할 개수: $detectionCount)")

            val localTouchX = localCrop.croppedBitmap.width / 2f
            val localTouchY = localCrop.croppedBitmap.height / 2f

            var bestBoxRect: android.graphics.RectF? = null
            var bestScore = -1f
            var minDistance = Float.MAX_VALUE
            val maxAllowedDistance = min(localCrop.croppedBitmap.width, localCrop.croppedBitmap.height) * 0.3f

            for (i in 0 until detectionCount) {
                val score = outScores[0][i]
                if (!score.isFinite() || score < 0.4f) continue

                val classId = outClasses[0][i]
                val ymin = outBoxes[0][i][0]
                val xmin = outBoxes[0][i][1]
                val ymax = outBoxes[0][i][2]
                val xmax = outBoxes[0][i][3]

                if (!ymin.isFinite() || !xmin.isFinite() || !ymax.isFinite() || !xmax.isFinite()) continue

                val rect = android.graphics.RectF(
                    xmin * localCrop.croppedBitmap.width,
                    ymin * localCrop.croppedBitmap.height,
                    xmax * localCrop.croppedBitmap.width,
                    ymax * localCrop.croppedBitmap.height
                )

                if (rect.width() <= 0f || rect.height() <= 0f) continue

                rect.left = rect.left.coerceIn(0f, localCrop.croppedBitmap.width.toFloat())
                rect.right = rect.right.coerceIn(0f, localCrop.croppedBitmap.width.toFloat())
                rect.top = rect.top.coerceIn(0f, localCrop.croppedBitmap.height.toFloat())
                rect.bottom = rect.bottom.coerceIn(0f, localCrop.croppedBitmap.height.toFloat())

                if (rect.width() <= 0f || rect.height() <= 0f) continue

                if (rect.contains(localTouchX, localTouchY)) {
                    if (score > bestScore) {
                        bestScore = score
                        bestBoxRect = rect
                    }
                } else if (bestBoxRect == null) {
                    val cx = rect.centerX()
                    val cy = rect.centerY()
                    val dist = Math.hypot((cx - localTouchX).toDouble(), (cy - localTouchY).toDouble()).toFloat()

                    if (dist < minDistance && dist < maxAllowedDistance) {
                        minDistance = dist
                        bestBoxRect = rect
                        bestScore = score
                    }
                }
            }

            if (resizedBitmap != null && resizedBitmap !== localCrop.croppedBitmap) {
                resizedBitmap.recycle()
                resizedBitmap = null
            }

            if (bestBoxRect != null) {
                val box = bestBoxRect

                val globalLineBox = android.graphics.Rect(
                    localCrop.offsetX + box.left.toInt(),
                    localCrop.offsetY + box.top.toInt(),
                    localCrop.offsetX + box.right.toInt(),
                    localCrop.offsetY + box.bottom.toInt()
                )

                updateDebugStep("[8] 최적의 번호판 박스 도출: $globalLineBox\nOpenCV 정밀화로 이동합니다.")
                localCrop.croppedBitmap.recycle()

                // OpenCV 마스킹 단계로 이동 (디버그 UI 유지하려면 약간의 딜레이 후 넘겨도 됨)
                runOnUiThread { debugHudContainer?.visibility = View.GONE }
                buildFinalWireframe(safeBitmap, globalLineBox, currentSession, debugInterceptor)
                return

            } else {
                val debugBmp = localCrop.croppedBitmap.copy(Bitmap.Config.ARGB_8888, true)
                updateDebugStep("[8] ❌ 실패: 터치 영역 내에서 번호판 박스를 찾지 못했습니다.")
                
                debugInterceptor.pauseAndShowStep(
                    "디버그 1단계: [FAIL] AI 모델 탐색 실패",
                    debugBmp,
                    "[FAIL] 터치 영역 내 번호판 없음",
                    listOf("검출 개수: $detectionCount", "번호판 중앙을 다시 터치해주세요.")
                )

                localCrop.croppedBitmap.recycle()
                safeBitmap.recycle()
                fallbackToManualMode(currentSession, "해당 위치 주변에서 번호판을 찾지 못했습니다.")
                return
            }

        } catch (e: Throwable) {
            Log.e("AI_DEBUG", "TFLite Pipeline 오류", e)

            if (resizedBitmap != null && !resizedBitmap.isRecycled && resizedBitmap !== localCrop.croppedBitmap) {
                resizedBitmap.recycle()
            }
            if (!localCrop.croppedBitmap.isRecycled) {
                localCrop.croppedBitmap.recycle()
            }
            safeBitmap.recycle()

            runOnUiThread {
                android.app.AlertDialog.Builder(this)
                    .setTitle("AI 파이프라인 에러")
                    .setMessage(e.message)
                    .setPositiveButton("확인", null)
                    .show()
                debugHudContainer?.visibility = View.GONE
            }

            fallbackToManualMode(currentSession, "AI 처리 중 오류가 발생했습니다.")
            return
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

        synchronized(tfliteLock) {
            try {
                tflite?.close()
            } catch (e: Exception) {
                Log.e("AI_DEBUG", "TFLite 종료 중 오류", e)
            }
            tflite = null
        }

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
