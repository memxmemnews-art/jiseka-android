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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.hypot

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

    // 🛠️ [디버그 전용] 스레드 제어 및 임시 UI 버튼
    private var debugLatch: CountDownLatch? = null
    private var btnDebugNext: Button? = null

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

        setupDebugUI() // 🛠️ 디버그 버튼 초기화
        setupUIListeners()
        setupOrientationListener()
        resetToLiveMode()

        if (allPermissionsGranted()) viewFinder?.post { startCamera() }
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    // 🛠️ [디버그 전용] 동적으로 하단에 임시 버튼 추가
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
        val params = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            bottomMargin = 300 
        }
        addContentView(btnDebugNext, params)
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
            Toast.makeText(this, "경고: 텍스처를 찾을 수 없어 기본 색상으로 대체됩니다.", Toast.LENGTH_LONG).show()
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

        // 🌟 선 긋기 리스너 매핑 (명시적 라벨 lineDrop@ 적용으로 컴파일 에러 완전 차단)
        nativeGuideView?.onLineDropListener = lineDrop@{ startUiPoint, endUiPoint ->
            if (!isMatrixReady) return@lineDrop

            val currentSession = captureSessionId.get()
            progressBar?.visibility = View.VISIBLE
            nativeGuideView?.visibility = View.GONE
            guideText?.visibility = View.GONE
            
            precomputeExecutor.execute {
                if (captureSessionId.get() != currentSession) return@execute 
                
                val safeBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) }
       
                if (safeBitmap != null) {
                    val startCoords = FloatArray(2).apply { this[0] = startUiPoint.x; this[1] = startUiPoint.y }
                    inverseMatrix.mapPoints(startCoords)

                    val endCoords = FloatArray(2).apply { this[0] = endUiPoint.x; this[1] = endUiPoint.y }
                    inverseMatrix.mapPoints(endCoords)
         
                    // 🛠️ [디버그 전용] 락(Lock)을 걸어 스레드를 일시정지 시키는 리스너
                    val debugInterceptor = object : PlateDetectionEngine.DetectionDebugListener {
                        override fun pauseAndShowStep(stageName: String, bitmap: Bitmap) {
                            debugLatch = CountDownLatch(1)
                            
                            runOnUiThread {
                                if (isFinishing || isDestroyed) return@runOnUiThread
                                Toast.makeText(this@MainActivity, stageName, Toast.LENGTH_SHORT).show()
                                nativeBackgroundView?.setImageBitmap(bitmap)
                                btnDebugNext?.visibility = View.VISIBLE 
                                progressBar?.visibility = View.GONE 
                            }
                            
                            debugLatch?.await() // 버튼 누를 때까지 대기
                            
                            runOnUiThread { 
                                btnDebugNext?.visibility = View.GONE 
                                progressBar?.visibility = View.VISIBLE 
                            }
                        }
                    }

                    val targetPolygon = PlateDetectionEngine.rescuePlateFromLine(
                        safeBitmap, 
                        startCoords[0], startCoords[1], 
                        endCoords[0], endCoords[1],
                        debugListener = debugInterceptor
                    )
                    safeBitmap.recycle()

                    runOnUiThread {
                        if (isFinishing || isDestroyed || captureSessionId.get() != currentSession) return@runOnUiThread
                        
                        if (targetPolygon != null && targetPolygon.isNotEmpty()) {
                            triggerInstantMasking(targetPolygon)
                        } else {
                            progressBar?.visibility = View.GONE
                            nativeGuideView?.visibility = View.VISIBLE
                            Toast.makeText(this@MainActivity, "번호판 추적 최종 실패.", Toast.LENGTH_LONG).show()
                        }
                    }
                } else {
                    runOnUiThread { progressBar?.visibility = View.GONE }
                }
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
        
        debugLatch?.countDown() // 🛠️ 진행 중 취소 시 스레드 해방
        btnDebugNext?.visibility = View.GONE
        
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
                
                guideText?.text = "번호판 위로 선을 길게 그어주세요"
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

    private fun findVerticalIntersections(hull: List<Point>, targetX: Double): Pair<Double, Double>? {
        val intersections = mutableListOf<Double>()
        for (i in hull.indices) {
            val p1 = hull[i]
            val p2 = hull[(i + 1) % hull.size]

            val minX = min(p1.x, p2.x)
            val maxX = max(p1.x, p2.x)

            if (targetX < minX - 1e-5 || targetX > maxX + 1e-5) continue

            if (Math.abs(p2.x - p1.x) < 1e-6) {
                intersections.add(p1.y)
                intersections.add(p2.y)
            } else {
                val t = (targetX - p1.x) / (p2.x - p1.x)
                intersections.add(p1.y + t * (p2.y - p1.y))
            }
        }

        if (intersections.size < 2) return null

        intersections.sort()
        return Pair(intersections.first(), intersections.last())
    }

    private fun getPointAtArcLength(chain: List<Point>, targetRatio: Double): Point {
        if (chain.size == 1) return chain.first()
        if (targetRatio <= 0.0) return chain.first()
        if (targetRatio >= 1.0) return chain.last()

        val segmentLengths = mutableListOf<Double>()
        var totalLength = 0.0
        
        for (i in 0 until chain.size - 1) {
            val dist = hypot(chain[i+1].x - chain[i].x, chain[i+1].y - chain[i].y)
            segmentLengths.add(dist)
            totalLength += dist
        }

        val targetLength = totalLength * targetRatio
        var accumulatedLength = 0.0

        for (i in 0 until chain.size - 1) {
            val segLen = segmentLengths[i]
            if (accumulatedLength + segLen >= targetLength) {
                val remaining = targetLength - accumulatedLength
                val ratio = if (segLen > 0) remaining / segLen else 0.0
                val p1 = chain[i]
                val p2 = chain[i+1]
                return Point(p1.x + ratio * (p2.x - p1.x), p1.y + ratio * (p2.y - p1.y))
            }
            accumulatedLength += segLen
        }
        return chain.last()
    }

    private fun splitHullIntoChains(hullPts: List<Point>): Pair<List<Point>, List<Point>> {
        var leftIdx = 0
        var rightIdx = 0
        for (i in hullPts.indices) {
            if (hullPts[i].x < hullPts[leftIdx].x) leftIdx = i
            if (hullPts[i].x > hullPts[rightIdx].x) rightIdx = i
        }

        val n = hullPts.size
        val chain1 = mutableListOf<Point>()
        var curr = leftIdx
        while (curr != rightIdx) {
            chain1.add(hullPts[curr])
            curr = (curr + 1) % n
        }
        chain1.add(hullPts[rightIdx])

        val chain2 = mutableListOf<Point>()
        curr = rightIdx
        while (curr != leftIdx) {
            chain2.add(hullPts[curr])
            curr = (curr + 1) % n
        }
        chain2.add(hullPts[leftIdx])
        
        val chain1AvgY = chain1.map { it.y }.average()
        val chain2AvgY = chain2.map { it.y }.average()

        return if (chain1AvgY < chain2AvgY) {
            val topChain = if (chain1.first().x < chain1.last().x) chain1 else chain1.reversed()
            val bottomChain = if (chain2.first().x < chain2.last().x) chain2 else chain2.reversed()
            Pair(topChain, bottomChain)
        } else {
            val topChain = if (chain2.first().x < chain2.last().x) chain2 else chain2.reversed()
            val bottomChain = if (chain1.first().x < chain1.last().x) chain1 else chain1.reversed()
            Pair(topChain, bottomChain)
        }
    }

    private fun applyMaskToMat(mat: Mat, corners: List<ImmutablePoint>, textureInput: Mat) {
        if (corners.isEmpty()) return

        var maskMat: Mat? = null; var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null; var alphaMat: Mat? = null
        var warpedTexture: Mat? = null
        var preparedTexture: Mat? = null
        
        var stripWarped: Mat? = null
        var stripMask: Mat? = null
        val scalarZero = Scalar(0.0)
        val scalar255 = Scalar(255.0)
        
        val matChannels = ArrayList<Mat>()
        val textureChannels = ArrayList<Mat>()

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
                if (mat.channels() == 3 && textureInput.channels() == 4) Imgproc.cvtColor(textureInput, preparedTexture, Imgproc.COLOR_RGBA2RGB)
                else textureInput.copyTo(preparedTexture)
            } else {
                textureInput.copyTo(preparedTexture)
            }

            warpedTexture = Mat.zeros(mat.size(), mat.type()) 
            
            if (pts.size < 4) {
                fallbackSingleWarp(pts, preparedTexture, warpedTexture)
            } else {
                val (topChain, bottomChain) = splitHullIntoChains(pts)
                
                val segments = 8 
                val texWidth = preparedTexture.cols().toDouble()
                val texHeight = preparedTexture.rows().toDouble()

                stripWarped = Mat(mat.size(), mat.type())
                stripMask = Mat.zeros(mat.size(), CvType.CV_8UC1)

                for (i in 0 until segments) {
                    val progressLeft = i.toDouble() / segments
                    val progressRight = (i + 1).toDouble() / segments
                    
                    val ptTopLeft = getPointAtArcLength(topChain, progressLeft)
                    val ptTopRight = getPointAtArcLength(topChain, progressRight)
                    val ptBottomLeft = getPointAtArcLength(bottomChain, progressLeft)
                    val ptBottomRight = getPointAtArcLength(bottomChain, progressRight)

                    val srcPts = MatOfPoint2f(
                        Point(progressLeft * texWidth, 0.0),
                        Point(progressRight * texWidth, 0.0),
                        Point(progressRight * texWidth, texHeight),
                        Point(progressLeft * texWidth, texHeight)
                    )

                    val dstPts = MatOfPoint2f(
                        ptTopLeft,
                        ptTopRight,
                        ptBottomRight,
                        ptBottomLeft
                    )

                    val perspectiveMat = Imgproc.getPerspectiveTransform(srcPts, dstPts)
                    
                    stripWarped.setTo(scalarZero)
                    stripMask.setTo(scalarZero)
                    
                    Imgproc.warpPerspective(preparedTexture, stripWarped, perspectiveMat, mat.size(), Imgproc.INTER_LINEAR)
                    
                    val stripContour = org.opencv.core.MatOfPoint(*dstPts.toArray())
                    Imgproc.fillPoly(stripMask, listOf(stripContour), scalar255)
                    
                    stripWarped.copyTo(warpedTexture, stripMask)

                    srcPts.release(); dstPts.release(); perspectiveMat.release()
                    stripContour.release()
                }
            }

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

        } finally {
            maskMat?.release(); contour?.release(); blurredMask?.release()
            alphaMat?.release(); warpedTexture?.release()
            preparedTexture?.release()
            
            stripWarped?.release()
            stripMask?.release()
            
            matChannels.forEach { it.release() }; textureChannels.forEach { it.release() }
            
            if (originalWasRgba) Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2RGBA)
        }
    }

    private fun fallbackSingleWarp(pts: List<Point>, textureInput: Mat, outputTexture: Mat) {
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
            Point(0.0, 0.0), Point(textureInput.cols().toDouble(), 0.0),
            Point(textureInput.cols().toDouble(), textureInput.rows().toDouble()), Point(0.0, textureInput.rows().toDouble())
        )

        val perspectiveMat = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        Imgproc.warpPerspective(textureInput, outputTexture, perspectiveMat, outputTexture.size(), Imgproc.INTER_LINEAR)
        
        srcPts.release(); dstPts.release(); perspectiveMat.release()
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
        
        debugLatch?.countDown() // 🛠️ 스레드 해방

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
