package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.YuvImage
import android.media.ExifInterface
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    data class Line(
        val x1: Double,
        val y1: Double,
        val x2: Double,
        val y2: Double,
        val length: Double,
        val angle: Double
    )

    private var webView: WebView? = null
    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    @Volatile
    private var isProcessing = false

    private var lastCapturedBitmap: Bitmap? = null
    private var previewBitmapRef: Bitmap? = null
    private val bitmapLock = Any()

    private lateinit var previewDir: File
    private lateinit var assetLoader: WebViewAssetLoader

    private val recognizer by lazy { 
        TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) 
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("JiSeKa Engine", "OpenCV 초기화 실패")
        }

        viewFinder = findViewById(R.id.viewFinder)
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        webView = findViewById(R.id.webView)

        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()

        previewDir = File(filesDir, "preview")
        if (!previewDir.exists()) {
            previewDir.mkdirs()
        }

        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler(
                "/preview/",
                WebViewAssetLoader.InternalStoragePathHandler(this, previewDir)
            )
            .build()

        viewFinder?.apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        }

        setupWebView()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun safeEvaluate(js: String) {
        runOnUiThread {
            if (!isDestroyed && !isFinishing && webView != null) {
                try {
                    webView?.evaluateJavascript(js, null)
                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "JS evaluate 실패", e)
                }
            }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView?.apply {
            setLayerType(View.LAYER_TYPE_SOFTWARE, null)
            setBackgroundColor(Color.TRANSPARENT)
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = true
                allowContentAccess = true
                allowFileAccessFromFileURLs = true
                allowUniversalAccessFromFileURLs = true
                mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            }
            
            webViewClient = object : WebViewClientCompat() {
                override fun shouldInterceptRequest(
                    view: WebView,
                    request: WebResourceRequest
                ): WebResourceResponse? {
                    return assetLoader.shouldInterceptRequest(request.url)
                }
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest) = false
            }
            
            webChromeClient = WebChromeClient()
            addJavascriptInterface(AndroidBridge(), "AndroidBridge")
            
            loadUrl("https://ziseka-app.vercel.app/?v=" + System.currentTimeMillis())
        }
    }

    inner class AndroidBridge {
        @JavascriptInterface
        fun takePhoto() { this@MainActivity.takePhoto() }

        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread { 
                if (isDestroyed || isFinishing) return@runOnUiThread
                if (isVisible) {
                    isProcessing = false
                    viewFinder?.visibility = View.VISIBLE
                    nativeBackgroundView?.visibility = View.GONE
                } else {
                    viewFinder?.visibility = View.GONE
                    nativeBackgroundView?.visibility = View.GONE
                }
            }
        }

        @JavascriptInterface
        fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            Log.d("JiSeKa Engine", "JS -> Android 브릿지 정상 호출됨: Text-First 분석 파이프라인 시작")
            
            analysisExecutor.execute {
                if (isDestroyed || isFinishing) return@execute
                
                var processingBitmap: Bitmap? = null
                var guideBitmap: Bitmap? = null
                
                try {
                    val rawBitmap = synchronized(bitmapLock) { 
                        lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
                    } ?: return@execute

                    processingBitmap = scaleBitmapDownToLimit(rawBitmap, 1920, 1080)
                    if (processingBitmap !== rawBitmap) rawBitmap.recycle()

                    val bmpW = processingBitmap.width.toFloat()
                    val bmpH = processingBitmap.height.toFloat()

                    val inputCorners = JSONObject(cornersJsonStr).getJSONArray("corners")
                    val mappedPoints = mutableListOf<PointF>()

                    for (i in 0 until 4) {
                        val p = inputCorners.getJSONObject(i)
                        val normX = p.getDouble("x").toFloat()
                        val normY = p.getDouble("y").toFloat()
                        val targetBmpX = (normX * bmpW).coerceIn(0f, bmpW - 1f)
                        val targetBmpY = (normY * bmpH).coerceIn(0f, bmpH - 1f)
                        mappedPoints.add(PointF(targetBmpX, targetBmpY))
                    }

                    // 1단계: 가이드 박스 기반 관심 영역(ROI) 1차 크롭 (10% 패딩)
                    val xs = mappedPoints.map { it.x }
                    val ys = mappedPoints.map { it.y }
                    val baseMinX = max(0, xs.minOrNull()?.toInt() ?: 0)
                    val baseMinY = max(0, ys.minOrNull()?.toInt() ?: 0)
                    val baseMaxX = min(processingBitmap.width - 1, xs.maxOrNull()?.toInt() ?: (processingBitmap.width - 1))
                    val baseMaxY = min(processingBitmap.height - 1, ys.maxOrNull()?.toInt() ?: (processingBitmap.height - 1))

                    val baseWidth = baseMaxX - baseMinX
                    val baseHeight = baseMaxY - baseMinY

                    val padX = (baseWidth * 0.10).toInt()
                    val padY = (baseHeight * 0.10).toInt()

                    val safeMinX = max(0, baseMinX - padX)
                    val safeMinY = max(0, baseMinY - padY)
                    val safeMaxX = min(processingBitmap.width - 1, baseMaxX + padX)
                    val safeMaxY = min(processingBitmap.height - 1, baseMaxY + padY)

                    val guideRect = android.graphics.Rect(safeMinX, safeMinY, safeMaxX, safeMaxY)
                    guideBitmap = Bitmap.createBitmap(processingBitmap, guideRect.left, guideRect.top, guideRect.width(), guideRect.height())

                    // 2단계: OCR로 텍스트 마스터 사각형 확보
                    val inputImage = InputImage.fromBitmap(guideBitmap, 0)
                    
                    recognizer.process(inputImage).addOnCompleteListener { task ->
                        // 💡 비동기 콜백이 UI 스레드를 막지 않도록 다시 Background Executor로 진입
                        analysisExecutor.execute {
                            if (isDestroyed || isFinishing) {
                                guideBitmap?.recycle()
                                processingBitmap?.recycle()
                                return@execute
                            }

                            var finalPoints = mappedPoints // 최악의 경우를 대비한 가이드 박스 백업

                            try {
                                if (task.isSuccessful && task.result.textBlocks.isNotEmpty()) {
                                    val textBlocks = task.result.textBlocks
                                    var tMinX = Int.MAX_VALUE
                                    var tMinY = Int.MAX_VALUE
                                    var tMaxX = 0
                                    var tMaxY = 0
                                    var hasValidText = false

                                    for (block in textBlocks) {
                                        if (isValidLicensePlatePattern(block.text)) {
                                            hasValidText = true
                                            block.boundingBox?.let { box ->
                                                tMinX = min(tMinX, box.left)
                                                tMinY = min(tMinY, box.top)
                                                tMaxX = max(tMaxX, box.right)
                                                tMaxY = max(tMaxY, box.bottom)
                                            }
                                        }
                                    }

                                    // 3단계: 텍스트 사각형 상하좌우 30% 확장
                                    if (hasValidText && tMinX < tMaxX && tMinY < tMaxY) {
                                        val tW = tMaxX - tMinX
                                        val tH = tMaxY - tMinY

                                        // 💡 상하좌우 30% 강제 확장 (노이즈 침범을 막고 번호판 물리적 테두리만 포함하는 최적의 수치)
                                        val ePadX = (tW * 0.30f).toInt()
                                        val ePadY = (tH * 0.30f).toInt()

                                        val eMinX = max(0, tMinX - ePadX)
                                        val eMinY = max(0, tMinY - ePadY)
                                        val eMaxX = min(guideBitmap!!.width - 1, tMaxX + ePadX)
                                        val eMaxY = min(guideBitmap!!.height - 1, tMaxY + ePadY)

                                        val expandedRect = org.opencv.core.Rect(eMinX, eMinY, eMaxX - eMinX, eMaxY - eMinY)

                                        // 4~5단계: 확장된 사각형 안쪽만 OpenCV 전처리 및 테두리 스케치
                                        val refinedPointsLocal = performTextAnchoredEdgeDetection(guideBitmap!!, expandedRect)
                                        
                                        if (refinedPointsLocal != null) {
                                            // OpenCV 좌표계를 원본 해상도(processingBitmap) 좌표계로 다시 매핑
                                            finalPoints = refinedPointsLocal.map {
                                                PointF(it.x + guideRect.left, it.y + guideRect.top)
                                            }
                                            Log.d("JiSeKa Engine", "텍스트 앵커링 테두리 검출 및 맵핑 성공!")
                                        } else {
                                            Log.w("JiSeKa Engine", "테두리 검출 실패. 유저 가이드박스(Fallback) 사용")
                                        }
                                    } else {
                                        Log.w("JiSeKa Engine", "유효한 텍스트 사각형 없음. 유저 가이드박스(Fallback) 사용")
                                    }
                                } else {
                                    Log.w("JiSeKa Engine", "OCR 실패. 유저 가이드박스(Fallback) 사용")
                                }

                                // 6단계: 검증된 점(finalPoints)으로 가림막 합성 및 JS 전송
                                val previewBitmap = processPerspectiveOverlay(processingBitmap!!, finalPoints)
                                val fileUriStr = saveBitmapToCacheFile(previewBitmap)
                                sendCachedPreviewToJs(finalPoints, fileUriStr, bmpW, bmpH)
                                previewBitmap.recycle()

                            } catch (e: Exception) {
                                Log.e("JiSeKa Engine", "비동기 파이프라인 크래시", e)
                                val fallbackUri = saveBitmapToCacheFile(processingBitmap!!)
                                sendCachedPreviewToJs(mappedPoints, fallbackUri, bmpW, bmpH)
                            } finally {
                                guideBitmap?.recycle()
                                processingBitmap?.recycle()
                            }
                        }
                    }

                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "분석 파이프라인 초기화 실패", e)
                    val safeFallbackJson = JSONObject.quote("{\"corners\":null, \"preview\":null}")
                    safeEvaluate("window.onNativeSuccess($safeFallbackJson)")
                    isProcessing = false 
                }
            }
        }

        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            analysisExecutor.execute {
                if (isDestroyed || isFinishing) return@execute
                
                var baseBitmap: Bitmap? = null
                var overlayResult: Bitmap? = null
                try {
                    baseBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } ?: return@execute
                    val jsonObj = JSONObject(cornersJsonStr)

                    if (jsonObj.isNull("corners")) {
                        saveBitmapToGallery(baseBitmap)
                        safeEvaluate("window.onNativeSaveComplete()")
                        return@execute
                    }

                    val corners = jsonObj.getJSONArray("corners")
                    val pts = mutableListOf<PointF>()
                    for (i in 0 until 4) {
                        val p = corners.getJSONObject(i)
                        val absX = p.getDouble("x").toFloat() * baseBitmap.width
                        val absY = p.getDouble("y").toFloat() * baseBitmap.height
                        pts.add(PointF(absX, absY))
                    }

                    overlayResult = processPerspectiveOverlay(baseBitmap, pts)
                    saveBitmapToGallery(overlayResult!!)
                    safeEvaluate("window.onNativeSaveComplete()")
                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "오버레이 저장 실패", e)
                } finally {
                    baseBitmap?.recycle()
                    overlayResult?.let { if (it !== baseBitmap) it.recycle() }
                }
            }
        }
        
        @JavascriptInterface
        fun showToast(msg: String) {
            runOnUiThread { 
                if (!isDestroyed && !isFinishing) Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() 
            }
        }
    }

    private fun scaleBitmapDownToLimit(bitmap: Bitmap, maxW: Int, maxH: Int): Bitmap {
        val oW = bitmap.width
        val oH = bitmap.height
        if (oW <= maxW && oH <= maxH) return bitmap
        val scale = min(maxW.toFloat() / oW, maxH.toFloat() / oH)
        return Bitmap.createScaledBitmap(bitmap, (oW * scale).toInt(), (oH * scale).toInt(), true)
    }

    private fun isValidLicensePlatePattern(text: String): Boolean {
        return text.replace(Regex("[^a-zA-Z0-9가-힣]"), "").length >= 3
    }

    // 💡 Text-First 구조에 맞게 완전히 새롭게 작성된 OpenCV 정밀 탐지 로직
    private fun performTextAnchoredEdgeDetection(bitmap: Bitmap, expandedRect: org.opencv.core.Rect): List<PointF>? {
        val mat = org.opencv.core.Mat()
        val gray = org.opencv.core.Mat()
        val roiMat = org.opencv.core.Mat()
        val edgeMat = org.opencv.core.Mat()
        val lines = org.opencv.core.Mat()
        val contours = mutableListOf<MatOfPoint>()

        try {
            Utils.bitmapToMat(bitmap, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

            val claheObj = Imgproc.createCLAHE(4.0, org.opencv.core.Size(8.0, 8.0))
            try { claheObj.apply(gray, gray) } finally { claheObj.collectGarbage() }

            // 💡 핵심: 그릴이나 범퍼를 피하기 위해, 텍스트가 있는 공간(expandedRect)만 칼같이 오려냅니다.
            gray.submat(expandedRect).copyTo(roiMat)

            Imgproc.GaussianBlur(roiMat, roiMat, org.opencv.core.Size(3.0, 3.0), 0.0)

            val median = computeMedian(roiMat)
            val lower = max(0.0, 0.66 * median)
            val upper = min(255.0, 1.33 * median)
            Imgproc.Canny(roiMat, edgeMat, lower, upper)

            val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(5.0, 5.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            morphKernel.release()

            if (isDestroyed || isFinishing) return null

            Imgproc.HoughLinesP(edgeMat, lines, 1.0, Math.PI / 180, 30, expandedRect.width * 0.10, 10.0)

            val rawTopLines = mutableListOf<Line>()
            val rawBottomLines = mutableListOf<Line>()
            val rawLeftLines = mutableListOf<Line>()
            val rawRightLines = mutableListOf<Line>()
            val roiCenterY = expandedRect.height / 2.0
            val roiCenterX = expandedRect.width / 2.0

            for (i in 0 until lines.rows()) {
                val vec = lines.get(i, 0) ?: continue
                val dx = vec[2] - vec[0]
                val dy = vec[3] - vec[1]

                val length = sqrt(dx * dx + dy * dy)
                var angle = Math.atan2(dy, dx) * 180.0 / Math.PI
                if (angle < 0) angle += 180.0

                val midX = (vec[0] + vec[2]) / 2.0
                val midY = (vec[1] + vec[3]) / 2.0
                val lineObj = Line(vec[0], vec[1], vec[2], vec[3], length, angle)

                if (abs(dx) > abs(dy)) {
                    if (midY < roiCenterY) rawTopLines.add(lineObj) else rawBottomLines.add(lineObj)
                } else {
                    if (midX < roiCenterX) rawLeftLines.add(lineObj) else rawRightLines.add(lineObj)
                }
            }

            val tEdge = mergeLinesLinearRegression(filterByDominantAngle(rawTopLines), true, expandedRect.width, expandedRect.height)
            val bEdge = mergeLinesLinearRegression(filterByDominantAngle(rawBottomLines), true, expandedRect.width, expandedRect.height)
            val lEdge = mergeLinesLinearRegression(filterByDominantAngle(rawLeftLines), false, expandedRect.width, expandedRect.height)
            val rEdge = mergeLinesLinearRegression(filterByDominantAngle(rawRightLines), false, expandedRect.width, expandedRect.height)

            if (tEdge != null && bEdge != null && lEdge != null && rEdge != null) {
                val tl = getIntersection(tEdge, lEdge)
                val tr = getIntersection(tEdge, rEdge)
                val br = getIntersection(bEdge, rEdge)
                val bl = getIntersection(bEdge, lEdge)

                if (tl != null && tr != null && br != null && bl != null) {
                    val candidateQuad = listOf(tl, tr, br, bl)
                    if (validateQuadrilateral(candidateQuad, expandedRect.width, expandedRect.height)) {
                        // 💡 검출된 점을 원래 해상도에 맞게 오프셋(+ left, top) 처리
                        return applySubPixelRefinement(gray, expandedRect, candidateQuad)
                    }
                }
            }

            val hierarchy = org.opencv.core.Mat()
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestContourPoints: List<PointF>? = null
            var maxArea = 0.0

            if (isDestroyed || isFinishing) return null

            for (contour in contours) {
                val m2f = MatOfPoint2f()
                val approx = MatOfPoint2f()
                try {
                    m2f.fromArray(*contour.toArray())
                    val peri = Imgproc.arcLength(m2f, true)
                    Imgproc.approxPolyDP(m2f, approx, 0.02 * peri, true)

                    val approxArray = approx.toArray()
                    val rawPoints = approxArray.map { PointF(it.x.toFloat(), it.y.toFloat()) }
                    val quadPoints = extractFourCorners(rawPoints)

                    if (quadPoints != null && validateQuadrilateral(quadPoints, expandedRect.width, expandedRect.height)) {
                        val contourMat = MatOfPoint(*approxArray)
                        val area = abs(Imgproc.contourArea(contourMat))
                        if (area > maxArea) {
                            maxArea = area
                            bestContourPoints = sortCornersStandard(quadPoints)
                        }
                        contourMat.release()
                    }
                } finally {
                    approx.release()
                    m2f.release()
                }
            }
            hierarchy.release()

            if (bestContourPoints != null) {
                return applySubPixelRefinement(gray, expandedRect, bestContourPoints)
            }

            return null
        } finally {
            mat.release()
            gray.release()
            roiMat.release()
            edgeMat.release()
            lines.release()
            contours.forEach { it.release() }
            contours.clear()
        }
    }

    private fun validateQuadrilateral(pts: List<PointF>, imageW: Int, imageH: Int): Boolean {
        if (pts.size != 4) return false
        val ordered = sortCornersStandard(pts)
        val contour = MatOfPoint(*ordered.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        
        return try {
            if (!Imgproc.isContourConvex(contour)) return false
            
            // 💡 텍스트 앵커링 구조에서는 이미 그릴이나 범퍼가 잘려나간 상태이므로, 
            // 발견된 다각형이 ROI 면적의 10% 이상을 차지하기만 하면 진짜 번호판으로 인정합니다.
            val area = abs(Imgproc.contourArea(contour))
            val totalArea = imageW * imageH.toDouble()
            if (area < totalArea * 0.1) return false
            
            true
        } finally {
            contour.release()
        }
    }

    private fun cleanupOldPreviewFiles() {
        previewDir.listFiles()?.forEach {
            if (System.currentTimeMillis() - it.lastModified() > 60_000) {
                it.delete()
            }
        }
    }

    private fun saveBitmapToCacheFile(bitmap: Bitmap): String? {
        if (bitmap.isRecycled) return null
        return try {
            cleanupOldPreviewFiles()
            val fileName = "preview_${UUID.randomUUID()}.jpg"
            val file = File(previewDir, fileName)
            FileOutputStream(file).use { out -> 
                val success = bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
                out.flush() 
                if (!success) {
                    file.delete()
                    return null
                }
            }
            "https://appassets.androidplatform.net/preview/$fileName"
        } catch (e: Exception) { null }
    }

    private fun sendCachedPreviewToJs(points: List<PointF>, fileUriStr: String?, bmpW: Float, bmpH: Float) {
        val outputArray = JSONArray()
        for (pt in points) { outputArray.put(JSONObject().put("x", (pt.x / bmpW).toDouble()).put("y", (pt.y / bmpH).toDouble())) }
        
        val resultJson = JSONObject().apply { 
            put("corners", outputArray)
            put("preview", fileUriStr ?: JSONObject.NULL) 
        }.toString()
        
        val safeEscapedJson = JSONObject.quote(resultJson)
        safeEvaluate("window.onNativeSuccess($safeEscapedJson)")
        isProcessing = false 
    }

    private fun computeMedian(mat: org.opencv.core.Mat): Double {
        val hist = org.opencv.core.Mat()
        val ranges = org.opencv.core.MatOfFloat(0f, 256f)
        val histSize = org.opencv.core.MatOfInt(256)
        return try {
            Imgproc.calcHist(listOf(mat), org.opencv.core.MatOfInt(0), org.opencv.core.Mat(), hist, histSize, ranges)
            val total = mat.total()
            var sum = 0.0
            for (i in 0 until 256) {
                sum += hist.get(i, 0)[0]
                if (sum >= total / 2.0) return i.toDouble()
            }
            127.0
        } finally {
            hist.release()
            ranges.release()
            histSize.release()
        }
    }

    private fun extractFourCorners(pts: List<PointF>): List<PointF>? {
        if (pts.size < 4 || pts.size > 10) return null
        if (pts.size == 4) return pts
        val tl = pts.minByOrNull { it.x + it.y } ?: return null
        val br = pts.maxByOrNull { it.x + it.y } ?: return null
        val tr = pts.maxByOrNull { it.x - it.y } ?: return null
        val bl = pts.minByOrNull { it.x - it.y } ?: return null
        return listOf(tl, tr, br, bl)
    }

    private fun circularAngleDiff(a: Double, b: Double): Double {
        val diff = abs(a - b)
        return min(diff, 180.0 - diff)
    }

    private fun filterByDominantAngle(lines: List<Line>): List<Line> {
        if (lines.size <= 1) return lines
        val angleBuckets = HashMap<Int, Double>()
        for (line in lines) {
            val normalizedAngle = ((line.angle + 90.0) % 180.0) - 90.0
            val bucket = ((normalizedAngle + 2.5) / 5.0).toInt() * 5
            angleBuckets[bucket] = (angleBuckets[bucket] ?: 0.0) + line.length
        }
        val dominantBucket = angleBuckets.maxByOrNull { it.value }?.key?.toDouble() ?: return lines
        val dominantAngle = if (dominantBucket < 0) dominantBucket + 180.0 else dominantBucket
        return lines.filter { circularAngleDiff(it.angle, dominantAngle) <= 15.0 }
    }

    private fun applySubPixelRefinement(gray: org.opencv.core.Mat, roi: org.opencv.core.Rect, points: List<PointF>): List<PointF> {
        if (gray.empty()) return sortCornersStandard(points.map { PointF(it.x + roi.x, it.y + roi.y) })

        val globalPoints = points.map { PointF(it.x + roi.x, it.y + roi.y) }
        val sorted = sortCornersStandard(globalPoints)
        
        val winSizeInt = max(3, roi.width / 50)
        val winSize = winSizeInt.toDouble()
        
        val isSafeToRefine = sorted.all { pt ->
            pt.x >= winSizeInt && pt.y >= winSizeInt && pt.x < (gray.cols() - winSizeInt) && pt.y < (gray.rows() - winSizeInt)
        }

        if (!isSafeToRefine) return sorted

        val subPixMat = MatOfPoint2f()
        subPixMat.fromArray(*sorted.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        
        return try {
            val searchWindow = org.opencv.core.Size(winSize, winSize)
            val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.01)
            Imgproc.cornerSubPix(gray, subPixMat, searchWindow, org.opencv.core.Size(-1.0, -1.0), criteria)
            subPixMat.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        } catch (e: Exception) { 
            sorted 
        } finally { 
            subPixMat.release() 
        }
    }

    private fun mergeLinesLinearRegression(lines: List<Line>, isHorizontal: Boolean, roiWidth: Int, roiHeight: Int): Line? {
        if (lines.isEmpty()) return null
        if (lines.size == 1) return lines[0]
        var sumW = 0.0; var sumX = 0.0; var sumY = 0.0; var sumXY = 0.0; var sumX2 = 0.0; var sumY2 = 0.0
        for (line in lines) {
            val w = line.length; sumW += w * 2.0; sumX += w * line.x1; sumY += w * line.y1
            sumXY += w * line.x1 * line.y1; sumX2 += w * line.x1 * line.x1; sumY2 += w * line.y1 * line.y1
            sumX += w * line.x2; sumY += w * line.y2; sumXY += w * line.x2 * line.y2
            sumX2 += w * line.x2 * line.x2; sumY2 += w * line.y2 * line.y2
        }
        if (sumW <= 0.0) return null
        val meanX = sumX / sumW
        val meanY = sumY / sumW

        return if (isHorizontal) {
            val denom = sumX2 - sumW * meanX * meanX
            if (abs(denom) < 1e-5) Line(meanX, 0.0, meanX, roiHeight.toDouble(), sumW, 90.0)
            else {
                val slope = (sumXY - sumW * meanX * meanY) / denom
                val intercept = meanY - slope * meanX
                Line(0.0, intercept, roiWidth.toDouble(), slope * roiWidth + intercept, sumW, Math.atan(slope) * 180.0 / Math.PI)
            }
        } else {
            val denom = sumY2 - sumW * meanY * meanY
            if (abs(denom) < 1e-5) Line(0.0, meanY, roiWidth.toDouble(), meanY, sumW, 0.0)
            else {
                val slope = (sumXY - sumW * meanX * meanY) / denom
                val intercept = meanX - slope * meanY
                Line(intercept, 0.0, slope * roiHeight + intercept, roiHeight.toDouble(), sumW, 90.0 - Math.atan(slope) * 180.0 / Math.PI)
            }
        }
    }

    private fun getIntersection(line1: Line, line2: Line): PointF? {
        val a1 = line1.y2 - line1.y1; val b1 = line1.x1 - line1.x2; val c1 = a1 * line1.x1 + b1 * line1.y1
        val a2 = line2.y2 - line2.y1; val b2 = line2.x1 - line2.x2; val c2 = a2 * line2.x1 + b2 * line2.y1
        val det = a1 * b2 - a2 * b1
        if (abs(det) < 1e-6) return null
        return PointF(((b2 * c1 - b1 * c2) / det).toFloat(), ((a1 * c2 - a2 * c1) / det).toFloat())
    }

    private fun sortCornersStandard(pts: List<PointF>): List<PointF> {
        if (pts.size != 4) return pts
        val cx = pts.map { it.x }.average().toFloat()
        val cy = pts.map { it.y }.average().toFloat()
        val radialSorted = pts.sortedBy { atan2(it.y - cy, it.x - cx) }
        var startIndex = 0; var minSum = Float.MAX_VALUE
        for (i in 0 until 4) {
            val sum = radialSorted[i].x + radialSorted[i].y
            if (sum < minSum) { minSum = sum; startIndex = i }
        }
        return listOf(
            radialSorted[startIndex], radialSorted[(startIndex + 1) % 4],
            radialSorted[(startIndex + 2) % 4], radialSorted[(startIndex + 3) % 4]
        )
    }

    private fun processPerspectiveOverlay(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val targetMat = org.opencv.core.Mat(); val maskMat = org.opencv.core.Mat()
        val warpedMask = org.opencv.core.Mat(); val alphaMask = org.opencv.core.Mat(); val finalBlended = org.opencv.core.Mat()

        try {
            Utils.bitmapToMat(result, targetMat)
            val resId = resources.getIdentifier("plate_mask", "drawable", packageName)
            val maskBmp = if (resId != 0) BitmapFactory.decodeResource(resources, resId) 
                          else Bitmap.createBitmap(600, 150, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.LTGRAY) }
            Utils.bitmapToMat(maskBmp, maskMat)
            maskBmp.recycle()

            val srcPts = MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), 0.0),
                org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()), org.opencv.core.Point(0.0, maskMat.rows().toDouble())
            )
            val sortedTargets = sortCornersStandard(targetCorners)
            val dstPts = MatOfPoint2f(*sortedTargets.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(maskMat, warpedMask, transform, targetMat.size(), Imgproc.INTER_LINEAR)

            Imgproc.cvtColor(warpedMask, alphaMask, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.threshold(alphaMask, alphaMask, 1.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.GaussianBlur(alphaMask, alphaMask, org.opencv.core.Size(7.0, 7.0), 0.0)

            val warpedMatFloat = org.opencv.core.Mat(); val targetMatFloat = org.opencv.core.Mat(); val alphaMaskFloat = org.opencv.core.Mat()

            warpedMask.convertTo(warpedMatFloat, CvType.CV_32FC4)
            targetMat.convertTo(targetMatFloat, CvType.CV_32FC4)
            alphaMask.convertTo(alphaMaskFloat, CvType.CV_32FC1, 1.0 / 255.0)

            val warpedChannels = mutableListOf<org.opencv.core.Mat>()
            val targetChannels = mutableListOf<org.opencv.core.Mat>()

            Core.split(warpedMatFloat, warpedChannels)
            Core.split(targetMatFloat, targetChannels)

            val invAlpha = org.opencv.core.Mat()
            Core.subtract(org.opencv.core.Mat.ones(alphaMaskFloat.size(), CvType.CV_32FC1), alphaMaskFloat, invAlpha)

            val targetChan = org.opencv.core.Mat()
            for (i in 0 until 3) { 
                Core.multiply(warpedChannels[i], alphaMaskFloat, warpedChannels[i])
                Core.multiply(targetChannels[i], invAlpha, targetChan)
                Core.add(warpedChannels[i], targetChan, warpedChannels[i])
            }

            if (warpedChannels.size > 3 && targetChannels.size > 3) {
                val alphaClone = targetChannels[3].clone()
                warpedChannels[3].release(); warpedChannels[3] = alphaClone
            }
            
            Core.merge(warpedChannels, finalBlended)
            finalBlended.convertTo(finalBlended, CvType.CV_8UC4)
            Utils.matToBitmap(finalBlended, result)

            invAlpha.release(); targetChan.release(); srcPts.release(); dstPts.release(); transform.release()
            warpedMatFloat.release(); targetMatFloat.release(); alphaMaskFloat.release()
            warpedChannels.forEach { it.release() }; targetChannels.forEach { it.release() }

            return result
        } finally {
            targetMat.release(); maskMat.release(); warpedMask.release(); alphaMask.release(); finalBlended.release()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val targetRatio = AspectRatio.RATIO_16_9 
            val preview = Preview.Builder().setTargetAspectRatio(targetRatio).build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).setTargetAspectRatio(targetRatio).build()
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) { Log.e("JiSeKa", "Camera bind failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) viewFinder?.post { startCamera() }
    }

    private fun takePhoto() {
        if (isProcessing) return
        isProcessing = true
        val capture = imageCapture ?: run { isProcessing = false; return }

        cleanupOldPreviewFiles()
        val fileName = "capture_${UUID.randomUUID()}.jpg"
        val photoFile = File(previewDir, fileName)
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        capture.takePicture(outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                try {
                    val targetW = viewFinder?.width ?: 1080
                    val targetH = viewFinder?.height ?: 1920
                    val safeBitmap = loadSafeBitmap(photoFile.absolutePath, targetW, targetH)
                    
                    synchronized(bitmapLock) { 
                        lastCapturedBitmap?.let { if (!it.isRecycled) { it.recycle() } }
                        lastCapturedBitmap = safeBitmap 
                    }
                    
                    val previewUri = "https://appassets.androidplatform.net/preview/$fileName"
                    
                    runOnUiThread { 
                        if (isDestroyed || isFinishing) return@runOnUiThread

                        nativeBackgroundView?.setImageDrawable(null)
                        val safeEscapedUri = JSONObject.quote(previewUri)
                        viewFinder?.visibility = View.GONE
                        
                        previewBitmapRef?.let { if (!it.isRecycled) it.recycle() }
                        val previewW = max(1, safeBitmap.width / 2); val previewH = max(1, safeBitmap.height / 2)
                        
                        previewBitmapRef = Bitmap.createScaledBitmap(safeBitmap, previewW, previewH, true)
                        nativeBackgroundView?.setImageBitmap(previewBitmapRef)
                        nativeBackgroundView?.visibility = View.VISIBLE
                       
                        val bmpW = safeBitmap.width; val bmpH = safeBitmap.height
                        safeEvaluate("window.onNativePhotoCaptured($safeEscapedUri, $bmpW, $bmpH)")
                    }
                } catch (t: Throwable) { 
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing) {
                            isProcessing = false; viewFinder?.visibility = View.VISIBLE; nativeBackgroundView?.visibility = View.GONE
                            Toast.makeText(this@MainActivity, "사진 처리 중 문제가 발생했습니다.", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
            override fun onError(e: ImageCaptureException) { 
                runOnUiThread {
                    if (!isDestroyed && !isFinishing) {
                        isProcessing = false; Toast.makeText(this@MainActivity, "캡처 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }

    private fun loadSafeBitmap(path: String, maxW: Int, maxH: Int): Bitmap {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, options)
        options.inSampleSize = calculateInSampleSize(options, maxW, maxH)
        options.inJustDecodeBounds = false
        val bitmap = BitmapFactory.decodeFile(path, options) ?: throw RuntimeException("Bitmap load failed")
        val exif = ExifInterface(path)
        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        }
        if (matrix.isIdentity) return bitmap
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val (height: Int, width: Int) = options.outHeight to options.outWidth
        var inSampleSize = 1
        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2; val halfWidth: Int = width / 2
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) { inSampleSize *= 2 }
        }
        return inSampleSize
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "JiSeKa_${System.currentTimeMillis()}.jpg")
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/JiSeKa")
        }
        var success = false
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        uri?.let { 
            contentResolver.openOutputStream(it)?.use { stream -> success = bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream) } 
        }
        runOnUiThread { 
            if (!isDestroyed && !isFinishing) {
                if (success) Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() 
                else Toast.makeText(this, "저장공간 부족 등으로 저장에 실패했습니다.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() { 
        synchronized(bitmapLock) { 
            lastCapturedBitmap?.let { if (!it.isRecycled) it.recycle() }
            lastCapturedBitmap = null 
        }
        previewBitmapRef?.let { if (!it.isRecycled) it.recycle() }
        previewBitmapRef = null
        recognizer.close(); cameraExecutor.shutdownNow(); analysisExecutor.shutdownNow()
        previewDir.listFiles()?.forEach { it.delete() }
        webView?.apply { stopLoading(); clearHistory(); removeAllViews(); destroy() }
        webView = null
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
