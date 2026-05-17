package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.webkit.*
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
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
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
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
    private val bitmapLock = Any()

    private var cachedPreviewFile: File? = null
    
    // [해결책] WebViewAssetLoader 선언
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

        // 캐시 파일 설정 (AssetLoader와 경로를 맞추기 위해 내부 cacheDir 사용)
        cachedPreviewFile = File(cacheDir, "preview_cache.jpg")

        // [해결책] WebViewAssetLoader 초기화 (로컬 파일을 가상 HTTPS 도메인으로 서빙)
        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler(
                "/cache/",
                WebViewAssetLoader.InternalStoragePathHandler(this, cacheDir)
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

        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()
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
                // [해결책] HTTPS 페이지 안에서의 다양한 콘텐츠 로드 허용
                mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            }
            
            // [해결책] WebViewClientCompat을 사용하여 AssetLoader로 리소스 요청 가로채기
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
            
            loadUrl("https://ziseka-app.vercel.app/")
        }
    }

    inner class AndroidBridge {
        @JavascriptInterface
        fun takePhoto() { this@MainActivity.takePhoto() }

        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread { 
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
            analysisExecutor.execute {
                var processingBitmap: Bitmap? = null
                try {
                    val rawBitmap = synchronized(bitmapLock) { 
                        lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
                    } ?: return@execute

                    processingBitmap = scaleBitmapDownToLimit(rawBitmap, 1920, 1080)

                    val bmpW = processingBitmap.width.toFloat()
                    val bmpH = processingBitmap.height.toFloat()
                    val viewW = viewFinder?.width?.toFloat() ?: 1f
                    val viewH = viewFinder?.height?.toFloat() ?: 1f

                    val inputCorners = JSONObject(cornersJsonStr).getJSONArray("corners")
                    val mappedPoints = mutableListOf<PointF>()

                    val scale = min(viewW / bmpW, viewH / bmpH)
                    val displayedW = bmpW * scale
                    val displayedH = bmpH * scale
                    val offsetX = (viewW - displayedW) / 2f
                    val offsetY = (viewH - displayedH) / 2f

                    for (i in 0 until 4) {
                        val p = inputCorners.getJSONObject(i)
                        val normX = p.getDouble("x").toFloat()
                        val normY = p.getDouble("y").toFloat()

                        val screenPixelX = normX * viewW
                        val screenPixelY = normY * viewH

                        val targetBmpX = ((screenPixelX - offsetX) / scale).coerceIn(0f, bmpW - 1f)
                        val targetBmpY = ((screenPixelY - offsetY) / scale).coerceIn(0f, bmpH - 1f)
                        mappedPoints.add(PointF(targetBmpX, targetBmpY))
                    }

                    val refinedPoints = extractPlateCornersViaLineFitting(processingBitmap, mappedPoints)
                    verifyAndGenerateFilePreview(processingBitmap, refinedPoints, mappedPoints, bmpW, bmpH)

                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "분석 파이프라인 크래시 방어", e)
                    val safeFallbackJson = JSONObject.quote("{\"corners\":null, \"preview\":null}")
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSuccess($safeFallbackJson)", null) }
                }
            }
        }

        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            analysisExecutor.execute {
                var baseBitmap: Bitmap? = null
                var overlayResult: Bitmap? = null
                try {
                    baseBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } ?: return@execute
                    val jsonObj = JSONObject(cornersJsonStr)

                    if (jsonObj.isNull("corners")) {
                        saveBitmapToGallery(baseBitmap)
                        runOnUiThread { webView?.evaluateJavascript("window.onNativeSaveComplete()", null) }
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
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSaveComplete()", null) }
                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "오버레이 저장 실패", e)
                }
            }
        }
        
        @JavascriptInterface
        fun showToast(msg: String) {
            runOnUiThread { Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() }
        }
    }

    private fun scaleBitmapDownToLimit(bitmap: Bitmap, maxW: Int, maxH: Int): Bitmap {
        val oW = bitmap.width
        val oH = bitmap.height
        if (oW <= maxW && oH <= maxH) return bitmap
        val scale = min(maxW.toFloat() / oW, maxH.toFloat() / oH)
        return Bitmap.createScaledBitmap(bitmap, (oW * scale).toInt(), (oH * scale).toInt(), true)
    }

    private fun verifyAndGenerateFilePreview(
        sourceBitmap: Bitmap, targetPoints: List<PointF>, fallbackPoints: List<PointF>, bmpW: Float, bmpH: Float
    ) {
        val flatBmp = rectifyToFlatPlate(sourceBitmap, targetPoints)
    
        if (flatBmp != null) {
            val inputImage = InputImage.fromBitmap(flatBmp, 0)
            recognizer.process(inputImage).addOnCompleteListener { task ->
                val finalPoints = if (task.isSuccessful && isValidLicensePlatePattern(task.result.text)) {
                    targetPoints
                } else {
                    fallbackPoints
                }
                val previewBitmap = processPerspectiveOverlay(sourceBitmap, finalPoints)
                val fileUriStr = saveBitmapToCacheFile(previewBitmap)
                sendCachedPreviewToJs(finalPoints, fileUriStr, bmpW, bmpH)
            }
        } else {
            val fileUriStr = saveBitmapToCacheFile(sourceBitmap)
            sendCachedPreviewToJs(fallbackPoints, fileUriStr, bmpW, bmpH)
        }
    }

    private fun saveBitmapToCacheFile(bitmap: Bitmap): String? {
        return try {
            cachedPreviewFile?.let { file ->
                if (file.exists()) file.delete()
                
                FileOutputStream(file).use { out -> 
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
                    out.flush() 
                }
                // [해결책] WebViewAssetLoader의 가상 HTTPS 주소로 반환하여 CORS 차단 원천 봉쇄
                "https://appassets.androidplatform.net/cache/preview_cache.jpg?t=${System.currentTimeMillis()}"
            }
        } catch (e: Exception) { 
            Log.e("JiSeKa Engine", "캐시 저장 실패", e)
            null 
        }
    }

    private fun sendCachedPreviewToJs(points: List<PointF>, fileUriStr: String?, bmpW: Float, bmpH: Float) {
        val outputArray = JSONArray()
        for (pt in points) { outputArray.put(JSONObject().put("x", (pt.x / bmpW).toDouble()).put("y", (pt.y / bmpH).toDouble())) }
        
        val resultJson = JSONObject().apply { 
            put("corners", outputArray)
            put("preview", fileUriStr ?: JSONObject.NULL) 
        }.toString()
        
        val safeEscapedJson = JSONObject.quote(resultJson)
        runOnUiThread { webView?.evaluateJavascript("window.onNativeSuccess($safeEscapedJson)", null) }
    }

    private fun isValidLicensePlatePattern(text: String): Boolean = text.replace(Regex("\\s+"), "").count { it.isDigit() } >= 3

    private fun extractPlateCornersViaLineFitting(bitmap: Bitmap, pts: List<PointF>): List<PointF> {
        val mat = org.opencv.core.Mat()
        val gray = org.opencv.core.Mat()
        val roiMat = org.opencv.core.Mat()
        val edgeMat = org.opencv.core.Mat()
        val lines = org.opencv.core.Mat()

        try {
            Utils.bitmapToMat(bitmap, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            
            val claheObj = Imgproc.createCLAHE(2.5, org.opencv.core.Size(8.0, 8.0))
            claheObj.apply(gray, gray)
            
            val xs = pts.map { it.x }
            val ys = pts.map { it.y }
            val baseMinX = max(0, xs.minOrNull()?.toInt() ?: 0)
            val baseMinY = max(0, ys.minOrNull()?.toInt() ?: 0)
            val baseMaxX = min(mat.cols() - 1, xs.maxOrNull()?.toInt() ?: (mat.cols() - 1))
            val baseMaxY = min(mat.rows() - 1, ys.maxOrNull()?.toInt() ?: (mat.rows() - 1))
            
            val baseWidth = baseMaxX - baseMinX
            val baseHeight = baseMaxY - baseMinY
            if (baseWidth <= 20 || baseHeight <= 20) return pts

            val padX = (baseWidth * 0.15).toInt()
            val padY = (baseHeight * 0.20).toInt()

            val safeMinX = max(0, baseMinX - padX)
            val safeMinY = max(0, baseMinY - padY)
            val safeMaxX = min(mat.cols() - 1, baseMaxX + padX)
            val safeMaxY = min(mat.rows() - 1, baseMaxY + padY)

            val paddedRoi = org.opencv.core.Rect(safeMinX, safeMinY, safeMaxX - safeMinX, safeMaxY - safeMinY)
            
            gray.submat(paddedRoi).copyTo(roiMat)
            Imgproc.GaussianBlur(roiMat, roiMat, org.opencv.core.Size(5.0, 5.0), 0.0)
            
            val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(3.0, 3.0))
            Imgproc.morphologyEx(roiMat, roiMat, Imgproc.MORPH_CLOSE, morphKernel)
            morphKernel.release()
            
            Imgproc.Canny(roiMat, edgeMat, 50.0, 150.0)

            Imgproc.HoughLinesP(edgeMat, lines, 1.0, Math.PI / 180, 30, paddedRoi.width * 0.3, 10.0)
            
            val rawTopLines = mutableListOf<Line>()
            val rawBottomLines = mutableListOf<Line>()
            val rawLeftLines = mutableListOf<Line>()
            val rawRightLines = mutableListOf<Line>()
            val roiCenterY = paddedRoi.height / 2.0
            val roiCenterX = paddedRoi.width / 2.0

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

            val tEdge = mergeLinesLinearRegression(filterByDominantAngle(rawTopLines), true, paddedRoi.width, paddedRoi.height)
            val bEdge = mergeLinesLinearRegression(filterByDominantAngle(rawBottomLines), true, paddedRoi.width, paddedRoi.height)
            val lEdge = mergeLinesLinearRegression(filterByDominantAngle(rawLeftLines), false, paddedRoi.width, paddedRoi.height)
            val rEdge = mergeLinesLinearRegression(filterByDominantAngle(rawRightLines), false, paddedRoi.width, paddedRoi.height)

            if (tEdge != null && bEdge != null && lEdge != null && rEdge != null) {
                val tl = getIntersection(tEdge, lEdge)
                val tr = getIntersection(tEdge, rEdge)
                val br = getIntersection(bEdge, rEdge)
                val bl = getIntersection(bEdge, lEdge)
                if (tl != null && tr != null && br != null && bl != null) {
                    val candidateQuad = listOf(tl, tr, br, bl)
                    
                    if (validateQuadrilateral(candidateQuad, paddedRoi.width, paddedRoi.height)) {
                        return applySubPixelRefinement(gray, paddedRoi, candidateQuad)
                    } else {
                        Log.w("JiSeKa Engine", "Hough 교점 기하학 검증 실패. 2순위 윤곽선 탐색으로 전환합니다.")
                    }
                }
            }

            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = org.opencv.core.Mat()
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            var bestContourPoints: List<PointF>? = null
            var maxArea = 0.0

            for (contour in contours) {
                val m2f = MatOfPoint2f(*contour.toArray())
                val peri = Imgproc.arcLength(m2f, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(m2f, approx, 0.02 * peri, true)

                val rawPoints = approx.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
                if (validateQuadrilateral(rawPoints, paddedRoi.width, paddedRoi.height)) {
                    val contourMat = MatOfPoint(*approx.toArray())
                    val area = abs(Imgproc.contourArea(contourMat))
                    if (area > maxArea) {
                        maxArea = area
                        bestContourPoints = sortCornersStandard(rawPoints)
                    }
                    contourMat.release()
                }
                m2f.release()
                approx.release()
            }
            hierarchy.release()
            contours.forEach { it.release() }

            if (bestContourPoints != null) {
                return applySubPixelRefinement(gray, paddedRoi, bestContourPoints)
            }

            return pts

        } finally {
            mat.release()
            gray.release()
            roiMat.release()
            edgeMat.release()
            lines.release()
        }
    }

    private fun validateQuadrilateral(pts: List<PointF>, imageW: Int, imageH: Int): Boolean {
        if (pts.size != 4) return false
        val ordered = sortCornersStandard(pts)

        val marginX = imageW * 0.5f
        val marginY = imageH * 0.5f
        val isInsideBounds = ordered.all {
            it.x in -marginX..(imageW + marginX) && it.y in -marginY..(imageH + marginY)
        }
        if (!isInsideBounds) return false

        val contour = MatOfPoint(*ordered.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        try {
            if (!Imgproc.isContourConvex(contour)) return false

            val area = abs(Imgproc.contourArea(contour))
            val totalArea = imageW * imageH.toDouble()
            if (area < totalArea * 0.005 || area > totalArea * 0.7) return false

            val distance = { p1: PointF, p2: PointF -> sqrt((p1.x - p2.x).pow(2) + (p1.y - p2.y).pow(2)) }
            val topW = distance(ordered[0], ordered[1])
            val bottomW = distance(ordered[3], ordered[2])
            val leftH = distance(ordered[0], ordered[3])
            val rightH = distance(ordered[1], ordered[2])

            val avgW = (topW + bottomW) / 2f
            val avgH = (leftH + rightH) / 2f
            if (avgH <= 0f) return false

            val ratio = avgW / avgH
            if (ratio < 2.0f || ratio > 6.5f) return false

            val minW = min(topW, bottomW)
            val maxW = max(topW, bottomW)
            val minH = min(leftH, rightH)
            val maxH = max(leftH, rightH)
            if (minW < 10f || minH < 5f || maxW / minW > 4.0f || maxH / minH > 4.0f) return false

            return true
        } finally {
            contour.release()
        }
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

        return lines.filter { line ->
            circularAngleDiff(line.angle, dominantAngle) <= 15.0
        }
    }

    private fun applySubPixelRefinement(gray: org.opencv.core.Mat, roi: org.opencv.core.Rect, points: List<PointF>): List<PointF> {
        val globalPoints = points.map { PointF(it.x + roi.x, it.y + roi.y) }
        val sorted = sortCornersStandard(globalPoints)
        
        val winSizeInt = max(3, roi.width / 50)
        val winSize = winSizeInt.toDouble()
        
        val isSafeToRefine = sorted.all { pt ->
            pt.x >= winSizeInt && pt.y >= winSizeInt &&
            pt.x < (gray.cols() - winSizeInt) && pt.y < (gray.rows() - winSizeInt)
        }

        if (!isSafeToRefine) return sorted

        val subPixMat = MatOfPoint2f()
        subPixMat.fromArray(*sorted.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        
        return try {
            val searchWindow = org.opencv.core.Size(winSize, winSize)
            val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.01)
            
            Imgproc.cornerSubPix(gray, subPixMat, searchWindow, org.opencv.core.Size(-1.0, -1.0), criteria)
            subPixMat.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        } catch (e: Exception) { sorted } finally { subPixMat.release() }
    }

    private fun mergeLinesLinearRegression(lines: List<Line>, isHorizontal: Boolean, roiWidth: Int, roiHeight: Int): Line? {
        if (lines.isEmpty()) return null
        if (lines.size == 1) return lines[0]
        var sumW = 0.0
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumX2 = 0.0
        var sumY2 = 0.0

        for (line in lines) {
            val w = line.length
            sumW += w * 2.0
            sumX += w * line.x1
            sumY += w * line.y1
            sumXY += w * line.x1 * line.y1
            sumX2 += w * line.x1 * line.x1
            sumY2 += w * line.y1 * line.y1
            sumX += w * line.x2
            sumY += w * line.y2
            sumXY += w * line.x2 * line.y2
            sumX2 += w * line.x2 * line.x2
            sumY2 += w * line.y2 * line.y2
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
        val a1 = line1.y2 - line1.y1
        val b1 = line1.x1 - line1.x2
        val c1 = a1 * line1.x1 + b1 * line1.y1
        val a2 = line2.y2 - line2.y1
        val b2 = line2.x1 - line2.x2
        val c2 = a2 * line2.x1 + b2 * line2.y1
        val det = a1 * b2 - a2 * b1
        if (abs(det) < 1e-6) return null
        return PointF(((b2 * c1 - b1 * c2) / det).toFloat(), ((a1 * c2 - a2 * c1) / det).toFloat())
    }

    private fun sortCornersStandard(pts: List<PointF>): List<PointF> {
        if (pts.size != 4) return pts
        val cx = pts.map { it.x }.average().toFloat()
        val cy = pts.map { it.y }.average().toFloat()

        val radialSorted = pts.sortedBy { atan2(it.y - cy, it.x - cx) }

        var startIndex = 0
        var minSum = Float.MAX_VALUE
        for (i in 0 until 4) {
            val sum = radialSorted[i].x + radialSorted[i].y
            if (sum < minSum) {
                minSum = sum
                startIndex = i
            }
        }

        return listOf(
            radialSorted[startIndex],
            radialSorted[(startIndex + 1) % 4],
            radialSorted[(startIndex + 2) % 4],
            radialSorted[(startIndex + 3) % 4]
        )
    }

    private fun rectifyToFlatPlate(sourceBitmap: Bitmap, pts: List<PointF>): Bitmap? {
        val srcMat = org.opencv.core.Mat()
        val destMat = org.opencv.core.Mat()
        var flatBitmap: Bitmap? = null
        return try {
            Utils.bitmapToMat(sourceBitmap, srcMat)
            val targetW = 400
            val targetH = 100
            val srcPts = MatOfPoint2f(*pts.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val dstPts = MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0), org.opencv.core.Point(targetW.toDouble(), 0.0),
                org.opencv.core.Point(targetW.toDouble(), targetH.toDouble()), org.opencv.core.Point(0.0, targetH.toDouble())
            )
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(srcMat, destMat, transform, org.opencv.core.Size(targetW.toDouble(), targetH.toDouble()))
            flatBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(destMat, flatBitmap)
            srcPts.release()
            dstPts.release()
            transform.release()
            flatBitmap
        } catch (e: Exception) { 
            null 
        } finally { 
            srcMat.release()
            destMat.release() 
        }
    }

    private fun processPerspectiveOverlay(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val targetMat = org.opencv.core.Mat()
        val maskMat = org.opencv.core.Mat()
        val warpedMask = org.opencv.core.Mat()
        val alphaMask = org.opencv.core.Mat()
        val finalBlended = org.opencv.core.Mat()

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

            val warpedMatFloat = org.opencv.core.Mat()
            val targetMatFloat = org.opencv.core.Mat()
            val alphaMaskFloat = org.opencv.core.Mat()

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
                warpedChannels[3].release()
                warpedChannels[3] = targetChannels[3].clone()
            }
            
            Core.merge(warpedChannels, finalBlended)
            finalBlended.convertTo(finalBlended, CvType.CV_8UC4)
            Utils.matToBitmap(finalBlended, result)

            invAlpha.release()
            targetChan.release()
            srcPts.release()
            dstPts.release()
            transform.release()
            warpedMatFloat.release()
            targetMatFloat.release()
            alphaMaskFloat.release()
            warpedChannels.forEach { it.release() }
            targetChannels.forEach { it.release() }

            return result
        } finally {
            targetMat.release()
            maskMat.release()
            warpedMask.release()
            alphaMask.release()
            finalBlended.release()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(1920, 1080)).build()
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
        val capture = imageCapture ?: run { 
            isProcessing = false
            return 
        }
        
        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    val bitmap = image.toBitmapExt()
                    
                    synchronized(bitmapLock) { 
                        lastCapturedBitmap = bitmap 
                    }
                    
                    val previewUri = saveBitmapToCacheFile(bitmap)
                    
                    runOnUiThread { 
                        if (previewUri != null) {
                            nativeBackgroundView?.setImageDrawable(null)
                            
                            val safeEscapedUri = JSONObject.quote(previewUri)
                            
                            viewFinder?.visibility = View.GONE
                            
                            val uiBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false)
                            nativeBackgroundView?.setImageBitmap(uiBitmap)
                            nativeBackgroundView?.visibility = View.VISIBLE
                            
                            webView?.evaluateJavascript("window.onNativePhotoCaptured($safeEscapedUri)", null)
                        } else {
                            isProcessing = false
                            viewFinder?.visibility = View.VISIBLE
                            nativeBackgroundView?.visibility = View.GONE
                            Toast.makeText(this@MainActivity, "이미지 캐시 생성에 실패했습니다.", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (t: Throwable) { 
                    Log.e("JiSeKa Engine", "비트맵 처리 또는 저장 중 크래시 발생", t)
                    runOnUiThread {
                        isProcessing = false
                        viewFinder?.visibility = View.VISIBLE
                        nativeBackgroundView?.visibility = View.GONE
                        Toast.makeText(this@MainActivity, "사진 처리 중 문제가 발생했습니다.", Toast.LENGTH_SHORT).show()
                    }
                } finally {
                    image.close()
                }
            }

            override fun onError(e: ImageCaptureException) { 
                runOnUiThread {
                    isProcessing = false
                    Toast.makeText(this@MainActivity, "캡처 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    private fun ImageProxy.toBitmapExt(): Bitmap {
        val bitmap = when (format) {
            ImageFormat.JPEG -> {
                val buffer = planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    ?: throw RuntimeException("JPEG decode failed")
            }
            ImageFormat.YUV_420_888 -> {
                if (planes.size < 3) {
                    throw RuntimeException("Invalid YUV planes: ${planes.size}")
                }
                
                val yPlane = planes[0]
                val uPlane = planes[1]
                val vPlane = planes[2]

                val yBuffer = yPlane.buffer
                val uBuffer = uPlane.buffer
                val vBuffer = vPlane.buffer

                val nv21 = ByteArray(width * height * 3 / 2)
                var pos = 0

                val yRowStride = yPlane.rowStride
                for (row in 0 until height) {
                    yBuffer.position(row * yRowStride)
                    yBuffer.get(nv21, pos, width)
                    pos += width
                }

                val uvHeight = height / 2
                val uvWidth = width / 2
                val vRowStride = vPlane.rowStride
                val vPixelStride = vPlane.pixelStride
                val uRowStride = uPlane.rowStride
                val uPixelStride = uPlane.pixelStride

                for (row in 0 until uvHeight) {
                    for (col in 0 until uvWidth) {
                        vBuffer.position(row * vRowStride + col * vPixelStride)
                        nv21[pos++] = vBuffer.get()
                        uBuffer.position(row * uRowStride + col * uPixelStride)
                        nv21[pos++] = uBuffer.get()
                    }
                }
                
                val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
                
                ByteArrayOutputStream().use { out ->
                    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
                    BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
                } ?: throw RuntimeException("YUV decode failed")
            }
            else -> {
                throw RuntimeException("Unsupported image format: $format")
            }
        }

        val isLandscape = imageInfo.rotationDegrees % 180 == 0
        val targetW = if (isLandscape) 1920 else 1080
        val targetH = if (isLandscape) 1080 else 1920
        
        val scaledBitmap = scaleBitmapDownToLimit(bitmap, targetW, targetH)
        
        if (scaledBitmap !== bitmap) {
            bitmap.recycle()
        }

        val matrix = Matrix().apply { 
            postRotate(imageInfo.rotationDegrees.toFloat()) 
        }
        
        val rotatedBitmap = Bitmap.createBitmap(
            scaledBitmap, 
            0, 
            0, 
            scaledBitmap.width, 
            scaledBitmap.height, 
            matrix, 
            true
        )
        
        if (rotatedBitmap !== scaledBitmap) {
            scaledBitmap.recycle()
        }

        return rotatedBitmap
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "JiSeKa_${System.currentTimeMillis()}.jpg")
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/JiSeKa")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        uri?.let { contentResolver.openOutputStream(it)?.use { stream -> bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream) } }
        runOnUiThread { Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() { 
        super.onDestroy()
        cameraExecutor.shutdown()
        analysisExecutor.shutdown()
        recognizer.close()
        try { cachedPreviewFile?.let { if (it.exists()) it.delete() } } catch (e: Exception) {}
        synchronized(bitmapLock) { 
            lastCapturedBitmap = null 
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
