package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PointF
import android.media.ExifInterface
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
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
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    data class Line(val x1: Double, val y1: Double, val x2: Double, val y2: Double, val length: Double, val angle: Double)

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

        if (!OpenCVLoader.initDebug()) Log.e("JiSeKa Engine", "OpenCV 초기화 실패")

        viewFinder = findViewById(R.id.viewFinder)
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        webView = findViewById(R.id.webView)
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()

        previewDir = File(filesDir, "preview")
        if (!previewDir.exists()) previewDir.mkdirs()

        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/preview/", WebViewAssetLoader.InternalStoragePathHandler(this, previewDir))
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
                try { webView?.evaluateJavascript(js, null) } catch (e: Exception) { Log.e("JiSeKa Engine", "JS evaluate 실패", e) }
            }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView?.apply {
            setLayerType(View.LAYER_TYPE_SOFTWARE, null)
            setBackgroundColor(Color.TRANSPARENT)
            settings.apply {
                javaScriptEnabled = true; domStorageEnabled = true; allowFileAccess = true; allowContentAccess = true
                allowFileAccessFromFileURLs = true; allowUniversalAccessFromFileURLs = true; mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            }
            webViewClient = object : WebViewClientCompat() {
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? = assetLoader.shouldInterceptRequest(request.url)
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
                viewFinder?.visibility = if (isVisible) View.VISIBLE else View.GONE
                nativeBackgroundView?.visibility = View.GONE
                if (isVisible) isProcessing = false
            }
        }

        @JavascriptInterface
        fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            analysisExecutor.execute {
                if (isDestroyed || isFinishing) return@execute
                
                var processingBitmap: Bitmap? = null
                var guideBitmap: Bitmap? = null
                
                try {
                    val rawBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } ?: return@execute
                    processingBitmap = scaleBitmapDownToLimit(rawBitmap, 1920, 1080)
                    if (processingBitmap !== rawBitmap) rawBitmap.recycle()

                    val bmpW = processingBitmap.width.toFloat()
                    val bmpH = processingBitmap.height.toFloat()

                    val inputCorners = JSONObject(cornersJsonStr).getJSONArray("corners")
                    val mappedPoints = mutableListOf<PointF>()
                    for (i in 0 until 4) {
                        val p = inputCorners.getJSONObject(i)
                        mappedPoints.add(PointF(p.getDouble("x").toFloat() * bmpW, p.getDouble("y").toFloat() * bmpH))
                    }

                    // 1단계: 10% 여유 크롭
                    val xs = mappedPoints.map { it.x }; val ys = mappedPoints.map { it.y }
                    val padX = ((xs.maxOrNull()!! - xs.minOrNull()!!) * 0.10).toInt()
                    val padY = ((ys.maxOrNull()!! - ys.minOrNull()!!) * 0.10).toInt()
                    val guideRect = android.graphics.Rect(
                        max(0, xs.minOrNull()!!.toInt() - padX), max(0, ys.minOrNull()!!.toInt() - padY),
                        min(processingBitmap.width, xs.maxOrNull()!!.toInt() + padX), min(processingBitmap.height, ys.maxOrNull()!!.toInt() + padY)
                    )
                    guideBitmap = Bitmap.createBitmap(processingBitmap, guideRect.left, guideRect.top, guideRect.width(), guideRect.height())

                    // 2단계: OCR 텍스트 앵커 확보
                    recognizer.process(InputImage.fromBitmap(guideBitmap!!, 0)).addOnCompleteListener { task ->
                        analysisExecutor.execute {
                            if (isDestroyed || isFinishing) { guideBitmap.recycle(); processingBitmap?.recycle(); return@execute }
                            
                            var finalPoints: List<PointF> = mappedPoints 
                            try {
                                if (task.isSuccessful && task.result.textBlocks.isNotEmpty()) {
                                    var tMinX = Int.MAX_VALUE; var tMinY = Int.MAX_VALUE
                                    var tMaxX = 0; var tMaxY = 0
                                    var hasValidText = false
                                    for (block in task.result.textBlocks) {
                                        if (isValidLicensePlatePattern(block.text)) {
                                            hasValidText = true
                                            block.boundingBox?.let { 
                                                tMinX = min(tMinX, it.left); tMinY = min(tMinY, it.top)
                                                tMaxX = max(tMaxX, it.right); tMaxY = max(tMaxY, it.bottom)
                                            }
                                        }
                                    }
                                    // 3단계: 30% 영역 확장
                                    if (hasValidText) {
                                        val ePadX = ((tMaxX - tMinX) * 0.30f).toInt()
                                        val ePadY = ((tMaxY - tMinY) * 0.30f).toInt()
                                        val expandedRect = org.opencv.core.Rect(
                                            max(0, tMinX - ePadX), max(0, tMinY - ePadY),
                                            min(guideBitmap.width - tMinX, (tMaxX - tMinX) + 2 * ePadX),
                                            min(guideBitmap.height - tMinY, (tMaxY - tMinY) + 2 * ePadY)
                                        )
                                        val refined = performTextAnchoredEdgeDetection(guideBitmap, expandedRect)
                                        if (refined != null) finalPoints = refined.map { PointF(it.x + guideRect.left, it.y + guideRect.top) }
                                    }
                                }
                                val previewBitmap = processPerspectiveOverlay(processingBitmap!!, finalPoints)
                                val fileUriStr = saveBitmapToCacheFile(previewBitmap)
                                sendCachedPreviewToJs(finalPoints, fileUriStr, bmpW, bmpH)
                                previewBitmap.recycle()
                            } catch (e: Exception) {
                                Log.e("JiSeKa Engine", "비동기 파이프라인 크래시", e)
                                val fallbackUri = saveBitmapToCacheFile(processingBitmap!!)
                                sendCachedPreviewToJs(mappedPoints, fallbackUri, bmpW, bmpH)
                            } finally { guideBitmap.recycle(); processingBitmap?.recycle() }
                        }
                    }
                } catch (e: Exception) { Log.e("JiSeKa Engine", "에러 발생", e); isProcessing = false }
            }
        }
        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            analysisExecutor.execute {
                var base: Bitmap? = null; var res: Bitmap? = null
                try {
                    base = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } ?: return@execute
                    val pts = mutableListOf<PointF>()
                    val corners = JSONObject(cornersJsonStr).getJSONArray("corners")
                    for (i in 0 until 4) {
                        val p = corners.getJSONObject(i)
                        pts.add(PointF(p.getDouble("x").toFloat() * base.width, p.getDouble("y").toFloat() * base.height))
                    }
                    res = processPerspectiveOverlay(base, pts)
                    saveBitmapToGallery(res!!)
                    safeEvaluate("window.onNativeSaveComplete()")
                } finally { base?.recycle(); res?.let { if (it !== base) it.recycle() } }
            }
        }
        @JavascriptInterface
        fun showToast(msg: String) { runOnUiThread { if (!isDestroyed && !isFinishing) Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() } }
    }

    private fun performTextAnchoredEdgeDetection(bitmap: Bitmap, expandedRect: org.opencv.core.Rect): List<PointF>? {
        val mat = Mat(); val gray = Mat(); val roiMat = Mat(); val edgeMat = Mat(); val lines = Mat()
        val contours = mutableListOf<MatOfPoint>()
        try {
            Utils.bitmapToMat(bitmap, mat); Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            val claheObj = Imgproc.createCLAHE(4.0, Size(8.0, 8.0))
            try { claheObj.apply(gray, gray) } finally { claheObj.collectGarbage() }
            gray.submat(expandedRect).copyTo(roiMat)
            Imgproc.GaussianBlur(roiMat, roiMat, Size(3.0, 3.0), 0.0)
            val lower = max(0.0, 0.66 * computeMedian(roiMat)); val upper = min(255.0, 1.33 * computeMedian(roiMat))
            Imgproc.Canny(roiMat, edgeMat, lower, upper)
            val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel); morphKernel.release()
            Imgproc.HoughLinesP(edgeMat, lines, 1.0, Math.PI / 180, 30, expandedRect.width * 0.10, 10.0)
            
            val rawTopLines = mutableListOf<Line>(); val rawBottomLines = mutableListOf<Line>()
            val rawLeftLines = mutableListOf<Line>(); val rawRightLines = mutableListOf<Line>()
            val roiCenterY = expandedRect.height / 2.0; val roiCenterX = expandedRect.width / 2.0

            for (i in 0 until lines.rows()) {
                val vec = lines.get(i, 0) ?: continue
                val dx = vec[2] - vec[0]; val dy = vec[3] - vec[1]
                val length = sqrt(dx * dx + dy * dy); var angle = atan2(dy, dx) * 180.0 / Math.PI
                if (angle < 0) angle += 180.0
                val midX = (vec[0] + vec[2]) / 2.0; val midY = (vec[1] + vec[3]) / 2.0
                val lineObj = Line(vec[0], vec[1], vec[2], vec[3], length, angle)
                if (abs(dx) > abs(dy)) { if (midY < roiCenterY) rawTopLines.add(lineObj) else rawBottomLines.add(lineObj) }
                else { if (midX < roiCenterX) rawLeftLines.add(lineObj) else rawRightLines.add(lineObj) }
            }

            val tEdge = mergeLinesLinearRegression(filterByDominantAngle(rawTopLines), true, expandedRect.width, expandedRect.height)
            val bEdge = mergeLinesLinearRegression(filterByDominantAngle(rawBottomLines), true, expandedRect.width, expandedRect.height)
            val lEdge = mergeLinesLinearRegression(filterByDominantAngle(rawLeftLines), false, expandedRect.width, expandedRect.height)
            val rEdge = mergeLinesLinearRegression(filterByDominantAngle(rawRightLines), false, expandedRect.width, expandedRect.height)

            if (tEdge != null && bEdge != null && lEdge != null && rEdge != null) {
                val tl = getIntersection(tEdge, lEdge); val tr = getIntersection(tEdge, rEdge)
                val br = getIntersection(bEdge, rEdge); val bl = getIntersection(bEdge, lEdge)
                if (tl != null && tr != null && br != null && bl != null) {
                    val candidateQuad = listOf(tl, tr, br, bl)
                    if (validateQuadrilateral(candidateQuad, expandedRect.width, expandedRect.height)) return applySubPixelRefinement(gray, expandedRect, candidateQuad)
                }
            }
            val hierarchy = Mat()
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            var bestPoints: List<PointF>? = null; var maxArea = 0.0
            for (contour in contours) {
                val m2f = MatOfPoint2f(); val approx = MatOfPoint2f()
                m2f.fromArray(*contour.toArray()); val peri = Imgproc.arcLength(m2f, true)
                Imgproc.approxPolyDP(m2f, approx, 0.02 * peri, true)
                val rawPoints = approx.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
                val quadPoints = extractFourCorners(rawPoints)
                if (quadPoints != null && validateQuadrilateral(quadPoints, expandedRect.width, expandedRect.height)) {
                    val area = abs(Imgproc.contourArea(MatOfPoint(*approx.toArray())))
                    if (area > maxArea) { maxArea = area; bestPoints = sortCornersStandard(quadPoints) }
                }
                m2f.release(); approx.release()
            }
            hierarchy.release()
            return bestPoints?.let { applySubPixelRefinement(gray, expandedRect, it) }
        } finally {
            mat.release(); gray.release(); roiMat.release(); edgeMat.release(); lines.release()
            contours.forEach { it.release() }; contours.clear()
        }
    }

    private fun validateQuadrilateral(pts: List<PointF>, imageW: Int, imageH: Int): Boolean {
        if (pts.size != 4) return false
        val ordered = sortCornersStandard(pts)
        val contour = MatOfPoint(*ordered.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        return try { Imgproc.isContourConvex(contour) && abs(Imgproc.contourArea(contour)) > (imageW * imageH * 0.1) } finally { contour.release() }
    }

    private fun computeMedian(mat: Mat): Double {
        val hist = Mat(); val ranges = org.opencv.core.MatOfFloat(0f, 256f); val histSize = org.opencv.core.MatOfInt(256)
        return try {
            Imgproc.calcHist(listOf(mat), org.opencv.core.MatOfInt(0), Mat(), hist, histSize, ranges)
            var sum = 0.0
            for (i in 0 until 256) { sum += hist.get(i, 0)[0]; if (sum >= mat.total() / 2.0) return i.toDouble() }
            127.0
        } finally { hist.release(); ranges.release(); histSize.release() }
    }

    private fun extractFourCorners(pts: List<PointF>): List<PointF>? {
        if (pts.size < 4 || pts.size > 10) return null
        if (pts.size == 4) return pts
        val tl = pts.minByOrNull { it.x + it.y }!!; val br = pts.maxByOrNull { it.x + it.y }!!
        val tr = pts.maxByOrNull { it.x - it.y }!!; val bl = pts.minByOrNull { it.x - it.y }!!
        return listOf(tl, tr, br, bl)
    }

    private fun processPerspectiveOverlay(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val targetMat = Mat(); val maskMat = Mat(); val warpedMask = Mat(); val alphaMask = Mat(); val finalBlended = Mat()
        try {
            Utils.bitmapToMat(result, targetMat)
            val maskBmp = BitmapFactory.decodeResource(resources, resources.getIdentifier("plate_mask", "drawable", packageName))
            Utils.bitmapToMat(maskBmp, maskMat); maskBmp.recycle()
            val srcPts = MatOfPoint2f(org.opencv.core.Point(0.0, 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()), org.opencv.core.Point(0.0, maskMat.rows().toDouble()))
            val dstPts = MatOfPoint2f(*sortCornersStandard(targetCorners).map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(maskMat, warpedMask, transform, targetMat.size(), Imgproc.INTER_LINEAR)
            Imgproc.cvtColor(warpedMask, alphaMask, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.threshold(alphaMask, alphaMask, 1.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.GaussianBlur(alphaMask, alphaMask, Size(7.0, 7.0), 0.0)
            val warpedChannels = mutableListOf<Mat>(); val targetChannels = mutableListOf<Mat>(); Core.split(warpedMask, warpedChannels); Core.split(targetMat, targetChannels)
            for (i in 0 until 3) { Core.multiply(warpedChannels[i], alphaMask, warpedChannels[i], 1.0/255.0); Core.multiply(targetChannels[i], Mat.ones(alphaMask.size(), CvType.CV_8U).subtract(alphaMask), targetChannels[i], 1.0/255.0); Core.add(warpedChannels[i], targetChannels[i], warpedChannels[i]) }
            Core.merge(warpedChannels, finalBlended); Utils.matToBitmap(finalBlended, result)
            srcPts.release(); dstPts.release(); transform.release(); warpedChannels.forEach { it.release() }; targetChannels.forEach { it.release() }
            return result
        } finally { targetMat.release(); maskMat.release(); warpedMask.release(); alphaMask.release(); finalBlended.release() }
    }

    private fun sortCornersStandard(pts: List<PointF>): List<PointF> {
        val cx = pts.map { it.x }.average().toFloat(); val cy = pts.map { it.y }.average().toFloat()
        val radialSorted = pts.sortedBy { atan2(it.y - cy, it.x - cx) }
        var startIndex = 0; var minSum = Float.MAX_VALUE
        for (i in 0 until 4) { if (radialSorted[i].x + radialSorted[i].y < minSum) { minSum = radialSorted[i].x + radialSorted[i].y; startIndex = i } }
        return listOf(radialSorted[startIndex], radialSorted[(startIndex + 1) % 4], radialSorted[(startIndex + 2) % 4], radialSorted[(startIndex + 3) % 4])
    }

    private fun filterByDominantAngle(lines: List<Line>): List<Line> {
        if (lines.size <= 1) return lines
        val buckets = HashMap<Int, Double>()
        for (l in lines) { val b = (((atan2(l.y2-l.y1, l.x2-l.x1)*180/Math.PI + 90) % 180 - 90 + 2.5) / 5).toInt() * 5; buckets[b] = (buckets[b] ?: 0.0) + l.length }
        val dom = buckets.maxByOrNull { it.value }?.key?.toDouble() ?: 0.0
        return lines.filter { abs(min(((atan2(it.y2-it.y1, it.x2-it.x1)*180/Math.PI + 90) % 180 - 90) - dom, 180.0 - abs(((atan2(it.y2-it.y1, it.x2-it.x1)*180/Math.PI + 90) % 180 - 90) - dom))) <= 15.0 }
    }
    
    private fun mergeLinesLinearRegression(lines: List<Line>, isH: Boolean, w: Int, h: Int): Line? {
        if (lines.isEmpty()) return null
        var sW=0.0; var sX=0.0; var sY=0.0; var sXY=0.0; var sX2=0.0; var sY2=0.0
        for (l in lines) { val wL = l.length; sW+=wL; sX+=wL*l.x1; sY+=wL*l.y1; sXY+=wL*l.x1*l.y1; sX2+=wL*l.x1*l.x1; sY2+=wL*l.y1*l.y1 }
        val mX = sX/sW; val mY = sY/sW
        return if (isH) Line(0.0, mY, w.toDouble(), mY, sW, 0.0) else Line(mX, 0.0, mX, h.toDouble(), sW, 90.0)
    }

    private fun getIntersection(l1: Line, l2: Line): PointF? {
        val a1=l1.y2-l1.y1; val b1=l1.x1-l1.x2; val c1=a1*l1.x1+b1*l1.y1
        val a2=l2.y2-l2.y1; val b2=l2.x1-l2.x2; val c2=a2*l2.x1+b2*l2.y1
        val det=a1*b2-a2*b1
        return if (abs(det)<1e-6) null else PointF(((b2*c1-b1*c2)/det).toFloat(), ((a1*c2-a2*c1)/det).toFloat())
    }

    private fun scaleBitmapDownToLimit(b: Bitmap, mW: Int, mH: Int): Bitmap {
        val sc = min(mW.toFloat()/b.width, mH.toFloat()/b.height)
        return if (sc >= 1) b else Bitmap.createScaledBitmap(b, (b.width*sc).toInt(), (b.height*sc).toInt(), true)
    }

    private fun isValidLicensePlatePattern(t: String): Boolean = t.replace(Regex("[^a-zA-Z0-9가-힣]"), "").length >= 3

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        previewBitmapRef?.recycle(); recognizer.close(); cameraExecutor.shutdownNow(); analysisExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
