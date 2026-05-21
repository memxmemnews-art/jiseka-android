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
import org.opencv.core.Rect
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
import kotlin.math.pow
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

    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

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
                try { webView?.evaluateJavascript(js, null) } catch (e: Exception) {}
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
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? = assetLoader.shouldInterceptRequest(request.url)
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest) = false
            }
            webChromeClient = WebChromeClient()
            addJavascriptInterface(AndroidBridge(), "AndroidBridge")
            loadUrl("https://ziseka-app.vercel.app/?v=" + System.currentTimeMillis())
        }
    }

    inner class AndroidBridge {
        @JavascriptInterface fun takePhoto() { this@MainActivity.takePhoto() }
        @JavascriptInterface fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                viewFinder?.visibility = if (isVisible) View.VISIBLE else View.GONE
                nativeBackgroundView?.visibility = View.GONE
                if (isVisible) isProcessing = false
            }
        }

        @JavascriptInterface fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            analysisExecutor.execute {
                if (isDestroyed || isFinishing) return@execute
                var processingBitmap: Bitmap? = null
                var guideBitmap: Bitmap? = null
                var blobBitmap: Bitmap? = null 
                
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

                    val xs = mappedPoints.map { it.x }
                    val ys = mappedPoints.map { it.y }
                    val baseMinX = max(0, xs.minOrNull()!!.toInt())
                    val baseMinY = max(0, ys.minOrNull()!!.toInt())
                    val baseMaxX = min(processingBitmap.width - 1, xs.maxOrNull()!!.toInt())
                    val baseMaxY = min(processingBitmap.height - 1, ys.maxOrNull()!!.toInt())
                    
                    val guideRect = android.graphics.Rect(baseMinX, baseMinY, baseMaxX, baseMaxY)
                    
                    if (guideRect.width() <= 10 || guideRect.height() <= 10) {
                        val fallbackUri = saveBitmapToCacheFile(processingBitmap)
                        sendCachedPreviewToJs(mappedPoints, fallbackUri, bmpW, bmpH)
                        processingBitmap.recycle()
                        return@execute
                    }

                    // 1단계: 가이드 영역 자르기
                    guideBitmap = Bitmap.createBitmap(processingBitmap, guideRect.left, guideRect.top, guideRect.width(), guideRect.height())

                    // 2단계: OpenCV 모폴로지 연산으로 글자 덩어리(Blob) 찾기 (Region Proposal)
                    val proposedBlobRect = findTextBlobs(guideBitmap)

                    if (proposedBlobRect == null) {
                        Log.w("JiSeKa", "No valid text blob found by OpenCV")
                        val fallbackUri = saveBitmapToCacheFile(processingBitmap)
                        sendCachedPreviewToJs(mappedPoints, fallbackUri, bmpW, bmpH)
                        guideBitmap.recycle()
                        processingBitmap.recycle()
                        return@execute
                    }

                    // 3단계: 제안된 덩어리 영역만 잘라내기 (ML Kit 검증용)
                    blobBitmap = Bitmap.createBitmap(guideBitmap, proposedBlobRect.x, proposedBlobRect.y, proposedBlobRect.width, proposedBlobRect.height)

                    // 4단계: ML Kit로 텍스트(1글자 이상) 유무 검증
                    recognizer.process(InputImage.fromBitmap(blobBitmap!!, 0)).addOnCompleteListener { task ->
                        analysisExecutor.execute {
                            if (isDestroyed || isFinishing) { 
                                guideBitmap?.recycle()
                                processingBitmap?.recycle()
                                blobBitmap?.recycle()
                                return@execute 
                            }
                            
                            var finalPoints: List<PointF> = mappedPoints 
        
                            try {
                                if (task.isSuccessful && task.result.text.trim().isNotEmpty()) {
                                    // 5단계: 제안된 덩어리를 상하좌우 30% 확장
                                    val ePadX = (proposedBlobRect.width * 0.30f).toInt()
                                    val ePadY = (proposedBlobRect.height * 0.30f).toInt()
                                    
                                    val startX = max(0, proposedBlobRect.x - ePadX)
                                    val startY = max(0, proposedBlobRect.y - ePadY)
                                    val endX = min(guideBitmap!!.width, proposedBlobRect.x + proposedBlobRect.width + ePadX)
                                    val endY = min(guideBitmap!!.height, proposedBlobRect.y + proposedBlobRect.height + ePadY)
                                    
                                    val expandedRect = Rect(startX, startY, endX - startX, endY - startY)
                   
                                    // 6단계: 확장된 영역 안에서 기존 OpenCV 꼭짓점(Edge) 정밀 탐지 로직 실행
                                    val refined = performTextAnchoredEdgeDetection(guideBitmap!!, expandedRect)
                                    if (refined != null) {
                                        finalPoints = refined.map { PointF(it.x + guideRect.left, it.y + guideRect.top) }
                                    }
                                } else {
                                    Log.w("JiSeKa", "ML Kit verification failed: No text found in blob")
                                }
                                
                                val previewBitmap = processPerspectiveOverlay(processingBitmap!!, finalPoints)
                                val fileUriStr = saveBitmapToCacheFile(previewBitmap)
                                sendCachedPreviewToJs(finalPoints, fileUriStr, bmpW, bmpH)
                                previewBitmap.recycle()
               
                            } catch (e: Exception) { 
                                Log.e("JiSeKa", "Async Crash", e)
                                sendCachedPreviewToJs(mappedPoints, saveBitmapToCacheFile(processingBitmap!!), bmpW, bmpH) 
                            } finally { 
                                guideBitmap?.recycle()
                                processingBitmap?.recycle()
                                blobBitmap?.recycle()
                            }
                        }
                    }
                } catch (e: Exception) { 
                    Log.e("JiSeKa", "Init Crash", e)
                    safeEvaluate("window.onNativeSuccess(null)")
                    isProcessing = false 
                }
            }
        }

        @JavascriptInterface fun saveImageWithNativeOverlay(cornersJsonStr: String) {
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
                } finally { 
                    base?.recycle()
                    res?.let { if (it !== base) it.recycle() } 
                }
            }
        }
        @JavascriptInterface fun showToast(msg: String) { runOnUiThread { if (!isDestroyed && !isFinishing) Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() } }
    }

    private fun findTextBlobs(bitmap: Bitmap): org.opencv.core.Rect? {
        val mat = Mat(); val gray = Mat(); val edge = Mat(); val morphed = Mat()
        val contours = mutableListOf<MatOfPoint>()
        
        try {
            Utils.bitmapToMat(bitmap, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            
            Imgproc.Canny(gray, edge, 50.0, 150.0)
            
            // 💡 가로로 매우 긴 커널을 생성하여 띄어쓰기된 글자들을 한 덩어리로 뭉개기
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(30.0, 5.0))
            Imgproc.morphologyEx(edge, morphed, Imgproc.MORPH_CLOSE, kernel)
            kernel.release()
            
            val hierarchy = Mat()
            Imgproc.findContours(morphed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            hierarchy.release()
            
            var bestRect: org.opencv.core.Rect? = null
            var maxArea = 0.0
            
            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                val area = rect.area()
                val aspectRatio = rect.width.toDouble() / rect.height.toDouble()
                
                // 비율 조건: 가로가 긴 직사각형 (번호판 형태)
                if (aspectRatio in 1.5..6.0 && area > bitmap.width * bitmap.height * 0.02) {
                    if (area > maxArea) {
                        maxArea = area
                        bestRect = rect
                    }
                }
            }
            return bestRect
            
        } finally {
            mat.release(); gray.release(); edge.release(); morphed.release()
            contours.forEach { it.release() }
        }
    }

    private fun performTextAnchoredEdgeDetection(bitmap: Bitmap, expandedRect: Rect): List<PointF>? {
        val mat = Mat(); val gray = Mat(); val roiMat = Mat(); val edgeMat = Mat(); val lines = Mat(); val contours = mutableListOf<MatOfPoint>()
        try {
            Utils.bitmapToMat(bitmap, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            val clahe = Imgproc.createCLAHE(4.0, Size(8.0, 8.0))
            try { clahe.apply(gray, gray) } finally { clahe.collectGarbage() }
            gray.submat(expandedRect).copyTo(roiMat)
            Imgproc.GaussianBlur(roiMat, roiMat, Size(3.0, 3.0), 0.0)
            val median = computeMedian(roiMat)
            val lower = max(0.0, 0.66 * median)
            val upper = min(255.0, 1.33 * median)
            Imgproc.Canny(roiMat, edgeMat, lower, upper)
            val morph = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morph)
            morph.release()
            Imgproc.HoughLinesP(edgeMat, lines, 1.0, Math.PI / 180, 30, expandedRect.width * 0.10, 10.0)
            
            val rawTop = mutableListOf<Line>()
            val rawBottom = mutableListOf<Line>(); val rawLeft = mutableListOf<Line>(); val rawRight = mutableListOf<Line>()
            val roiCenterY = expandedRect.height / 2.0
            val roiCenterX = expandedRect.width / 2.0
            for (i in 0 until lines.rows()) {
                val v = lines.get(i, 0) ?: continue
                val l = sqrt((v[2]-v[0]).pow(2) + (v[3]-v[1]).pow(2))
                var a = atan2(v[3]-v[1], v[2]-v[0]) * 180/Math.PI
                if(a<0) a+=180.0
                val line = Line(v[0], v[1], v[2], v[3], l, a)
                if (abs(v[2]-v[0]) > abs(v[3]-v[1])) { if ((v[1]+v[3])/2 < roiCenterY) rawTop.add(line) else rawBottom.add(line) }
                else { if ((v[0]+v[2])/2 < roiCenterX) rawLeft.add(line) else rawRight.add(line) }
            }
          
            val t = mergeLinesLinearRegression(filterByDominantAngle(rawTop), true, expandedRect.width, expandedRect.height)
            val b = mergeLinesLinearRegression(filterByDominantAngle(rawBottom), true, expandedRect.width, expandedRect.height)
            val l = mergeLinesLinearRegression(filterByDominantAngle(rawLeft), false, expandedRect.width, expandedRect.height)
            val r = mergeLinesLinearRegression(filterByDominantAngle(rawRight), false, expandedRect.width, expandedRect.height)

            if (t != null && b != null && l != null && r != null) {
                val tl = getIntersection(t, l); val tr = getIntersection(t, r)
                val br = getIntersection(b, r); val bl = getIntersection(b, l)
                if (tl != null && tr != null && br != null && bl != null) {
                    val quad = listOf(tl, tr, br, bl)
                    if (validateQuadrilateral(quad, expandedRect.width, expandedRect.height)) return applySubPixelRefinement(gray, expandedRect, quad)
                }
            }
            val hierarchy = Mat()
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            var best: List<PointF>? = null; var maxArea = 0.0
            for (c in contours) {
                val m2f = MatOfPoint2f()
                val approx = MatOfPoint2f(); m2f.fromArray(*c.toArray())
                Imgproc.approxPolyDP(m2f, approx, 0.02 * Imgproc.arcLength(m2f, true), true)
                val quad = extractFourCorners(approx.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) })
                if (quad != null && validateQuadrilateral(quad, expandedRect.width, expandedRect.height)) {
                    val area = abs(Imgproc.contourArea(MatOfPoint(*approx.toArray())))
                    if (area > maxArea) { maxArea = area; best = sortCornersStandard(quad) }
                }
                m2f.release()
                approx.release()
            }
            hierarchy.release()
            return best?.let { applySubPixelRefinement(gray, expandedRect, it) }
        } finally { mat.release(); gray.release(); roiMat.release(); edgeMat.release(); lines.release(); contours.forEach { it.release() } }
    }

    private fun validateQuadrilateral(pts: List<PointF>, imageW: Int, imageH: Int): Boolean {
        if (pts.size != 4) return false
        val ordered = sortCornersStandard(pts)
        val contour = MatOfPoint(*ordered.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        return try { Imgproc.isContourConvex(contour) && abs(Imgproc.contourArea(contour)) > (imageW * imageH * 0.1) } finally { contour.release() }
    }

    private fun applySubPixelRefinement(gray: Mat, roi: Rect, points: List<PointF>): List<PointF> {
        val global = points.map { PointF(it.x + roi.x, it.y + roi.y) }
        val sorted = sortCornersStandard(global)
        if (gray.empty()) return sorted
        val subPixMat = MatOfPoint2f().apply { fromArray(*sorted.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()) }
        try {
            Imgproc.cornerSubPix(gray, subPixMat, Size(5.0, 5.0), Size(-1.0, -1.0), TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.01))
            return subPixMat.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        } finally { subPixMat.release() }
    }

    private fun computeMedian(mat: Mat): Double {
        val hist = Mat()
        val ranges = org.opencv.core.MatOfFloat(0f, 256f); val histSize = org.opencv.core.MatOfInt(256)
        return try { Imgproc.calcHist(listOf(mat), org.opencv.core.MatOfInt(0), Mat(), hist, histSize, ranges); var sum = 0.0; for (i in 0 until 256) { sum += hist.get(i, 0)[0]
                if (sum >= mat.total() / 2.0) return i.toDouble() }; 127.0 } finally { hist.release(); ranges.release();
            histSize.release() }
    }
    
    private fun extractFourCorners(pts: List<PointF>): List<PointF>? {
        if (pts.size < 4 || pts.size > 10) return null
        if (pts.size == 4) return pts
        val tl = pts.minByOrNull { it.x + it.y }!!
        val br = pts.maxByOrNull { it.x + it.y }!!
        val tr = pts.maxByOrNull { it.x - it.y }!!
        val bl = pts.minByOrNull { it.x - it.y }!!
        return listOf(tl, tr, br, bl)
    }

    private fun sortCornersStandard(pts: List<PointF>): List<PointF> {
        val cx = pts.map { it.x }.average().toFloat()
        val cy = pts.map { it.y }.average().toFloat()
        val sorted = pts.sortedBy { atan2(it.y - cy, it.x - cx) }
        var start = 0
        var minSum = Float.MAX_VALUE
        for (i in 0 until 4) { if (sorted[i].x + sorted[i].y < minSum) { minSum = sorted[i].x + sorted[i].y; start = i } }
        return listOf(sorted[start], sorted[(start+1)%4], sorted[(start+2)%4], sorted[(start+3)%4])
    }
    
    private fun mergeLinesLinearRegression(lines: List<Line>, isH: Boolean, w: Int, h: Int): Line? {
        if (lines.isEmpty()) return null
        var sW=0.0
        var sX=0.0; var sY=0.0; var sXY=0.0; var sX2=0.0
        for (l in lines) { val wL = l.length; sW+=wL; sX+=wL*l.x1; sY+=wL*l.y1; sXY+=wL*l.x1*l.y1; sX2+=wL*l.x1*l.x1 }
        val mX = sX/sW
        val mY = sY/sW
        return if (isH) Line(0.0, mY, w.toDouble(), mY, sW, 0.0) else Line(mX, 0.0, mX, h.toDouble(), sW, 90.0)
    }

    private fun getIntersection(l1: Line, l2: Line): PointF? {
        val det = (l1.x1-l1.x2)*(l2.y1-l2.y2) - (l1.y1-l1.y2)*(l2.x1-l2.x2)
        return if (abs(det)<1e-6) null else PointF(((l1.x1*l1.y2-l1.y1*l1.x2)*(l2.x1-l2.x2)-(l1.x1-l1.x2)*(l2.x1*l2.y2-l2.y1*l2.x2)/det).toFloat(), ((l1.x1*l1.y2-l1.y1*l1.x2)*(l2.y1-l2.y2)-(l1.y1-l1.y2)*(l2.x1*l2.y2-l2.y1*l2.x2)/det).toFloat())
    }

    private fun filterByDominantAngle(lines: List<Line>): List<Line> {
        if (lines.size <= 1) return lines
        val buckets = HashMap<Int, Double>()
        for (l in lines) { val b = (((atan2(l.y2-l.y1, l.x2-l.x1)*180/Math.PI + 90) % 180 - 90 + 2.5) / 5).toInt() * 5; buckets[b] = (buckets[b] ?: 0.0) + l.length }
        val dom = buckets.maxByOrNull { it.value }?.key?.toDouble() ?: 0.0
        return lines.filter { abs(min(((atan2(it.y2-it.y1, it.x2-it.x1)*180/Math.PI + 90) % 180 - 90) - dom, 180.0 - abs(((atan2(it.y2-it.y1, it.x2-it.x1)*180/Math.PI + 90) % 180 - 90) - dom))) <= 15.0 }
    }

    private fun processPerspectiveOverlay(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val targetMat = Mat()
        val maskMat = Mat(); val warpedMask = Mat(); val alphaMask = Mat()
        val finalBlended = Mat()
        try {
            Utils.bitmapToMat(result, targetMat)
            val resId = resources.getIdentifier("plate_mask", "drawable", packageName)
            val maskBmp = if (resId != 0) BitmapFactory.decodeResource(resources, resId) else Bitmap.createBitmap(600, 150, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.LTGRAY) }
            Utils.bitmapToMat(maskBmp, maskMat); maskBmp.recycle()
            
            val srcPts = MatOfPoint2f(org.opencv.core.Point(0.0, 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()), org.opencv.core.Point(0.0, maskMat.rows().toDouble()))
            val dstPts = MatOfPoint2f(*sortCornersStandard(targetCorners).map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            
            Imgproc.warpPerspective(maskMat, warpedMask, transform, targetMat.size(), Imgproc.INTER_LINEAR)
   
            Imgproc.cvtColor(warpedMask, alphaMask, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.threshold(alphaMask, alphaMask, 1.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.GaussianBlur(alphaMask, alphaMask, Size(7.0, 7.0), 0.0)
            
            val warpedChannels = mutableListOf<Mat>()
            val targetChannels = mutableListOf<Mat>()
            warpedMask.convertTo(warpedMask, CvType.CV_32FC4); targetMat.convertTo(targetMat, CvType.CV_32FC4)
            alphaMask.convertTo(alphaMask, CvType.CV_32FC1, 1.0 / 255.0)
            Core.split(warpedMask, warpedChannels)
            Core.split(targetMat, targetChannels)

            // 💡 1. 32FC1 Float 정밀도 행렬에 맞는 올바른 빼기 기반 행렬 반전 (1.0 - alpha)
            val inverseMask = Mat()
            val ones = Mat(alphaMask.size(), CvType.CV_32FC1, org.opencv.core.Scalar(1.0))
            Core.subtract(ones, alphaMask, inverseMask)

            // 💡 2. RGB 뿐만 아니라 알파 채널(Index 3)까지 완벽히 동일 연산하여 ARGB 일관성 100% 확보
            val numChannels = min(warpedChannels.size, targetChannels.size)
            for (i in 0 until numChannels) {
                Core.multiply(warpedChannels[i], alphaMask, warpedChannels[i])
                // 💡 3. 배경 픽셀 파괴 주범이었던 1.0/255.0 스케일 배율 인자 완벽 제거
                Core.multiply(targetChannels[i], inverseMask, targetChannels[i]) 
                Core.add(warpedChannels[i], targetChannels[i], warpedChannels[i])
            }

            Core.merge(warpedChannels, finalBlended)
            finalBlended.convertTo(finalBlended, CvType.CV_8UC4)
            Utils.matToBitmap(finalBlended, result)
     
            srcPts.release()
            dstPts.release(); transform.release(); inverseMask.release(); ones.release()
            warpedChannels.forEach { it.release() }
            targetChannels.forEach { it.release() }
            return result
        } finally { targetMat.release()
            maskMat.release(); warpedMask.release(); alphaMask.release(); finalBlended.release() }
    }

    private fun scaleBitmapDownToLimit(b: Bitmap, mW: Int, mH: Int): Bitmap {
        val sc = min(mW.toFloat()/b.width, mH.toFloat()/b.height)
        return if (sc >= 1) b else Bitmap.createScaledBitmap(b, (b.width*sc).toInt(), (b.height*sc).toInt(), true)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null 
        }
        previewBitmapRef?.recycle(); recognizer.close(); cameraExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        previewDir.listFiles()?.forEach { it.delete() }
        webView?.apply { stopLoading()
            clearHistory(); removeAllViews(); destroy() }
        webView = null
        super.onDestroy()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build()
   
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) { Log.e("JiSeKa Engine", "카메라 바인딩 실패", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        if (isProcessing) return
        isProcessing = true

        val imageCapture = imageCapture ?: run {
            isProcessing = false
            return
        }
        
        val photoFile = File(previewDir, "capture_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        
        imageCapture.takePicture(
            outputOptions, cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    try {
                        val bitmap = loadSafeBitmap(photoFile.absolutePath, viewFinder?.width ?: 1080, viewFinder?.height ?: 1920)
                        synchronized(bitmapLock) {
                            lastCapturedBitmap?.recycle()
                            lastCapturedBitmap = bitmap
                        }
                        
                        val previewW = max(1, bitmap.width / 2)
                        val previewH = max(1, bitmap.height / 2)
                        
                        previewBitmapRef?.recycle()
                        previewBitmapRef = Bitmap.createScaledBitmap(bitmap, previewW, previewH, true)
                        
                        runOnUiThread { 
                            nativeBackgroundView?.setImageBitmap(previewBitmapRef)
                            nativeBackgroundView?.visibility = View.VISIBLE
                            viewFinder?.visibility = View.GONE
                            
                            val safeEscapedUri = JSONObject.quote("https://appassets.androidplatform.net/preview/${photoFile.name}") 
                            
                            safeEvaluate("window.onNativePhotoCaptured($safeEscapedUri, ${bitmap.width}, ${bitmap.height})") 
                        }
                    } catch (e: Exception) {
                        Log.e("JiSeKa Engine", "이미지 처리 실패", e)
                        runOnUiThread {
                            isProcessing = false
                            Toast.makeText(this@MainActivity, "이미지 처리 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
                
                override fun onError(exception: ImageCaptureException) {
                    Log.e("JiSeKa Engine", "촬영 실패", exception)
                    runOnUiThread { 
                        isProcessing = false
                        Toast.makeText(this@MainActivity, "캡처 실패", Toast.LENGTH_SHORT).show() 
                    }
                }
            }
        )
    }

    private fun loadSafeBitmap(path: String, maxW: Int, maxH: Int): Bitmap {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, options)
        options.inSampleSize = calculateInSampleSize(options, maxW, maxH)
        options.inJustDecodeBounds = false
        val bitmap = BitmapFactory.decodeFile(path, options) ?: throw RuntimeException("Bitmap load failed")
        val orientation = ExifInterface(path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
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
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) { inSampleSize *= 2 }
        }
        return inSampleSize
    }

    private fun saveBitmapToCacheFile(bitmap: Bitmap): String? {
        val file = File(previewDir, "preview_${UUID.randomUUID()}.jpg")
        FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        return "https://appassets.androidplatform.net/preview/${file.name}"
    }

    private fun sendCachedPreviewToJs(points: List<PointF>, uri: String?, bmpW: Float, bmpH: Float) {
        val arr = JSONArray()
        for (p in points) { val obj = JSONObject()
            obj.put("x", p.x / bmpW); obj.put("y", p.y / bmpH); arr.put(obj) }
        val result = JSONObject()
        result.put("corners", arr); result.put("preview", uri ?: JSONObject.NULL)
        safeEvaluate("window.onNativeSuccess(${JSONObject.quote(result.toString())})")
        isProcessing = false
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "JiSeKa_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/JiSeKa")
        }
    
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        runOnUiThread { Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "카메라 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
