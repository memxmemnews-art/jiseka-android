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
import java.io.OutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private var webView: WebView? = null
    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    // 스레드 경합 방지를 위한 메인 소스 비트맵 유지 변수
    private var lastCapturedBitmap: Bitmap? = null
    private val bitmapLock = Any()

    // 로컬 파일 캐싱을 위한 파일 객체 관리
    private var cachedPreviewFile: File? = null

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

        // 캐시 디렉토리 초기화
        cachedPreviewFile = File(externalCacheDir ?: cacheDir, "preview_cache.jpg")

        viewFinder?.apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
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
                // 로컬 파일 접근 허용 강제 (file:// 프리뷰 로드용)
                allowFileAccessFromFileURLs = true
                allowUniversalAccessFromFileURLs = true
            }
            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?) = false
            }
            webChromeClient = WebChromeClient()
            addJavascriptInterface(AndroidBridge(), "AndroidBridge")
            loadUrl("file:///android_asset/index.html")
        }
    }

    inner class AndroidBridge {
        @JavascriptInterface
        fun takePhoto() { this@MainActivity.takePhoto() }

        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread { 
                if (isVisible) {
                    viewFinder?.visibility = View.VISIBLE
                    nativeBackgroundView?.visibility = View.GONE
                } else {
                    viewFinder?.visibility = View.INVISIBLE
                }
            }
        }

        /**
         * 🚨 [TOP 1 & 2 해결] JS 수신부: 정규화 좌표를 실제 비트맵 좌표로 역매핑하고 파일 URI를 반환합니다.
         */
        @JavascriptInterface
        fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            analysisExecutor.execute {
                var processingBitmap: Bitmap? = null
                try {
                    // 1. 스레드 세이프 원본 복사본 생성
                    val rawBitmap = synchronized(bitmapLock) { 
                        lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
                    } ?: return@execute

                    // 🚨 [TOP 3 해결] OOM 방지를 위해 최대 1920x1080 이하로 스케일링 제한 강제
                    processingBitmap = scaleBitmapDownToLimit(rawBitmap, 1920, 1080)
                    if (processingBitmap !== rawBitmap) {
                        rawBitmap.recycle() // 원본 크기 임시 비트맵 조기 해제
                    }

                    val bmpW = processingBitmap.width.toFloat()
                    val bmpH = processingBitmap.height.toFloat()
                    
                    // 🚨 [TOP 1 해결] 뷰파인더 해상도 확보 (스레드 동기화를 위해 변수 사전 복사)
                    val viewW = viewFinder?.width?.toFloat() ?: 1f
                    val viewH = viewFinder?.height?.toFloat() ?: 1f

                    val inputCorners = JSONObject(cornersJsonStr).getJSONArray("corners")
                    val mappedPoints = mutableListOf<PointF>()

                    // 🚨 [TOP 1 핵심 로직] Letterbox/Pillarbox 영역을 수식으로 도출하여 원본 픽셀로 다이렉트 역변환
                    val scale = min(viewW / bmpW, viewH / bmpH)
                    val displayedW = bmpW * scale
                    val displayedH = bmpH * scale
                    val offsetX = (viewW - displayedW) / 2f
                    val offsetY = (viewH - displayedH) / 2f

                    for (i in 0 until 4) {
                        val p = inputCorners.getJSONObject(i)
                        val normX = p.getDouble("x").toFloat()
                        val normY = p.getDouble("y").toFloat()

                        // 웹뷰 절대 픽셀 좌표 도출
                        val screenPixelX = normX * viewW
                        val screenPixelY = normY * viewH

                        // 오버레이 영역 바운더리 제거 후 실제 비트맵 내부 픽셀로 환산
                        val targetBmpX = ((screenPixelX - offsetX) / scale).coerceIn(0f, bmpW - 1f)
                        val targetBmpY = ((screenPixelY - offsetY) / scale).coerceIn(0f, bmpH - 1f)
                        mappedPoints.add(PointF(targetBmpX, targetBmpY))
                    }

                    // 허프 변환 직선 피팅을 통한 미세 모서리 정제 가동
                    val refinedPoints = extractPlateCornersViaLineFitting(processingBitmap, mappedPoints)

                    // OCR 검증 및 로컬 캐시 파일 프리뷰 생성 파이프라인 가동
                    verifyAndGenerateFilePreview(processingBitmap, refinedPoints, mappedPoints, bmpW, bmpH)

                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "분석 파이프라인 크래시 방어", e)
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSuccess('{\"corners\":null, \"preview\":null}')", null) }
                } finally {
                    // 🚨 [TOP 3 해결] 연산 전용 비트맵 안전 해제 (UI용 lastCapturedBitmap은 유지됨)
                    processingBitmap?.recycle()
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

                    // 저장 시점의 좌표 환산 (현재 캐싱된 좌표는 이미 원본 픽셀 단위로 환산되어 있음)
                    val corners = jsonObj.getJSONArray("corners")
                    val pts = mutableListOf<PointF>()
                    for (i in 0 until 4) {
                        val p = corners.getJSONObject(i)
                        // 최종 저장 시 원본 비트맵 해상도 비율로 복원
                        val absX = p.getDouble("x").toFloat() * baseBitmap.width
                        val absY = p.getDouble("y").toFloat() * baseBitmap.height
                        pts.add(PointF(absX, absY))
                    }

                    overlayResult = processPerspectiveOverlay(baseBitmap, pts)
                    saveBitmapToGallery(overlayResult!!)
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSaveComplete()", null) }
                } finally { 
                    baseBitmap?.recycle()
                    overlayResult?.let { if (it !== baseBitmap) it.recycle() }
                }
            }
        }
    }

    /**
     * 🚨 [TOP 3 해결] OOM 방지를 위한 해상도 제한 다운샘플링 수식
     */
    private fun scaleBitmapDownToLimit(bitmap: Bitmap, maxW: Int, maxH: Int): Bitmap {
        val oW = bitmap.width
        val oH = bitmap.height
        if (oW <= maxW && oH <= maxH) return bitmap

        val scale = min(maxW.toFloat() / oW, maxH.toFloat() / oH)
        val scaledW = (oW * scale).toInt()
        val scaledH = (oH * scale).toInt()

        return Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
    }

    /**
     * 🚨 [TOP 2 해결] 파일 URI 기반의 초고속 프리뷰 파이프라인
     * Base64 변환을 완벽히 제거하고 로컬 디스크 파일로 저장하여 통신 부하를 소멸시킵니다.
     */
    private fun verifyAndGenerateFilePreview(
        sourceBitmap: Bitmap, 
        targetPoints: List<PointF>, 
        fallbackPoints: List<PointF>, 
        bmpW: Float, 
        bmpH: Float
    ) {
        val flatBmp = rectifyToFlatPlate(sourceBitmap, targetPoints)
        if (flatBmp != null) {
            val inputImage = InputImage.fromBitmap(flatBmp, 0)
            recognizer.process(inputImage).addOnCompleteListener { task ->
                val finalPoints = if (task.isSuccessful && isValidLicensePlatePattern(task.result.text)) {
                    Log.d("JiSeKa Engine", "✅ 정밀 좌표 피팅 완료")
                    targetPoints
                } else {
                    Log.w("JiSeKa Engine", "⚠️ 텍스트 유효성 미달. 드롭 영역으로 폴백")
                    fallbackPoints
                }
                
                // 가림막이 밀착 렌더링된 프리뷰용 비트맵 생성
                val previewBitmap = processPerspectiveOverlay(sourceBitmap, finalPoints)
                
                // 🚨 로컬 캐시 파일로 즉시 덮어쓰기 저장
                val fileUriStr = saveBitmapToCacheFile(previewBitmap)
                
                sendCachedPreviewToJs(finalPoints, fileUriStr, bmpW, bmpH)
                
                previewBitmap.recycle()
                flatBmp.recycle()
            }
        } else {
            val fileUriStr = saveBitmapToCacheFile(sourceBitmap)
            sendCachedPreviewToJs(fallbackPoints, fileUriStr, bmpW, bmpH)
        }
    }

    /**
     * 🚨 [TOP 2 해결] 비트맵을 로컬 캐시 파일로 렌더링하고 파일 경로 스트링 반환
     */
    private fun saveBitmapToCacheFile(bitmap: Bitmap): String? {
        return try {
            cachedPreviewFile?.let { file ->
                // 이전 파일 잔상 제거
                if (file.exists()) file.delete()
                
                FileOutputStream(file).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
                    out.flush()
                }
                // 브라우저 캐시 무효화를 위한 타임스탬프 파라미터 삽입
                "file://${file.absolutePath}?t=${System.currentTimeMillis()}"
            }
        } catch (e: Exception) {
            Log.e("JiSeKa Engine", "캐시 파일 저장 실패", e)
            null
        }
    }

    // 🚨 JS로 파일 URI 경로와 정규화 비율 데이터를 전송
    private fun sendCachedPreviewToJs(points: List<PointF>, fileUriStr: String?, bmpW: Float, bmpH: Float) {
        val outputArray = JSONArray()
        for (pt in points) {
            // JS 내부 저장을 위해 0.0~1.0 사이의 절대 정규화 비율로 변환하여 전달
            outputArray.put(JSONObject().put("x", (pt.x / bmpW).toDouble()).put("y", (pt.y / bmpH).toDouble()))
        }
        
        val resultJson = JSONObject().apply {
            put("corners", outputArray)
            put("preview", fileUriStr ?: JSONObject.NULL)
        }.toString()
        
        runOnUiThread { 
            webView?.evaluateJavascript("window.onNativeSuccess('$resultJson')", null) 
        }
    }

    private fun isValidLicensePlatePattern(text: String): Boolean {
        val cleanText = text.replace(Regex("\\s+"), "")
        if (cleanText.length < 3) return false
        val strictPattern = Regex("\\d{2,3}[가-힣]\\d{4}")
        if (strictPattern.containsMatchIn(cleanText)) return true
        val digitCount = cleanText.count { it.isDigit() }
        val hasKorean = cleanText.any { it in '가'..'힣' }
        return digitCount >= 3 && hasKorean
    }

    /**
     * 🚨 [TOP 3 해결] JNI 힙 메모리 안전 관리를 강제한 OpenCV 피팅 알고리즘
     */
    private fun extractPlateCornersViaLineFitting(bitmap: Bitmap, pts: List<PointF>): List<PointF> {
        val mat = org.opencv.core.Mat()
        val gray = org.opencv.core.Mat()
        val roiMat = org.opencv.core.Mat()
        val lines = org.opencv.core.Mat()
        var subPixMat: MatOfPoint2f? = null

        try {
            Utils.bitmapToMat(bitmap, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            
            val xs = pts.map { it.x }; val ys = pts.map { it.y }
            
            val minX = max(0, xs.minOrNull()?.toInt() ?: 0)
            val minY = max(0, ys.minOrNull()?.toInt() ?: 0)
            val maxX = min(mat.cols() - 1, xs.maxOrNull()?.toInt() ?: (mat.cols() - 1))
            val maxY = min(mat.rows() - 1, ys.maxOrNull()?.toInt() ?: (mat.rows() - 1))
            
            val roi = org.opencv.core.Rect(minX, minY, maxX - minX, maxY - minY)
            if (roi.width <= 20 || roi.height <= 20) return pts
            
            gray.submat(roi).copyTo(roiMat) // 안전한 하위 메모리 할당
            Imgproc.GaussianBlur(roiMat, roiMat, org.opencv.core.Size(5.0, 5.0), 0.0)
            
            val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(3.0, 3.0))
            Imgproc.morphologyEx(roiMat, roiMat, Imgproc.MORPH_CLOSE, morphKernel)
            morphKernel.release()
            
            Imgproc.Canny(roiMat, roiMat, 50.0, 150.0)
            Imgproc.HoughLinesP(roiMat, lines, 1.0, Math.PI / 180, 30, roi.width * 0.3, 10.0)

            val topLines = mutableListOf<Line>()
            val bottomLines = mutableListOf<Line>()
            val leftLines = mutableListOf<Line>()
            val rightLines = mutableListOf<Line>()

            val roiCenterY = roi.height / 2.0
            val roiCenterX = roi.width / 2.0

            for (i in 0 until lines.rows()) {
                val vec = lines.get(i, 0) ?: continue
                val x1 = vec[0]; val y1 = vec[1]; val x2 = vec[2]; val y2 = vec[3]
                
                val dx = x2 - x1; val dy = y2 - y1
                val angle = abs(Math.atan2(dy, dx) * 180.0 / Math.PI)
                val midX = (x1 + x2) / 2.0; val midY = (y1 + y2) / 2.0
                val length = sqrt(dx * dx + dy * dy)

                val lineObj = Line(x1, y1, x2, y2, length)

                if (angle <= 35.0 || angle >= 145.0) {
                    if (midY < roiCenterY) topLines.add(lineObj) else bottomLines.add(lineObj)
                } else if (angle in 55.0..125.0) {
                    if (midX < roiCenterX) leftLines.add(lineObj) else rightLines.add(lineObj)
                }
            }

            val topEdge = topLines.maxByOrNull { it.length }
            val bottomEdge = bottomLines.maxByOrNull { it.length }
            val leftEdge = leftLines.maxByOrNull { it.length }
            val rightEdge = rightLines.maxByOrNull { it.length }

            var fittedPoints = pts
            if (topEdge != null && bottomEdge != null && leftEdge != null && rightEdge != null) {
                val tl = getIntersection(topEdge, leftEdge)
                val tr = getIntersection(topEdge, rightEdge)
                val br = getIntersection(bottomEdge, rightEdge)
                val bl = getIntersection(bottomEdge, leftEdge)

                if (tl != null && tr != null && br != null && bl != null) {
                    val rawCorners = listOf(
                        PointF(tl.x.toFloat() + roi.x, tl.y.toFloat() + roi.y),
                        PointF(tr.x.toFloat() + roi.x, tr.y.toFloat() + roi.y),
                        PointF(br.x.toFloat() + roi.x, br.y.toFloat() + roi.y),
                        PointF(bl.x.toFloat() + roi.x, bl.y.toFloat() + roi.y)
                    )

                    val sortedCorners = sortCornersStandard(rawCorners)
                    val safePoints = sortedCorners.filter {
                        it.x >= 4f && it.y >= 4f && 
                        it.x < (gray.cols() - 4f) && 
                        it.y < (gray.rows() - 4f)
                    }

                    if (safePoints.size == 4) {
                        subPixMat = MatOfPoint2f()
                        subPixMat.fromArray(
                            org.opencv.core.Point(sortedCorners[0].x.toDouble(), sortedCorners[0].y.toDouble()),
                            org.opencv.core.Point(sortedCorners[1].x.toDouble(), sortedCorners[1].y.toDouble()),
                            org.opencv.core.Point(sortedCorners[2].x.toDouble(), sortedCorners[2].y.toDouble()),
                            org.opencv.core.Point(sortedCorners[3].x.toDouble(), sortedCorners[3].y.toDouble())
                        )

                        val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.05)
                        Imgproc.cornerSubPix(gray, subPixMat, org.opencv.core.Size(4.0, 4.0), org.opencv.core.Size(-1.0, -1.0), criteria)

                        val subPixArray = subPixMat.toArray()
                        fittedPoints = listOf(
                            PointF(subPixArray[0].x.toFloat(), subPixArray[0].y.toFloat()),
                            PointF(subPixArray[1].x.toFloat(), subPixArray[1].y.toFloat()),
                            PointF(subPixArray[2].x.toFloat(), subPixArray[2].y.toFloat()),
                            PointF(subPixArray[3].x.toFloat(), subPixArray[3].y.toFloat())
                        )
                    } else {
                        fittedPoints = sortedCorners
                    }
                }
            }
            return fittedPoints
        } finally {
            // 🚨 [TOP 3 해결] 예외 발생 여부와 무관하게 네이티브 힙 메모리 즉각 반환 강제
            mat.release(); gray.release(); roiMat.release(); lines.release()
            subPixMat?.release()
        }
    }

    data class Line(val x1: Double, val y1: Double, val x2: Double, val y2: Double, val length: Double)

    private fun getIntersection(line1: Line, line2: Line): PointF? {
        val a1 = line1.y2 - line1.y1
        val b1 = line1.x1 - line1.x2
        val c1 = a1 * line1.x1 + b1 * line1.y1

        val a2 = line2.y2 - line2.y1
        val b2 = line2.x1 - line2.x2
        val c2 = a2 * line2.x1 + b2 * line2.y1

        val det = a1 * b2 - a2 * b1
        if (abs(det) < 1e-6) return null

        val cx = (b2 * c1 - b1 * c2) / det
        val cy = (a1 * c2 - a2 * c1) / det
        return PointF(cx.toFloat(), cy.toFloat())
    }

    private fun sortCornersStandard(pts: List<PointF>): List<PointF> {
        if (pts.size != 4) return pts
        val tl = pts.minByOrNull { it.x + it.y } ?: pts[0]
        val br = pts.maxByOrNull { it.x + it.y } ?: pts[1]
        val tr = pts.maxByOrNull { it.x - it.y } ?: pts[2]
        val bl = pts.minByOrNull { it.x - it.y } ?: pts[3]
        return listOf(tl, tr, br, bl)
    }

    private fun rectifyToFlatPlate(sourceBitmap: Bitmap, pts: List<PointF>): Bitmap? {
        val srcMat = org.opencv.core.Mat()
        val destMat = org.opencv.core.Mat()
        var flatBitmap: Bitmap? = null
        try {
            Utils.bitmapToMat(sourceBitmap, srcMat)
            val targetW = 400; val targetH = 100
            val srcPts = MatOfPoint2f(
                org.opencv.core.Point(pts[0].x.toDouble(), pts[0].y.toDouble()),
                org.opencv.core.Point(pts[1].x.toDouble(), pts[1].y.toDouble()),
                org.opencv.core.Point(pts[2].x.toDouble(), pts[2].y.toDouble()),
                org.opencv.core.Point(pts[3].x.toDouble(), pts[3].y.toDouble())
            )
            val dstPts = MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0),
                org.opencv.core.Point(targetW.toDouble(), 0.0),
                org.opencv.core.Point(targetW.toDouble(), targetH.toDouble()),
                org.opencv.core.Point(0.0, targetH.toDouble())
            )
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(srcMat, destMat, transform, org.opencv.core.Size(targetW.toDouble(), targetH.toDouble()))
            
            flatBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(destMat, flatBitmap)
            
            srcPts.release(); dstPts.release(); transform.release()
            return flatBitmap
        } catch (e: Exception) { 
            flatBitmap?.recycle()
            return null 
        } finally {
            srcMat.release(); destMat.release()
        }
    }

    /**
     * 🚨 [TOP 3 해결] 렌더링용 임시 텍스처 메모리 즉각 해제 보장
     */
    private fun processPerspectiveOverlay(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val targetMat = org.opencv.core.Mat()
        val maskMat = org.opencv.core.Mat()
        val warpedMask = org.opencv.core.Mat()
        var overlayBmp: Bitmap? = null

        try {
            Utils.bitmapToMat(result, targetMat)
            val resId = resources.getIdentifier("plate_mask", "drawable", packageName)
            val maskBmp = if (resId != 0) BitmapFactory.decodeResource(resources, resId) 
                          else Bitmap.createBitmap(600, 150, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.LTGRAY) }
            
            Utils.bitmapToMat(maskBmp, maskMat)
            maskBmp.recycle() // 소스 마스크 텍스처 조기 회수

            val srcPts = MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), 0.0),
                org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()), org.opencv.core.Point(0.0, maskMat.rows().toDouble())
            )
            val sortedTargets = sortCornersStandard(targetCorners)
            val dstPts = MatOfPoint2f(
                org.opencv.core.Point(sortedTargets[0].x.toDouble(), sortedTargets[0].y.toDouble()),
                org.opencv.core.Point(sortedTargets[1].x.toDouble(), sortedTargets[1].y.toDouble()),
                org.opencv.core.Point(sortedTargets[2].x.toDouble(), sortedTargets[2].y.toDouble()),
                org.opencv.core.Point(sortedTargets[3].x.toDouble(), sortedTargets[3].y.toDouble())
            )
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(maskMat, warpedMask, transform, targetMat.size(), Imgproc.INTER_LINEAR)
            
            overlayBmp = Bitmap.createBitmap(result.width, result.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(warpedMask, overlayBmp)
            Canvas(result).drawBitmap(overlayBmp, 0f, 0f, Paint(Paint.FILTER_BITMAP_FLAG))

            srcPts.release(); dstPts.release(); transform.release()
            return result
        } finally {
            targetMat.release(); maskMat.release(); warpedMask.release()
            overlayBmp?.recycle()
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
        val capture = imageCapture ?: return
        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bitmap = image.toBitmapExt()
                image.close()
                // 메인 소스 비트맵 할당 (절대 백그라운드 연산 중 .recycle()을 호출하지 않음)
                synchronized(bitmapLock) { 
                    lastCapturedBitmap?.recycle()
                    lastCapturedBitmap = bitmap 
                }
                
                runOnUiThread { 
                    viewFinder?.visibility = View.INVISIBLE
                    nativeBackgroundView?.setImageBitmap(bitmap)
                    nativeBackgroundView?.visibility = View.VISIBLE
                    webView?.evaluateJavascript("window.onNativePhotoCaptured()", null)
                }
            }
            override fun onError(e: ImageCaptureException) { Log.e("JiSeKa", "Capture failed", e) }
        })
    }

    private fun ImageProxy.toBitmapExt(): Bitmap {
        val yBuffer = planes[0].buffer; val uBuffer = planes[1].buffer; val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining(); val uSize = uBuffer.remaining(); val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize); vBuffer.get(nv21, ySize, vSize); uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
        val matrix = Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
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

    /**
     * 🚨 [TOP 2 & 3 해결] 앱 종료 시 캐시 파일 삭제 및 모든 잔여 비트맵 메모리 완벽 환산
     */
    override fun onDestroy() { 
        super.onDestroy()
        cameraExecutor.shutdown()
        analysisExecutor.shutdown()
        recognizer.close()
        
        // 로컬 프리뷰 파일 정리
        try { cachedPreviewFile?.let { if (it.exists()) it.delete() } } catch (e: Exception) {}

        synchronized(bitmapLock) {
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
