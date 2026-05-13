package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.util.Size
import android.view.View
import android.webkit.*
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
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    private var lastCapturedBitmap: Bitmap? = null
    private val bitmapLock = Any()

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
        webView = findViewById(R.id.webView)

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
            runOnUiThread { viewFinder?.visibility = if (isVisible) View.VISIBLE else View.INVISIBLE }
        }

        @JavascriptInterface
        fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            analysisExecutor.execute {
                val sourceBitmap = synchronized(bitmapLock) { 
                    lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
                } ?: return@execute

                try {
                    val inputCorners = JSONObject(cornersJsonStr).getJSONArray("corners")
                    val bmpW = sourceBitmap.width.toFloat()
                    val bmpH = sourceBitmap.height.toFloat()
                    val initialPoints = mutableListOf<PointF>()
                    for (i in 0 until 4) {
                        val p = inputCorners.getJSONObject(i)
                        initialPoints.add(PointF(p.getDouble("x").toFloat() * bmpW, p.getDouble("y").toFloat() * bmpH))
                    }

                    // 🚨 고도화: 허프 변환 직선 피팅 엔진 기반 정밀 좌표 추출
                    val refinedPoints = extractPlateCornersViaLineFitting(sourceBitmap, initialPoints)

                    // OCR 유효성 검증을 위한 1회차 평탄화 후보 생성
                    verifyCandidateAndRespond(sourceBitmap, refinedPoints, initialPoints, bmpW, bmpH)

                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "분석 에러", e)
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSuccess('{\"corners\":null}')", null) }
                    sourceBitmap.recycle()
                }
            }
        }

        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            val sourceBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } ?: return
            analysisExecutor.execute {
                var overlayResult: Bitmap? = null
                try {
                    val jsonObj = JSONObject(cornersJsonStr)
                    if (jsonObj.isNull("corners")) {
                        saveBitmapToGallery(sourceBitmap)
                        runOnUiThread { webView?.evaluateJavascript("window.onNativeSaveComplete()", null) }
                        return@execute
                    }

                    val corners = jsonObj.getJSONArray("corners")
                    val bmpW = sourceBitmap.width.toFloat()
                    val bmpH = sourceBitmap.height.toFloat()
                    val pts = mutableListOf<PointF>()
                    for (i in 0 until 4) {
                        val p = corners.getJSONObject(i)
                        pts.add(PointF(p.getDouble("x").toFloat() * bmpW, p.getDouble("y").toFloat() * bmpH))
                    }

                    overlayResult = processPerspectiveOverlay(sourceBitmap, pts)
                    saveBitmapToGallery(overlayResult!!)
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSaveComplete()", null) }
                } finally { 
                    sourceBitmap.recycle()
                    overlayResult?.let { if (it != sourceBitmap) it.recycle() }
                }
            }
        }
    }

    private fun verifyCandidateAndRespond(
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
                if (task.isSuccessful) {
                    val rawText = task.result.text.replace(Regex("\\s+"), "")
                    if (isValidLicensePlatePattern(rawText)) {
                        Log.d("JiSeKa Engine", "✅ 정밀 피팅 좌표 OCR 검증 통과 ($rawText)")
                        sendPointsToJs(targetPoints, bmpW, bmpH)
                    } else {
                        Log.w("JiSeKa Engine", "⚠️ 피팅 영역 텍스트 미달. 사용자 가이드 좌표 폴백")
                        sendPointsToJs(fallbackPoints, bmpW, bmpH)
                    }
                } else {
                    sendPointsToJs(fallbackPoints, bmpW, bmpH)
                }
                flatBmp.recycle()
                sourceBitmap.recycle()
            }
        } else {
            sendPointsToJs(fallbackPoints, bmpW, bmpH)
            sourceBitmap.recycle()
        }
    }

    private fun isValidLicensePlatePattern(text: String): Boolean {
        if (text.length < 3) return false
        val strictPattern = Regex("\\d{2,3}[가-힣]\\d{4}")
        if (strictPattern.containsMatchIn(text)) return true
        val digitCount = text.count { it.isDigit() }
        val hasKorean = text.any { it in '가'..'힣' }
        return digitCount >= 3 && hasKorean
    }

    private fun sendPointsToJs(points: List<PointF>?, bmpW: Float, bmpH: Float) {
        val resultJson = if (points == null) {
            "{\"corners\":null}"
        } else {
            val outputArray = JSONArray()
            for (pt in points) {
                outputArray.put(JSONObject().put("x", (pt.x / bmpW).toDouble()).put("y", (pt.y / bmpH).toDouble()))
            }
            JSONObject().put("corners", outputArray).toString()
        }
        runOnUiThread { webView?.evaluateJavascript("window.onNativeSuccess('$resultJson')", null) }
    }

    /**
     * 🚨 최우선순위 개선 적용: 허프 변환(HoughLinesP) 기반의 독립적 외곽선 추출 및 기하학적 피팅 엔진
     * 끊어지거나 노이즈가 낀 다각형 대신, 상/하/좌/우 4개의 강력한 대표 선분을 도출하여 완벽한 수학적 교차점을 찾아냅니다.
     */
    private fun extractPlateCornersViaLineFitting(bitmap: Bitmap, pts: List<PointF>): List<PointF> {
        val mat = org.opencv.core.Mat()
        Utils.bitmapToMat(bitmap, mat)
        val gray = org.opencv.core.Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
        
        val xs = pts.map { it.x }; val ys = pts.map { it.y }
        val minX = max(0, xs.min().toInt()); val minY = max(0, ys.min().toInt())
        val maxX = min(mat.cols() - 1, xs.max().toInt()); val maxY = min(mat.rows() - 1, ys.max().toInt())
        
        val roi = org.opencv.core.Rect(minX, minY, maxX - minX, maxY - minY)
        if (roi.width <= 20 || roi.height <= 20) {
            mat.release(); gray.release()
            return pts
        }
        
        val roiMat = org.opencv.core.Mat(gray, roi)
        Imgproc.GaussianBlur(roiMat, roiMat, org.opencv.core.Size(5.0, 5.0), 0.0)
        
        // 모폴로지 연산으로 문자 엣지 간섭 최소화 및 프레임 강화
        val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(3.0, 3.0))
        Imgproc.morphologyEx(roiMat, roiMat, Imgproc.MORPH_CLOSE, morphKernel)
        morphKernel.release()
        
        Imgproc.Canny(roiMat, roiMat, 50.0, 150.0)

        // 1단계: HoughLinesP 가동 (선분 조각 수집)
        val lines = org.opencv.core.Mat()
        Imgproc.HoughLinesP(roiMat, lines, 1.0, Math.PI / 180, 30, roi.width * 0.3, 10.0)

        val topLines = mutableListOf<Line()>()
        val bottomLines = mutableListOf<Line()>()
        val leftLines = mutableListOf<Line()>()
        val rightLines = mutableListOf<Line()>()

        val roiCenterY = roi.height / 2.0
        val roiCenterX = roi.width / 2.0

        // 2단계: 추출된 선분들을 위치와 기울기 기반으로 상/하/좌/우 4개 엣지로 엄격히 분류
        for (i in 0 until lines.rows()) {
            val vec = lines.get(i, 0) ?: continue
            val x1 = vec[0]; val y1 = vec[1]; val x2 = vec[2]; val y2 = vec[3]
            
            val dx = x2 - x1; val dy = y2 - y1
            val angle = abs(Math.atan2(dy, dx) * 180.0 / Math.PI)
            val midX = (x1 + x2) / 2.0; val midY = (y1 + y2) / 2.0
            val length = sqrt(dx * dx + dy * dy)

            val lineObj = Line(x1, y1, x2, y2, length)

            // 수평선 계열 (각도가 0~35도 또는 145~180도)
            if (angle <= 35.0 || angle >= 145.0) {
                if (midY < roiCenterY) topLines.add(lineObj) else bottomLines.add(lineObj)
            } 
            // 수직선 계열 (각도가 55~125도)
            else if (angle in 55.0..125.0) {
                if (midX < roiCenterX) leftLines.add(lineObj) else rightLines.add(lineObj)
            }
        }
        lines.release()

        // 3단계: 각 그룹에서 가장 길고 뚜렷한 대표 직선(Best Line) 선택
        val topEdge = topLines.maxByOrNull { it.length }
        val bottomEdge = bottomLines.maxByOrNull { it.length }
        val leftEdge = leftLines.maxByOrNull { it.length }
        val rightEdge = rightLines.maxByOrNull { it.length }

        // 4개의 엣지가 모두 확보된 경우에만 교차점 피팅을 가동, 누락 시 안전하게 근사화/폴백
        var fittedPoints = pts
        if (topEdge != null && bottomEdge != null && leftEdge != null && rightEdge != null) {
            val tl = getIntersection(topEdge, leftEdge)
            val tr = getIntersection(topEdge, rightEdge)
            val br = getIntersection(bottomEdge, rightEdge)
            val bl = getIntersection(bottomEdge, leftEdge)

            if (tl != null && tr != null && br != null && bl != null) {
                // ROI 내부 상대 좌표를 전체 이미지 절대 좌표로 매핑
                val rawCorners = listOf(
                    PointF(tl.x.toFloat() + roi.x, tl.y.toFloat() + roi.y),
                    PointF(tr.x.toFloat() + roi.x, tr.y.toFloat() + roi.y),
                    PointF(br.x.toFloat() + roi.x, br.y.toFloat() + roi.y),
                    PointF(bl.x.toFloat() + roi.x, bl.y.toFloat() + roi.y)
                )

                // 🚨 우선순위 4: 업계 표준 방식의 완벽한 정렬 적용 (x±y Ordering)
                val sortedCorners = sortCornersStandard(rawCorners)

                // 🚨 우선순위 3: 서브픽셀 정밀 흡착 (cornerSubPix Refinement)
                // 수학적 교차점을 실제 엣지의 최적 그라디언트 중심으로 미세 소수점 보정 유도
                val subPixMat = MatOfPoint2f()
                subPixMat.fromArray(
                    org.opencv.core.Point(sortedCorners[0].x.toDouble(), sortedCorners[0].y.toDouble()),
                    org.opencv.core.Point(sortedCorners[1].x.toDouble(), sortedCorners[1].y.toDouble()),
                    org.opencv.core.Point(sortedCorners[2].x.toDouble(), sortedCorners[2].y.toDouble()),
                    org.opencv.core.Point(sortedCorners[3].x.toDouble(), sortedCorners[3].y.toDouble())
                )

                val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.05)
                Imgproc.cornerSubPix(
                    gray, 
                    subPixMat, 
                    org.opencv.core.Size(4.0, 4.0), 
                    org.opencv.core.Size(-1.0, -1.0), 
                    criteria
                )

                val subPixArray = subPixMat.toArray()
                fittedPoints = listOf(
                    PointF(subPixArray[0].x.toFloat(), subPixArray[0].y.toFloat()),
                    PointF(subPixArray[1].x.toFloat(), subPixArray[1].y.toFloat()),
                    PointF(subPixArray[2].x.toFloat(), subPixArray[2].y.toFloat()),
                    PointF(subPixArray[3].x.toFloat(), subPixArray[3].y.toFloat())
                )
                subPixMat.release()
            }
        }

        mat.release(); gray.release()
        return fittedPoints
    }

    // 내부 연산용 데이터 클래스
    data class Line(val x1: Double, val y1: Double, val x2: Double, val y2: Double, val length: Double)

    /**
     * 🚨 우선순위 2: 두 직선의 방정식 교차점 도출 알고리즘 (Line Intersection)
     * 크래머 공식(Cramer's Rule)을 사용하여 두 선분이 연장되었을 때 만나는 수학적 꼭짓점을 도출합니다.
     */
    private fun getIntersection(line1: Line, line2: Line): PointF? {
        val a1 = line1.y2 - line1.y1
        val b1 = line1.x1 - line1.x2
        val c1 = a1 * line1.x1 + b1 * line1.y1

        val a2 = line2.y2 - line2.y1
        val b2 = line2.x1 - line2.x2
        val c2 = a2 * line2.x1 + b2 * line2.y1

        val det = a1 * b2 - a2 * b1
        if (abs(det) < 1e-6) return null // 평행선 방어

        val cx = (b2 * c1 - b1 * c2) / det
        val cy = (a1 * c2 - a2 * c1) / det
        return PointF(cx.toFloat(), cy.toFloat())
    }

    /**
     * 🚨 우선순위 4: 업계 표준 방식인 (x+y), (x-y) 연산 기반 꼭짓점 정렬
     * 꼬임 현상을 완벽히 차단하고 시계방향(TL ➔ TR ➔ BR ➔ BL) 슬롯을 보장합니다.
     */
    private fun sortCornersStandard(pts: List<PointF>): List<PointF> {
        if (pts.size != 4) return pts
        val tl = pts.minByOrNull { it.x + it.y } ?: pts[0]
        val br = pts.maxByOrNull { it.x + it.y } ?: pts[1]
        val tr = pts.maxByOrNull { it.x - it.y } ?: pts[2]
        val bl = pts.minByOrNull { it.x - it.y } ?: pts[3]
        return listOf(tl, tr, br, bl)
    }

    private fun rectifyToFlatPlate(sourceBitmap: Bitmap, pts: List<PointF>): Bitmap? {
        try {
            val srcMat = org.opencv.core.Mat()
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
            val destMat = org.opencv.core.Mat()
            Imgproc.warpPerspective(srcMat, destMat, transform, org.opencv.core.Size(targetW.toDouble(), targetH.toDouble()))
            val flatBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(destMat, flatBitmap)
            srcMat.release(); destMat.release(); srcPts.release(); dstPts.release(); transform.release()
            return flatBitmap
        } catch (e: Exception) { return null }
    }

    private fun processPerspectiveOverlay(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val targetMat = org.opencv.core.Mat()
        Utils.bitmapToMat(result, targetMat)
        val resId = resources.getIdentifier("plate_mask", "drawable", packageName)
        val maskBmp = if (resId != 0) BitmapFactory.decodeResource(resources, resId) 
                      else Bitmap.createBitmap(600, 150, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.LTGRAY) }
        val maskMat = org.opencv.core.Mat()
        Utils.bitmapToMat(maskBmp, maskMat)
        val srcPts = MatOfPoint2f(org.opencv.core.Point(0.0, 0.0), org.opencv.core.Point(maskMat.cols().toDouble(), 0.0),
                                  org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()), org.opencv.core.Point(0.0, maskMat.rows().toDouble()))
        val sortedTargets = sortCornersStandard(targetCorners)
        val dstPts = MatOfPoint2f(org.opencv.core.Point(sortedTargets[0].x.toDouble(), sortedTargets[0].y.toDouble()),
                                  org.opencv.core.Point(sortedTargets[1].x.toDouble(), sortedTargets[1].y.toDouble()),
                                  org.opencv.core.Point(sortedTargets[2].x.toDouble(), sortedTargets[2].y.toDouble()),
                                  org.opencv.core.Point(sortedTargets[3].x.toDouble(), sortedTargets[3].y.toDouble()))
        val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val warpedMask = org.opencv.core.Mat()
        Imgproc.warpPerspective(maskMat, warpedMask, transform, targetMat.size(), Imgproc.INTER_LINEAR)
        val overlayBmp = Bitmap.createBitmap(result.width, result.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(warpedMask, overlayBmp)
        Canvas(result).drawBitmap(overlayBmp, 0f, 0f, Paint(Paint.FILTER_BITMAP_FLAG))
        targetMat.release(); maskMat.release(); srcPts.release(); dstPts.release(); transform.release(); warpedMask.release(); overlayBmp.recycle()
        return result
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
                synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = bitmap }
                val base64 = bitmapToBase64(bitmap)
                runOnUiThread { webView?.evaluateJavascript("window.onNativePhotoCaptured('$base64')", null) }
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

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out)
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
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
