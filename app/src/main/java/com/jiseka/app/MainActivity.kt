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
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private var webView: WebView? = null
    private var viewFinder: PreviewView? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    private var lastCapturedBitmap: Bitmap? = null
    private val bitmapLock = Any()

    // ML Kit 한국어 텍스트 인식 클라이언트
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

                    // 1. 🚨 개선 5 & 7 & 8 & 9: 끊어진 엣지 연결(Morphology) 적용 후 최대 5개의 상위 사각형 추출
                    val candidates = extractPlateCandidatesOptimized(sourceBitmap, initialPoints)

                    if (candidates.isEmpty()) {
                        Log.w("JiSeKa Engine", "⚠️ 사각형 후보 없음: 원본 보존 프로세스 진행")
                        sendPointsToJs(null, bmpW, bmpH)
                        sourceBitmap.recycle() // 🚨 개선 1: 즉시 종료 시점에만 안전 회수
                        return@execute
                    }

                    // 2. 🚨 개선 1 & 2: 비동기 체인이 비트맵 참조를 유지하도록 제어 플로우 이관
                    verifyCandidatesSequentiallySafe(sourceBitmap, candidates, 0, bmpW, bmpH)

                } catch (e: Exception) {
                    Log.e("JiSeKa Engine", "분석 엔진 처리 에러", e)
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSuccess('{\"corners\":null}')", null) }
                    sourceBitmap.recycle()
                }
                // 🚨 주의: finally에서 무조건 회수하던 코드를 제거하여 Async 사용 중 회수되는 버그 원천 차단
            }
        }

        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            val sourceBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } ?: return
            analysisExecutor.execute {
                try {
                    val jsonObj = JSONObject(cornersJsonStr)
                    // 페이로드 정합성 유지 확인
                    if (jsonObj.isNull("corners")) {
                        Log.d("JiSeKa Engine", "ℹ️ 가림막 기각 대상: 원본 이미지 보존 저장")
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

                    val result = processPerspectiveOverlay(sourceBitmap, pts)
                    saveBitmapToGallery(result)
                    runOnUiThread { webView?.evaluateJavascript("window.onNativeSaveComplete()", null) }
                } finally { sourceBitmap.recycle() }
            }
        }
    }

    /**
     * 🚨 개선 1: 비동기 재귀 흐름에서 비트맵 수명주기 완벽 동기화
     */
    private fun verifyCandidatesSequentiallySafe(
        sourceBitmap: Bitmap, 
        candidates: List<List<PointF>>, 
        index: Int, 
        bmpW: Float, 
        bmpH: Float
    ) {
        if (index >= candidates.size) {
            Log.w("JiSeKa Engine", "⚠️ 모든 후보 사각형 검증 실패: 번호판 텍스트 미달로 기각")
            sendPointsToJs(null, bmpW, bmpH)
            sourceBitmap.recycle() // 🚨 안전 회수 지점
            return
        }

        val currentCandidate = candidates[index]
        val flatBmp = rectifyToFlatPlate(sourceBitmap, currentCandidate)

        if (flatBmp != null) {
            val inputImage = InputImage.fromBitmap(flatBmp, 0)
            recognizer.process(inputImage).addOnCompleteListener { task ->
                var isConfirmed = false
                if (task.isSuccessful) {
                    val rawText = task.result.text.replace(Regex("\\s+"), "")
                    // 🚨 개선 3: 정규식 기반 엄격한 패턴 필터링 (오탐율 대폭 감소)
                    if (isValidLicensePlatePattern(rawText)) {
                        Log.d("JiSeKa Engine", "✅ [후보 ${index + 1}/${candidates.size}] 정밀 검증 통과 ($rawText): 흡착 좌표 확정")
                        isConfirmed = true
                        sendPointsToJs(currentCandidate, bmpW, bmpH)
                        sourceBitmap.recycle() // 🚨 최종 성공 지점 회수
                    }
                }

                // 🚨 개선 6: ML Kit 엔진이 이미지 처리를 완료한 후 안전 회수 보장
                flatBmp.recycle()

                if (!isConfirmed) {
                    Log.d("JiSeKa Engine", "❌ [후보 ${index + 1}/${candidates.size}] 단순 패턴 기각. 다음 순위 스캔...")
                    verifyCandidatesSequentiallySafe(sourceBitmap, candidates, index + 1, bmpW, bmpH)
                }
            }
        } else {
            verifyCandidatesSequentiallySafe(sourceBitmap, candidates, index + 1, bmpW, bmpH)
        }
    }

    // 🚨 개선 3: 정규식을 통한 강력한 번호판 패턴 유효성 검사기
    private fun isValidLicensePlatePattern(text: String): Boolean {
        if (text.length < 3) return false
        // 표준 차량 번호판: 앞 2~3자리 숫자 + 한글 1자 + 뒤 4자리 숫자
        val strictPattern = Regex("\\d{2,3}[가-힣]\\d{4}")
        if (strictPattern.containsMatchIn(text)) return true

        // 임시 차량 또는 일부 훼손 환경을 커버하기 위한 최소 유효성: 숫자 3개 이상 및 한글 포함
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
     * 🚨 개선 5 & 7 & 8 & 9: 네이티브 누수 방어, 모폴로지 보완, RETR_LIST 및 상위 5개 후보 제한
     */
    private fun extractPlateCandidatesOptimized(bitmap: Bitmap, pts: List<PointF>): List<List<PointF>> {
        val mat = org.opencv.core.Mat()
        Utils.bitmapToMat(bitmap, mat)
        val gray = org.opencv.core.Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
        
        val xs = pts.map { it.x }; val ys = pts.map { it.y }
        val minX = max(0, xs.min().toInt())
        val minY = max(0, ys.min().toInt())
        val maxX = min(mat.cols() - 1, xs.max().toInt())
        val maxY = min(mat.rows() - 1, ys.max().toInt())
        
        val roi = org.opencv.core.Rect(minX, minY, maxX - minX, maxY - minY)
        if (roi.width <= 10 || roi.height <= 10) {
            mat.release(); gray.release()
            return emptyList()
        }
        
        val roiMat = org.opencv.core.Mat(gray, roi)
        Imgproc.GaussianBlur(roiMat, roiMat, org.opencv.core.Size(5.0, 5.0), 0.0)
        
        // 🚨 개선 9: 조명/훼손으로 끊어진 번호판 에지 복원을 위한 형태학적 닫기(Close) 연산 도입
        val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(5.0, 5.0))
        Imgproc.morphologyEx(roiMat, roiMat, Imgproc.MORPH_CLOSE, morphKernel)
        morphKernel.release() // JNI 누수 방어

        Imgproc.Canny(roiMat, roiMat, 50.0, 150.0)
        
        val contours = mutableListOf<MatOfPoint>()
        // 🚨 개선 8: 외부 윤곽선 우선에 의한 번호판 프레임 탈락 방지를 위해 RETR_LIST 탐색
        Imgproc.findContours(roiMat, contours, org.opencv.core.Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        
        val validCandidates = mutableListOf<Pair<Double, List<PointF>>>()
        val totalRoiArea = roi.width * roi.height.toDouble()

        for (contour in contours) {
            // 🚨 개선 5: 명시적 JNI 네이티브 누수 관리
            val contour2f = MatOfPoint2f(*contour.toArray())
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx, Imgproc.arcLength(contour2f, true) * 0.02, true)
            
            if (approx.rows() == 4) {
                val area = Imgproc.contourArea(approx)
                if (area > totalRoiArea * 0.1) {
                    val rawPoints = approx.toArray().map { PointF(it.x.toFloat() + roi.x, it.y.toFloat() + roi.y) }
                    validCandidates.add(Pair(area, sortCorners(rawPoints)))
                }
            }
            contour2f.release()
            approx.release() // 🚨 안전 해제
            contour.release()
        }

        mat.release(); gray.release()
        
        // 🚨 개선 7: 연산/배터리 폭주 방지를 위해 상위 5개의 핵심 다각형만 최종 채택
        return validCandidates.sortedByDescending { it.first }.map { it.second }.take(5)
    }

    private fun rectifyToFlatPlate(sourceBitmap: Bitmap, pts: List<PointF>): Bitmap? {
        try {
            val srcMat = org.opencv.core.Mat()
            Utils.bitmapToMat(sourceBitmap, srcMat)

            val targetW = 400
            val targetH = 100

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
        } catch (e: Exception) {
            Log.e("JiSeKa Engine", "평탄화 뷰 생성 실패", e)
            return null
        }
    }

    private fun sortCorners(pts: List<PointF>): List<PointF> {
        val sortedByY = pts.sortedBy { it.y }
        val top = sortedByY.take(2).sortedBy { it.x }
        val bottom = sortedByY.takeLast(2).sortedByDescending { it.x }
        return listOf(top[0], top[1], bottom[0], bottom[1])
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
        
        val sortedTargets = sortCorners(targetCorners)
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
        
        targetMat.release(); maskMat.release(); srcPts.release(); dstPts.release()
        transform.release(); warpedMask.release(); overlayBmp.recycle()
        
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
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
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
