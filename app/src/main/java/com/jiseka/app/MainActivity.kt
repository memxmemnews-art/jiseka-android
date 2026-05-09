package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.net.Uri
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.View
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
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
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var webView: WebView
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    private var lastCapturedBitmap: Bitmap? = null 
    private var isWebReady = false 

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!org.opencv.android.OpenCVLoader.initDebug()) Log.e("JiSeKa", "OpenCV 초기화 실패")

        viewFinder = findViewById(R.id.viewFinder)
        // 🚨 좌표계 통일을 위한 PreviewView ScaleType 강제 설정
        viewFinder.scaleType = PreviewView.ScaleType.FILL_CENTER

        webView = findViewById(R.id.webView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        webView.clearCache(true)
        webView.setLayerType(View.LAYER_TYPE_HARDWARE, null)
        webView.setBackgroundColor(Color.TRANSPARENT)
        
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            cacheMode = WebSettings.LOAD_NO_CACHE 
        }

        webView.webChromeClient = WebChromeClient()
        webView.addJavascriptInterface(WebAppInterface(), "AndroidBridge")
        
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val host = request?.url?.host ?: return true
                return host != Uri.parse(BuildConfig.VERCEL_URL).host
            }
            override fun onPageFinished(view: WebView?, url: String?) {
                isWebReady = true
            }
        }
        
        webView.loadUrl(BuildConfig.VERCEL_URL)

        if (allPermissionsGranted()) startCamera() 
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)
    }

    // 🚨 Activity Lifecycle 안전 접근 함수
    private fun safeEvaluateJavascript(script: String) {
        if (isFinishing || isDestroyed) return
        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread
            try {
                webView.evaluateJavascript(script, null)
            } catch (e: Exception) {
                Log.e("JiSeKa", "JS evaluate 실패", e)
            }
        }
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun takePhotoOnly() {
            if (!isWebReady) return
            runOnUiThread { capturePhoto() }
        }

        @JavascriptInterface
        fun analyzePlate(left: Float, top: Float, right: Float, bottom: Float) {
            cameraExecutor.execute { 
                // 🚨 Bitmap Recycle 충돌 방지를 위한 안전 복사본 사용
                lastCapturedBitmap?.let { bmp ->
                    val safeBitmap = bmp.copy(Bitmap.Config.ARGB_8888, false)
                    try {
                        processPlateAnalysis(safeBitmap, RectF(left, top, right, bottom))
                    } finally {
                        if (!safeBitmap.isRecycled) {
                            safeBitmap.recycle()
                        }
                    }
                } ?: sendErrorToWeb("분석할 사진이 메모리에 없습니다.")
            }
        }

        @JavascriptInterface
        fun showToast(msg: String) {
            runOnUiThread {
                if (!isFinishing && !isDestroyed) {
                    android.widget.Toast.makeText(this@MainActivity, msg, android.widget.Toast.LENGTH_SHORT).show()
                }
            }
        }

        @JavascriptInterface
        fun saveImageToGallery(base64Data: String) {
            // (임시 유지) 향후 Native Canvas 합성으로 전환 시 이곳을 수정합니다.
            Thread {
                var bitmap: Bitmap? = null
                try {
                    val base64Image = base64Data.substringAfter(",")
                    val imageBytes = Base64.decode(base64Image, Base64.DEFAULT)
                    bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size) ?: throw IllegalStateException("Bitmap decode failed")

                    val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
                    var outputStream: java.io.OutputStream? = null
                    var legacyFile: java.io.File? = null

                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                        val resolver = contentResolver
                        val contentValues = android.content.ContentValues().apply {
                            put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, filename)
                            put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                            put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_PICTURES + "/JiSeKa")
                        }
                        val imageUri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                        outputStream = imageUri?.let { resolver.openOutputStream(it) }
                    } else {
                        val picturesDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_PICTURES)
                        val saveDir = java.io.File(picturesDir, "JiSeKa")
                        if (!saveDir.exists()) saveDir.mkdirs()
                        legacyFile = java.io.File(saveDir, filename)
                        outputStream = java.io.FileOutputStream(legacyFile)
                    }

                    outputStream?.use { stream ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream)
                        stream.flush()
                    }

                    legacyFile?.let { file ->
                        android.media.MediaScannerConnection.scanFile(this@MainActivity, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)
                    }
                    showToast("사진이 갤러리에 저장되었습니다.")
                } catch (e: Exception) {
                    showToast("사진 저장 실패")
                } finally {
                    bitmap?.let { if (!it.isRecycled) it.recycle() }
                }
            }.start()
        }
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun capturePhoto() {
        val capture = imageCapture ?: return
        if (!isCapturing.compareAndSet(false, true)) return

        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            @SuppressLint("UnsafeOptInUsageError")
            override fun onCaptureSuccess(imageProxy: ImageProxy) {
                try {
                    // 🚨 백그라운드 스레드와 충돌 방지를 위해 recycle() 대신 GC에 위임
                    lastCapturedBitmap = null 
                    System.gc()

                    val bitmap = imageProxy.toBitmapExt()
                    val rotation = imageProxy.imageInfo.rotationDegrees

                    var rotatedBmp = bitmap
                    if (rotation != 0) {
                        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                        rotatedBmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        if (rotatedBmp != bitmap) bitmap.recycle()
                    }

                    lastCapturedBitmap = downscaleBitmapIfNeeded(rotatedBmp, 1024)
                    if (lastCapturedBitmap != rotatedBmp) rotatedBmp.recycle()
                    
                    val baos = ByteArrayOutputStream()
                    lastCapturedBitmap!!.compress(Bitmap.CompressFormat.JPEG, 70, baos)
                    val base64Img = "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

                    val safeJS = "javascript:window.onPhotoCaptured(${JSONObject.quote(base64Img)})"
                    safeEvaluateJavascript(safeJS)
                } catch (e: Exception) {
                    Log.e("JiSeKa", "Capture error", e)
                    sendErrorToWeb("사진 처리 중 내부 오류 발생")
                } finally {
                    imageProxy.close()
                    isCapturing.set(false)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                isCapturing.set(false)
                sendErrorToWeb("카메라 캡처 실패: ${exception.message}")
            }
        })
    }

    private fun processPlateAnalysis(bitmap: Bitmap, guideRectF: RectF) {
        val imgW = bitmap.width.toFloat()
        val imgH = bitmap.height.toFloat()
        
        val guideRectImg = Rect(
            kotlin.math.max(0, (guideRectF.left * imgW).toInt()),
            kotlin.math.max(0, (guideRectF.top * imgH).toInt()),
            kotlin.math.min(imgW.toInt(), (guideRectF.right * imgW).toInt()),
            kotlin.math.min(imgH.toInt(), (guideRectF.bottom * imgH).toInt())
        )

        if (guideRectImg.width() <= 0 || guideRectImg.height() <= 0) {
            sendErrorToWeb("가이드 영역이 올바르지 않습니다.")
            return
        }

        val corners = extractGeometryCorners(bitmap, guideRectImg)
        if (corners == null) {
            sendErrorToWeb("번호판을 찾을 수 없습니다.")
            return
        }

        val rectifiedBmp = rectifyToFlatPlate(bitmap, corners)
        if (rectifiedBmp == null) {
            sendErrorToWeb("이미지 보정에 실패했습니다.")
            return
        }

        val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
        recognizer.process(inputImage).addOnCompleteListener { task ->
            val text = task.result.text.replace(Regex("\\s+"), "")
            val plateRegex = Regex("\\d{2,3}[가-힣]\\d{4}") 
            
            val ocrValid = task.isSuccessful && plateRegex.containsMatchIn(text)
            if (!ocrValid) {
                sendErrorToWeb("번호판 텍스트를 인식할 수 없습니다.")
                rectifiedBmp.recycle()
                return@addOnCompleteListener
            }
            sendSuccessToWeb(corners, imgW, imgH)
            rectifiedBmp.recycle()
        }
    }

    private fun sendErrorToWeb(msg: String) {
        val js = "javascript:window.onNativeError(${JSONObject.quote(msg)})"
        safeEvaluateJavascript(js)
    }

    private fun sendSuccessToWeb(corners: Array<PointF>, imgW: Float, imgH: Float) {
        val payload = JSONObject().apply {
            put("version", 1)
            put("corners", JSONArray().apply {
                corners.forEach { 
                    put(JSONObject().apply { 
                        put("x", it.x / imgW) 
                        put("y", it.y / imgH) 
                    }) 
                }
            })
        }
        val js = "javascript:window.onNativeSuccess(${JSONObject.quote(payload.toString())})"
        safeEvaluateJavascript(js)
    }

    private fun downscaleBitmapIfNeeded(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val maxDim = kotlin.math.max(bitmap.width, bitmap.height)
        if (maxDim <= maxDimension) return bitmap
        val scale = maxDimension.toFloat() / maxDim
        return Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
    }

    private fun extractGeometryCorners(bitmap: Bitmap, guideRect: Rect): Array<PointF>? {
        var mat: org.opencv.core.Mat? = null; var roiMat: org.opencv.core.Mat? = null;
        var gray: org.opencv.core.Mat? = null; var edges: org.opencv.core.Mat? = null
        val contours = ArrayList<org.opencv.core.MatOfPoint>()
        try {
            mat = org.opencv.core.Mat(); org.opencv.android.Utils.bitmapToMat(bitmap, mat)
            val roiX = kotlin.math.max(0, guideRect.left)
            val roiY = kotlin.math.max(0, guideRect.top)
            val roiW = kotlin.math.min(guideRect.width(), (mat.cols() - roiX))
            val roiH = kotlin.math.min(guideRect.height(), (mat.rows() - roiY))
            
            if (roiW <= 0 || roiH <= 0) return null
            roiMat = org.opencv.core.Mat(mat, org.opencv.core.Rect(roiX, roiY, roiW, roiH))
            val roiArea = roiW * roiH 
            
            gray = org.opencv.core.Mat(); org.opencv.imgproc.Imgproc.cvtColor(roiMat, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)
            edges = org.opencv.core.Mat(); org.opencv.imgproc.Imgproc.Canny(gray, edges, 50.0, 150.0)
            org.opencv.imgproc.Imgproc.findContours(edges, contours, org.opencv.core.Mat(), org.opencv.imgproc.Imgproc.RETR_EXTERNAL, org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE)

            var bestScore = 0.0; var bestBox: Array<PointF>? = null
            for (contour in contours) {
                val contour2f = org.opencv.core.MatOfPoint2f(*contour.toArray())
                val minRect = org.opencv.imgproc.Imgproc.minAreaRect(contour2f)
                contour2f.release()

                val rw = minRect.size.width; val rh = minRect.size.height
                val rectArea = rw * rh
                if (rectArea < roiArea * 0.05) continue 
                
                val actualArea = org.opencv.imgproc.Imgproc.contourArea(contour)
                val rectangularity = if (rectArea > 0) actualArea / rectArea else 0.0
                val aspect = kotlin.math.max(rw, rh) / kotlin.math.min(rw, rh).coerceAtLeast(1.0)
                
                if (aspect in 2.0..6.0 && rectangularity > 0.6) {
                    val overlapRatio = (rectArea / roiArea.toDouble()).coerceIn(0.0, 1.0)
                    val score = (overlapRatio * 1.5) + (1.0 / aspect) + rectangularity
                    if (score > bestScore) {
                        bestScore = score
                        val pts = org.opencv.core.Mat(); org.opencv.imgproc.Imgproc.boxPoints(minRect, pts)
                        bestBox = Array(4) { i -> PointF(pts.get(i,0)[0].toFloat() + roiX, pts.get(i,1)[0].toFloat() + roiY) }
                        pts.release()
                    }
                }
            }
            
            bestBox?.let { pts ->
                val sorted = Array(4) { PointF(0f, 0f) }
                sorted[0] = pts.minByOrNull { it.x + it.y } ?: pts[0] 
                sorted[2] = pts.maxByOrNull { it.x + it.y } ?: pts[0] 
                sorted[1] = pts.maxByOrNull { it.x - it.y } ?: pts[0] 
                sorted[3] = pts.minByOrNull { it.x - it.y } ?: pts[0] 
                return sorted
            }
            return null
        } finally {
            mat?.release(); roiMat?.release(); gray?.release(); edges?.release()
            for (c in contours) c.release()
        }
    }

    // 🚨 OCR 비율 동적 계산 적용 (고정 400x100 해제)
    private fun rectifyToFlatPlate(bitmap: Bitmap, corners: Array<PointF>): Bitmap? {
        var bgMat: org.opencv.core.Mat? = null
        var dest: org.opencv.core.Mat? = null

        try {
            bgMat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(bitmap, bgMat)

            fun dist(a: PointF, b: PointF): Double {
                return kotlin.math.hypot((a.x - b.x).toDouble(), (a.y - b.y).toDouble())
            }

            val topW = dist(corners[0], corners[1])
            val bottomW = dist(corners[3], corners[2])
            val leftH = dist(corners[0], corners[3])
            val rightH = dist(corners[1], corners[2])

            val targetW = kotlin.math.max(topW,
