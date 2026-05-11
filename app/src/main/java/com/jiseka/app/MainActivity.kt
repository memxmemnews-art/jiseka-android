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
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
import android.util.Size
import android.view.View
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
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
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private var viewFinder: PreviewView? = null
    private var webView: WebView? = null
    
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService
    
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null 
    
    private var isWebReady = false 
    private var currentPerspectiveMode = "FRONT"

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        try {
            setContentView(R.layout.activity_main)
        } catch (e: Exception) {
            showToast("🚨 [에러] activity_main.xml 구조에 문제가 있습니다.")
            return
        }

        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            Log.e("JiSeKa", "OpenCV 초기화 실패")
            showToast("🚨 OpenCV 로드 실패")
        }

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

        if (viewFinder == null || webView == null) {
            showToast("🚨 XML 로드 실패")
            return
        }

        viewFinder?.scaleType = PreviewView.ScaleType.FIT_CENTER
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()

        webView?.apply {
            if (BuildConfig.DEBUG) {
                clearHistory()
                clearFormData()
                CookieManager.getInstance().removeAllCookies(null)
                clearCache(true)
            }
            
            setLayerType(View.LAYER_TYPE_SOFTWARE, null)
            setBackgroundColor(Color.TRANSPARENT)
            
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                cacheMode = WebSettings.LOAD_NO_CACHE 
                allowFileAccess = true
            }

            webChromeClient = WebChromeClient()
            addJavascriptInterface(WebAppInterface(), "AndroidBridge")
            
            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    return false
                }
                override fun onPageFinished(view: WebView?, url: String?) {
                    isWebReady = true
                }
            }
            
            loadUrl("https://ziseka-app.vercel.app")
        }

        if (allPermissionsGranted()) startCamera() 
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)
    }

    override fun onDestroy() {
        super.onDestroy()
        
        webView?.apply {
            stopLoading()
            webChromeClient = null
            webViewClient = WebViewClient()
            destroy()
        }

        recognizer.close()
        cameraExecutor.shutdown()
        analysisExecutor.shutdown()

        synchronized(bitmapLock) {
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null
        }
    }

    private fun showToast(msg: String) {
        runOnUiThread {
            if (!isFinishing && !isDestroyed) {
                Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun safeEvaluateJavascript(script: String) {
        if (isFinishing || isDestroyed) return
        runOnUiThread {
            if (isFinishing || isDestroyed || !isWebReady) return@runOnUiThread
            try {
                webView?.evaluateJavascript(script, null)
            } catch (e: Exception) {
                Log.e("JiSeKa", "JS evaluate 실패", e)
            }
        }
    }

    private fun sendErrorToWeb(msg: String) {
        val js = "javascript:window.onNativeError(${JSONObject.quote(msg)})"
        safeEvaluateJavascript(js)
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun takePhotoOnly() {
            if (!isWebReady) return
            runOnUiThread { capturePhoto() }
        }

        @JavascriptInterface
        fun analyzePlate(left: Float, top: Float, right: Float, bottom: Float) {
            analyzePlateWithMode(left, top, right, bottom, "FRONT")
        }

        @JavascriptInterface
        fun analyzePlateWithMode(left: Float, top: Float, right: Float, bottom: Float, mode: String) {
            currentPerspectiveMode = mode.uppercase()
            analysisExecutor.execute { 
                val safeBitmap = synchronized(bitmapLock) {
                    lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true)
                }
                
                safeBitmap?.let { bmp ->
                    try {
                        val cLeft = kotlin.math.max(0f, kotlin.math.min(1f, left))
                        val cTop = kotlin.math.max(0f, kotlin.math.min(1f, top))
                        val cRight = kotlin.math.max(0f, kotlin.math.min(1f, right))
                        val cBottom = kotlin.math.max(0f, kotlin.math.min(1f, bottom))
                        
                        processPlateAnalysis(bmp, RectF(cLeft, cTop, cRight, cBottom))
                    } catch (e: Exception) {
                        Log.e("JiSeKa", "분석 에러", e)
                        sendErrorToWeb("분석 중 네이티브 에러 발생")
                    } finally {
                        if (!bmp.isRecycled) bmp.recycle()
                    }
                } ?: sendErrorToWeb("분석할 사진이 메모리에 없습니다.")
            }
        }

        @JavascriptInterface
        fun showToast(msg: String) {
            this@MainActivity.showToast(msg)
        }

        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersStr: String) {
            analysisExecutor.execute {
                var bgBmp: Bitmap? = null
                var maskBmp: Bitmap? = null
                var resultBmp: Bitmap? = null
                
                var bgMat: org.opencv.core.Mat? = null
                var maskMat: org.opencv.core.Mat? = null
                var warpedMask: org.opencv.core.Mat? = null
                var grayRoi: org.opencv.core.Mat? = null
                var bgRoi: org.opencv.core.Mat? = null
                var maskGray: org.opencv.core.Mat? = null
                var perspectiveTransform: org.opencv.core.Mat? = null
                
                try {
                    bgBmp = synchronized(bitmapLock) {
                        lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true)
                    } ?: throw IllegalStateException("메모리에 원본 비트맵이 없습니다.")
                    
                    val resId = resources.getIdentifier("cbf082_grande", "drawable", packageName)
                    if (resId == 0) throw IllegalStateException("cbf082_grande 텍스처 리소스 누락")
                    
                    maskBmp = BitmapFactory.decodeResource(resources, resId)
                        ?: throw IllegalStateException("마스크 디코딩 실패")

                    val cornersArray = JSONArray(cornersStr)
                    if (cornersArray.length() != 4) throw IllegalArgumentException("모서리 좌표 오류")

                    val imgW = bgBmp.width.toFloat()
                    val imgH = bgBmp.height.toFloat()

                    val dstPoints = ArrayList<org.opencv.core.Point>()
                    var minX = imgW; var maxX = 0f
                    var minY = imgH; var maxY = 0f

                    for (i in 0 until 4) {
                        val ptObj = cornersArray.getJSONObject(i)
                        val px = (ptObj.getDouble("x") * imgW).toFloat()
                        val py = (ptObj.getDouble("y") * imgH).toFloat()
                        dstPoints.add(org.opencv.core.Point(px.toDouble(), py.toDouble()))
                        if (px < minX) minX = px; if (px > maxX) maxX = px
                        if (py < minY) minY = py; if (py > maxY) maxY = py
                    }

                    bgMat = org.opencv.core.Mat()
                    maskMat = org.opencv.core.Mat()
                    org.opencv.android.Utils.bitmapToMat(bgBmp, bgMat)
                    org.opencv.android.Utils.bitmapToMat(maskBmp, maskMat)

                    if (bgMat.empty() || maskMat.empty()) throw IllegalStateException("OpenCV 비트맵 매트릭스 변환 실패")

                    val cols = bgMat.cols()
                    val rows = bgMat.rows()
                    val x1 = minX.toInt().coerceIn(0, cols - 1)
                    val y1 = minY.toInt().coerceIn(0, rows - 1)
                    val x2 = maxX.toInt().coerceIn(x1 + 1, cols)
                    val y2 = maxY.toInt().coerceIn(y1 + 1, rows)
                    
                    val safeRoiRect = org.opencv.core.Rect(x1, y1, x2 - x1, y2 - y1)
                    
                    bgRoi = org.opencv.core.Mat(bgMat, safeRoiRect)
                    grayRoi = org.opencv.core.Mat()
                    org.opencv.imgproc.Imgproc.cvtColor(bgRoi, grayRoi, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)
                    
                    val meanColor = org.opencv.core.Core.mean(grayRoi)
                    val targetLuminance = meanColor.`val`[0]

                    maskGray = org.opencv.core.Mat()
                    org.opencv.imgproc.Imgproc.cvtColor(maskMat, maskGray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)
                    val maskMean = org.opencv.core.Core.mean(maskGray).`val`[0]

                    if (maskMean > 10.0) {
                        val brightnessRatio = targetLuminance / maskMean
                        val clampedRatio = kotlin.math.max(0.6, kotlin.math.min(1.3, brightnessRatio))
                        maskMat.convertTo(maskMat, -1, clampedRatio, 0.0)
                    }

                    val srcMatPts = org.opencv.core.MatOfPoint2f(
                        org.opencv.core.Point(0.0, 0.0),
                        org.opencv.core.Point(maskMat.cols().toDouble(), 0.0),
                        org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()),
                        org.opencv.core.Point(0.0, maskMat.rows().toDouble())
                    )
                    val dstMatPts = org.opencv.core.MatOfPoint2f(*dstPoints.toTypedArray())

                    perspectiveTransform = org.opencv.imgproc.Imgproc.getPerspectiveTransform(srcMatPts, dstMatPts)
                    warpedMask = org.opencv.core.Mat()
                    
                    org.opencv.imgproc.Imgproc.warpPerspective(
                        maskMat, warpedMask, perspectiveTransform, 
                        bgMat.size(), org.opencv.imgproc.Imgproc.INTER_LINEAR
                    )

                    val warpedBmp = Bitmap.createBitmap(bgBmp.width, bgBmp.height, Bitmap.Config.ARGB_8888)
                    org.opencv.android.Utils.matToBitmap(warpedMask, warpedBmp)

                    resultBmp = Bitmap.createBitmap(bgBmp.width, bgBmp.height, Bitmap.Config.ARGB_8888)
                    val canvas = android.graphics.Canvas(resultBmp)
                    canvas.drawBitmap(bgBmp, 0f, 0f, null)
                    
                    val blendPaint = android.graphics.Paint().apply {
                        isAntiAlias = true
                        isFilterBitmap = true
                    }
                    canvas.drawBitmap(warpedBmp, 0f, 0f, blendPaint)
                    warpedBmp.recycle()

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
                        resultBmp!!.compress(Bitmap.CompressFormat.JPEG, 96, stream)
                        stream.flush()
                    }

                    legacyFile?.let { file ->
                        android.media.MediaScannerConnection.scanFile(this@MainActivity, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)
                    }

                    this@MainActivity.showToast("자연스러운 번호판 마스킹 합성 완료!")

                } catch (e: Exception) {
                    Log.e("JiSeKa", "네이티브 합성 캡처 실패", e)
                    this@MainActivity.showToast("합성 저장 실패: " + e.message)
                } finally {
                    bgMat?.release()
                    maskMat?.release()
                    warpedMask?.release()
                    grayRoi?.release()
                    bgRoi?.release()
                    maskGray?.release()
                    perspectiveTransform?.release()
                    
                    bgBmp?.recycle()
                    maskBmp?.recycle()
                    
                    runOnUiThread {
                        Handler(Looper.getMainLooper()).postDelayed({
                            resultBmp?.recycle()
                        }, 500)
                        safeEvaluateJavascript("javascript:window.onNativeSaveComplete()")
                    }
                }
            }
        }
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }
            
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(1920, 1080))
                .build()
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
                showToast("🚨 카메라 시작 실패")
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
                    val bitmap = imageProxy.toBitmapExt()
                    val rotation = imageProxy.imageInfo.rotationDegrees

                    var rotatedBmp = bitmap
                    if (rotation != 0) {
                        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                        rotatedBmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        if (rotatedBmp != bitmap) bitmap.recycle()
                    }

                    val downscaledBmp = downscaleBitmapIfNeeded(rotatedBmp, 2400)
                    if (downscaledBmp != rotatedBmp) rotatedBmp.recycle()
                    
                    synchronized(bitmapLock) {
                        lastCapturedBitmap?.recycle()
                        lastCapturedBitmap = downscaledBmp
                    }
                    
                    val baos = ByteArrayOutputStream()
                    downscaledBmp.compress(Bitmap.CompressFormat.JPEG, 75, baos)
                    val base64Img = "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

                    val safeJS = "javascript:window.onPhotoCaptured(${JSONObject.quote(base64Img)})"
                    safeEvaluateJavascript(safeJS)
                } catch (e: Exception) {
                    sendErrorToWeb("사진 처리 오류")
                } finally {
                    imageProxy.close()
                    isCapturing.set(false)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                isCapturing.set(false)
                sendErrorToWeb("캡처 실패")
            }
        })
    }

    private fun processPlateAnalysis(bitmap: Bitmap, guideRectF: RectF) {
        val imgW = bitmap.width.toFloat()
        val imgH = bitmap.height.toFloat()
        
        val baseLeft = (guideRectF.left * imgW).toInt()
        val baseTop = (guideRectF.top * imgH).toInt()
        val baseRight = (guideRectF.right * imgW).toInt()
        val baseBottom = (guideRectF.bottom * imgH).toInt()

        val baseRectImg = Rect(baseLeft, baseTop, baseRight, baseBottom)
        val paddingX = (baseRectImg.width() * 0.08f).toInt()
        val paddingY = (baseRectImg.height() * 0.08f).toInt()

        val x1 = kotlin.math.max(0, baseLeft - paddingX)
        val y1 = kotlin.math.max(0, baseTop - paddingY)
        val x2 = kotlin.math.min(bitmap.width, baseRight + paddingX)
        val y2 = kotlin.math.min(bitmap.height, baseBottom + paddingY)

        val expandedRectImg = Rect(x1, y1, x2, y2)

        if (expandedRectImg.width() <= 0 || expandedRectImg.height() <= 0) {
            sendErrorToWeb("가이드 영역 오류")
            return
        }

        val corners = extractGeometryCorners(bitmap, expandedRectImg)
        if (corners == null) {
            sendErrorToWeb("번호판 구조 복원 실패: 가이드 박스를 더 정확히 맞춰주세요.")
            return
        }

        val rectifiedBmp = rectifyToFlatPlate(bitmap, corners)
        if (rectifiedBmp == null) {
            sendSuccessToWeb(corners, imgW, imgH)
            return
        }

        val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
        recognizer.process(inputImage).addOnCompleteListener { task ->
            if (isDestroyed || isFinishing) {
                rectifiedBmp.recycle()
                return@addOnCompleteListener
            }
            val text = if (task.isSuccessful) task.result.text.replace(Regex("\\s+"), "") else ""
            
            if (text.length >= 3) {
                Log.d("JiSeKa", "OCR Text Validator Confirmed: $text")
            } else {
                Log.w("JiSeKa", "OCR Text is short or empty, relying on geometry stability: $text")
            }
            
            sendSuccessToWeb(corners, imgW, imgH)
            rectifiedBmp.recycle()
        }
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
        safeEvaluateJavascript("javascript:window.onNativeSuccess(${JSONObject.quote(payload.toString())})")
    }

    private fun downscaleBitmapIfNeeded(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val maxDim = kotlin.math.max(bitmap.width, bitmap.height)
        if (maxDim <= maxDimension) return bitmap
        val scale = maxDimension.toFloat() / maxDim
        return Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
    }

    private fun extractGeometryCorners(bitmap: Bitmap, searchRect: Rect): Array<PointF>? {
        var mat: org.opencv.core.Mat? = null
        var roiMat: org.opencv.core.Mat? = null
        var gray: org.opencv.core.Mat? = null
        var sobel: org.opencv.core.Mat? = null
        var morph: org.opencv.core.Mat? = null
        var edges: org.opencv.core.Mat? = null
        var subGray: org.opencv.core.Mat? = null
        var lines: org.opencv.core.Mat? = null
        var clahe: org.opencv.imgproc.CLAHE? = null
        var textContours: ArrayList<org.opencv.core.MatOfPoint>? = null

        try {
            mat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(bitmap, mat)

            if (mat.empty()) return null

            val roiX = kotlin.math.max(0, searchRect.left)
            val roiY = kotlin.math.max(0, searchRect.top)
            val roiW = kotlin.math.min(mat.cols() - roiX, searchRect.width())
            val roiH = kotlin.math.min(mat.rows() - roiY, searchRect.height())

            if (roiW <= 0 || roiH <= 0) return null

            roiMat = org.opencv.core.Mat(mat, org.opencv.core.Rect(roiX, roiY, roiW, roiH))
            gray = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.cvtColor(roiMat, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)

            if (gray.cols() < 12 || gray.rows() < 12) return null

            clahe = org.opencv.imgproc.Imgproc.createCLAHE(2.0, org.opencv.core.Size(8.0, 8.0))
            clahe.apply(gray, gray)

            org.opencv.imgproc.Imgproc.GaussianBlur(gray, gray, org.opencv.core.Size(3.0, 3.0), 0.0)

            sobel = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.Sobel(gray, sobel, org.opencv.core.CvType.CV_8U, 1, 0, 3, 1.0, 0.0, org.opencv.core.Core.BORDER_DEFAULT)
            org.opencv.imgproc.Imgproc.threshold(sobel, sobel, 0.0, 255.0, org.opencv.imgproc.Imgproc.THRESH_BINARY or org.opencv.imgproc.Imgproc.THRESH_OTSU)

            morph = org.opencv.core.Mat()
            val kernel = org.opencv.imgproc.Imgproc.getStructuringElement(org.opencv.imgproc.Imgproc.MORPH_RECT, org.opencv.core.Size(25.0, 5.0))
            org.opencv.imgproc.Imgproc.morphologyEx(sobel, morph, org.opencv.imgproc.Imgproc.MORPH_CLOSE, kernel)
            kernel.release()

            textContours = ArrayList()
            val hierarchy = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.findContours(morph, textContours, hierarchy, org.opencv.imgproc.Imgproc.RETR_EXTERNAL, org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE)
            hierarchy.release()

            var plateRegionRect: org.opencv.core.Rect? = null
            var maxTextScore = -9999.0
            val totalRoiArea = roiW * roiH

            for (c in textContours) {
                val r = org.opencv.imgproc.Imgproc.boundingRect(c)
                val area = r.width * r.height
                
                if (area < totalRoiArea * 0.05 || area > totalRoiArea * 0.60) continue

                val aspect = r.width.toFloat() / r.height.toFloat().coerceAtLeast(1.0f)
                if (aspect in 1.5f..6.0f) {
                    var modeWeight = 1.0f
                    if (currentPerspectiveMode == "PASSENGER" && r.x > roiW * 0.3f) modeWeight = 1.2f 
                    else if (currentPerspectiveMode == "DRIVER" && r.x < roiW * 0.7f) modeWeight = 1.2f 

                    val score = area * aspect * modeWeight
                    if (score > maxTextScore) {
                        maxTextScore = score.toDouble()
                        plateRegionRect = r
                    }
                }
            }

            val safePlateRect = plateRegionRect ?: org.opencv.core.Rect(0, 0, roiW, roiH)
            val subRoiX = safePlateRect.x
            val subRoiY = safePlateRect.y
            val subRoiW = safePlateRect.width
            val subRoiH = safePlateRect.height

            subGray = org.opencv.core.Mat(gray, org.opencv.core.Rect(subRoiX, subRoiY, subRoiW, subRoiH))
            edges = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.Canny(subGray, edges, 10.0, 50.0)

            lines = org.opencv.core.Mat()
            val minLineLen = kotlin.math.max(20.0, subRoiW * 0.12)
            org.opencv.imgproc.Imgproc.HoughLinesP(edges, lines, 1.0, kotlin.math.PI / 180.0, 15, minLineLen, 15.0)

            val horizontalLines = ArrayList<LineSeg>()
            val verticalLines = ArrayList<LineSeg>()
            val subCenterX = subRoiW / 2.0f

            for (i in 0 until lines.rows()) {
                val vec = lines.get(i, 0)
                val lx1 = vec[0].toFloat(); val ly1 = vec[1].toFloat()
                val lx2 = vec[2].toFloat(); val ly2 = vec[3].toFloat()
                
                val angle = kotlin.math.abs(kotlin.math.atan2((ly2 - ly1).toDouble(), (lx2 - lx1).toDouble()) * 180.0 / kotlin.math.PI)
                val seg = LineSeg(PointF(lx1, ly1), PointF(lx2, ly2))

                if (angle < 20.0 || angle > 160.0) {
                    horizontalLines.add(seg)
                } else {
                    var isValidVertical = false
                    if (currentPerspectiveMode == "PASSENGER" && angle in 100.0..160.0) isValidVertical = true 
                    else if (currentPerspectiveMode == "DRIVER" && angle in 20.0..80.0) isValidVertical = true 
                    else if (currentPerspectiveMode == "FRONT" && angle in 60.0..120.0) isValidVertical = true 
                    else if (currentPerspectiveMode == "FRONT" && angle in 20.0..160.0) isValidVertical = true 

                    if (isValidVertical) {
                        val midX = (lx1 + lx2) / 2.0f
                        if (kotlin.math.abs(midX - subCenterX) > subRoiW * 0.15f) {
                            verticalLines.add(seg)
                        }
                    }
                }
            }

            if (horizontalLines.size < 2 || verticalLines.size < 2) {
                return fallbackPolyReconstruction(edges, roiX + subRoiX, roiY + subRoiY, subRoiW, subRoiH, gray)
            }

            val topEdge = clusterAndGetBestLine(horizontalLines, true, true) ?: return null
            val bottomEdge = clusterAndGetBestLine(horizontalLines, true, false) ?: return null
            val leftEdge = clusterAndGetBestLine(verticalLines, false, true) ?: return null
            val rightEdge = clusterAndGetBestLine(verticalLines, false, false) ?: return null

            val tl = intersection(topEdge, leftEdge) ?: return null
            val tr = intersection(topEdge, rightEdge) ?: return null
            val br = intersection(bottomEdge, rightEdge) ?: return null
            val bl = intersection(bottomEdge, leftEdge) ?: return null

            val baseOffset = PointF((roiX + subRoiX).toFloat(), (roiY + subRoiY).toFloat())
            
            val maxPx = gray.cols() - 6f
            val maxPy = gray.rows() - 6f

            val clampTL = org.opencv.core.Point((tl.x + baseOffset.x).toDouble().coerceIn(5.0, maxPx.toDouble()), (tl.y + baseOffset.y).toDouble().coerceIn(5.0, maxPy.toDouble()))
            val clampTR = org.opencv.core.Point((tr.x + baseOffset.x).toDouble().coerceIn(5.0, maxPx.toDouble()), (tr.y + baseOffset.y).toDouble().coerceIn(5.0, maxPy.toDouble()))
            val clampBR = org.opencv.core.Point((br.x + baseOffset.x).toDouble().coerceIn(5.0, maxPx.toDouble()), (br.y + baseOffset.y).toDouble().coerceIn(5.0, maxPy.toDouble()))
            val clampBL = org.opencv.core.Point((bl.x + baseOffset.x).toDouble().coerceIn(5.0, maxPx.toDouble()), (bl.y + baseOffset.y).toDouble().coerceIn(5.0, maxPy.toDouble()))

            val orderedPts = org.opencv.core.MatOfPoint2f(clampTL, clampTR, clampBR, clampBL)

            val criteria = org.opencv.core.TermCriteria(org.opencv.core.TermCriteria.EPS + org.opencv.core.TermCriteria.MAX_ITER, 30, 0.1)
            org.opencv.imgproc.Imgproc.cornerSubPix(gray, orderedPts, org.opencv.core.Size(5.0, 5.0), org.opencv.core.Size(-1.0, -1.0), criteria)

            val refinedArray = orderedPts.toArray()
            val finalResult = Array(4) { i -> PointF(refinedArray[i].x.toFloat(), refinedArray[i].y.toFloat()) }
            orderedPts.release()

            return finalResult

        } catch (e: Exception) {
            return null
        } finally {
            mat?.release()
            roiMat?.release()
            gray?.release()
            sobel?.release()
            morph?.release()
            edges?.release()
            subGray?.release()
            lines?.release()
            
            try { clahe?.release() } catch (err: Exception) { }
            
            // 🚨 최종 무결성 확보: 메인 탐색 루프의 에러 원인이었던 .release() 순회 완전 소멸
            textContours?.clear()
        }
    }

    data class LineSeg(val p1: PointF, val p2: PointF) {
        val length: Float get() = kotlin.math.hypot(p2.x - p1.x, p2.y - p1.y)
        val midY: Float get() = (p1.y + p2.y) / 2.0f
        val midX: Float get() = (p1.x + p2.x) / 2.0f
    }

    private fun clusterAndGetBestLine(lines: List<LineSeg>, isHorizontal: Boolean, isFirstCluster: Boolean): LineSeg? {
        if (lines.isEmpty()) return null
        if (lines.size == 1) return lines.first()

        val sorted = if (isHorizontal) lines.sortedBy { it.midY } else lines.sortedBy { it.midX }
        val midIndex = sorted.size / 2
        val targetCluster = if (isFirstCluster) sorted.take(midIndex) else sorted.takeLast(sorted.size - midIndex)
        
        if (targetCluster.isEmpty()) return sorted.first()

        var sumX1 = 0.0f; var sumY1 = 0.0f
        var sumX2 = 0.0f; var sumY2 = 0.0f
        var totalWeight = 0.0f

        for (l in targetCluster) {
            val w = l.length
            val p1 = if (isHorizontal) (if (l.p1.x < l.p2.x) l.p1 else l.p2) else (if (l.p1.y < l.p2.y) l.p1 else l.p2)
            val p2 = if (isHorizontal) (if (l.p1.x < l.p2.x) l.p2 else l.p1) else (if (l.p1.y < l.p2.y) l.p2 else l.p1)

            sumX1 += p1.x * w; sumY1 += p1.y * w
            sumX2 += p2.x * w; sumY2 += p2.y * w
            totalWeight += w
        }

        if (totalWeight < 1e-4f) return targetCluster.maxByOrNull { it.length }

        return LineSeg(PointF(sumX1 / totalWeight, sumY1 / totalWeight), PointF(sumX2 / totalWeight, sumY2 / totalWeight))
    }

    private fun intersection(l1: LineSeg, l2: LineSeg): PointF? {
        val x1 = l1.p1.x; val y1 = l1.p1.y
        val x2 = l1.p2.x; val y2 = l1.p2.y
        val x3 = l2.p1.x; val y3 = l2.p1.y
        val x4 = l2.p2.x; val y4 = l2.p2.y

        val denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (kotlin.math.abs(denom) < 1e-6f) return null 

        val t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        val px = x1 + t * (x2 - x1)
        val py = y1 + t * (y2 - y1)
        return PointF(px, py)
    }

    private fun fallbackPolyReconstruction(edges: org.opencv.core.Mat, absX: Int, absY: Int, sw: Int, sh: Int, gray: org.opencv.core.Mat): Array<PointF>? {
        val contours = ArrayList<org.opencv.core.MatOfPoint>()
        val hierarchy = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.findContours(edges, contours, hierarchy, org.opencv.imgproc.Imgproc.RETR_LIST, org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()

        var bestScore = -9999.0
        var bestQuad: Array<PointF>? = null
        val totalArea = sw * sh

        for (c in contours) {
            val contour2f = org.opencv.core.MatOfPoint2f(*c.toArray())
            val approx = org.opencv.core.MatOfPoint2f()
            val arcLen = org.opencv.imgproc.Imgproc.arcLength(contour2f, true)
            org.opencv.imgproc.Imgproc.approxPolyDP(contour2f, approx, arcLen * 0.02, true)
            contour2f.release()

            if (approx.rows() == 4 && org.opencv.imgproc.Imgproc.isContourConvex(org.opencv.core.MatOfPoint(*approx.toArray()))) {
                val ptsMat = approx.toArray()
                val quadPts = Array(4) { i -> PointF(ptsMat[i].x.toFloat() + absX, ptsMat[i].y.toFloat() + absY) }
                
                val xs = quadPts.map { it.x }; val ys = quadPts.map { it.y }
                val rw = (xs.maxOrNull() ?: 0f) - (xs.minOrNull() ?: 0f)
                val rh = (ys.maxOrNull() ?: 0f) - (ys.minOrNull() ?: 0f)
                val area = rw * rh

                if (area >= totalArea * 0.05 && area <= totalArea * 0.90) {
                    val aspect = kotlin.math.max(rw, rh) / kotlin.math.min(rw, rh).coerceAtLeast(1.0f)
                    if (aspect in 1.2f..6.0f) {
                        val score = area.toDouble()
                        if (score > bestScore) {
                            bestScore = score
                            bestQuad = quadPts
                        }
                    }
                }
            }
            approx.release()
        }
        contours.clear()

        bestQuad?.let { pts ->
            val sortedByY = pts.sortedBy { it.y }
            val topTwo = sortedByY.take(2).sortedBy { it.x }
            val bottomTwo = sortedByY.takeLast(2).sortedBy { it.x }

            val maxPx = gray.cols() - 6f
            val maxPy = gray.rows() - 6f

            val clampTL = org.opencv.core.Point(topTwo[0].x.toDouble().coerceIn(5.0, maxPx.toDouble()), topTwo[0].y.toDouble().coerceIn(5.0, maxPy.toDouble()))
            val clampTR = org.opencv.core.Point(topTwo[1].x.toDouble().coerceIn(5.0, maxPx.toDouble()), topTwo[1].y.toDouble().coerceIn(5.0, maxPy.toDouble()))
            val clampBR = org.opencv.core.Point(bottomTwo[1].x.toDouble().coerceIn(5.0, maxPx.toDouble()), bottomTwo[1].y.toDouble().coerceIn(5.0, maxPy.toDouble()))
            val clampBL = org.opencv.core.Point(bottomTwo[0].x.toDouble().coerceIn(5.0, maxPx.toDouble()), bottomTwo[0].y.toDouble().coerceIn(5.0, maxPy.toDouble()))

            val orderedPts = org.opencv.core.MatOfPoint2f(clampTL, clampTR, clampBR, clampBL)

            val criteria = org.opencv.core.TermCriteria(org.opencv.core.TermCriteria.EPS + org.opencv.core.TermCriteria.MAX_ITER, 30, 0.1)
            org.opencv.imgproc.Imgproc.cornerSubPix(gray, orderedPts, org.opencv.core.Size(5.0, 5.0), org.opencv.core.Size(-1.0, -1.0), criteria)
            
            val refinedArray = orderedPts.toArray()
            val finalResult = Array(4) { i -> PointF(refinedArray[i].x.toFloat(), refinedArray[i].y.toFloat()) }
            orderedPts.release()
            return finalResult
        }
        return null
    }

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

            val targetW = kotlin.math.max(topW, bottomW).toInt().coerceAtLeast(200)
            val targetH = kotlin.math.max(leftH, rightH).toInt().coerceAtLeast(60)

            val srcPts = org.opencv.core.MatOfPoint2f(*corners.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val dstPts = org.opencv.core.MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0),
                org.opencv.core.Point(targetW.toDouble(), 0.0),
                org.opencv.core.Point(targetW.toDouble(), targetH.toDouble()),
                org.opencv.core.Point(0.0, targetH.toDouble())
            )

            val transform = org.opencv.imgproc.Imgproc.getPerspectiveTransform(srcPts, dstPts)
            dest = org.opencv.core.Mat()

            org.opencv.imgproc.Imgproc.warpPerspective(bgMat, dest, transform, org.opencv.core.Size(targetW.toDouble(), targetH.toDouble()))

            val result = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            org.opencv.android.Utils.matToBitmap(dest, result)

            srcPts.release()
            dstPts.release()
            transform.release()
            return result

        } catch (e: Exception) {
            return null
        } finally {
            bgMat?.release()
            dest?.release()
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun ImageProxy.toBitmapExt(): Bitmap {
        val options = BitmapFactory.Options().apply { 
            inSampleSize = 2
            inPreferredConfig = Bitmap.Config.ARGB_8888 
        }
        if (format == ImageFormat.JPEG) {
            val buffer = planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) ?: throw IllegalStateException("JPEG Bitmap decode failed")
        } 
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options) ?: throw IllegalStateException("YUV Bitmap decode failed")
    }
}
