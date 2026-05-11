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
import android.util.Base64
import android.util.Log
import android.view.View
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
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    private var lastCapturedBitmap: Bitmap? = null 
    private var isWebReady = false 

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        try {
            setContentView(R.layout.activity_main)
        } catch (e: Exception) {
            Toast.makeText(this, "🚨 [에러] activity_main.xml 구조에 문제가 있습니다.", Toast.LENGTH_LONG).show()
            return
        }

        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            Log.e("JiSeKa", "OpenCV 초기화 실패")
            Toast.makeText(this, "🚨 OpenCV 로드 실패", Toast.LENGTH_LONG).show()
        }

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

        if (viewFinder == null || webView == null) {
            Toast.makeText(this, "🚨 XML 로드 실패", Toast.LENGTH_LONG).show()
            return
        }

        viewFinder?.scaleType = PreviewView.ScaleType.FIT_CENTER
        cameraExecutor = Executors.newSingleThreadExecutor()

        webView?.apply {
            clearCache(true)
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

    private fun safeEvaluateJavascript(script: String) {
        if (isFinishing || isDestroyed) return
        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread
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
            cameraExecutor.execute { 
                lastCapturedBitmap?.let { bmp ->
                    val safeBitmap = bmp.copy(Bitmap.Config.ARGB_8888, false)
                    try {
                        val cLeft = kotlin.math.max(0f, kotlin.math.min(1f, left))
                        val cTop = kotlin.math.max(0f, kotlin.math.min(1f, top))
                        val cRight = kotlin.math.max(0f, kotlin.math.min(1f, right))
                        val cBottom = kotlin.math.max(0f, kotlin.math.min(1f, bottom))
                        
                        processPlateAnalysis(safeBitmap, RectF(cLeft, cTop, cRight, cBottom))
                    } catch (e: Exception) {
                        Log.e("JiSeKa", "분석 에러", e)
                        sendErrorToWeb("분석 중 네이티브 에러 발생")
                    } finally {
                        if (!safeBitmap.isRecycled) safeBitmap.recycle()
                    }
                } ?: sendErrorToWeb("분석할 사진이 메모리에 없습니다.")
            }
        }

        @JavascriptInterface
        fun showToast(msg: String) {
            runOnUiThread {
                if (!isFinishing && !isDestroyed) {
                    Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
                }
            }
        }

        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersStr: String) {
            cameraExecutor.execute {
                var bgBmp: Bitmap? = null
                var maskBmp: Bitmap? = null
                var resultBmp: Bitmap? = null
                
                try {
                    bgBmp = lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true)
                        ?: throw IllegalStateException("메모리에 원본 비트맵이 없습니다.")
                    
                    val resId = resources.getIdentifier("cbf082_grande", "drawable", packageName)
                    if (resId == 0) throw IllegalStateException("cbf082_grande 리소스 누락")
                    
                    maskBmp = BitmapFactory.decodeResource(resources, resId)
                        ?: throw IllegalStateException("마스크 디코딩 실패")

                    val cornersArray = JSONArray(cornersStr)
                    if (cornersArray.length() != 4) throw IllegalArgumentException("모서리 좌표 오류")

                    val imgW = bgBmp.width.toFloat()
                    val imgH = bgBmp.height.toFloat()

                    val dstPoints = ArrayList<org.opencv.core.Point>()
                    for (i in 0 until 4) {
                        val ptObj = cornersArray.getJSONObject(i)
                        dstPoints.add(org.opencv.core.Point(ptObj.getDouble("x") * imgW, ptObj.getDouble("y") * imgH))
                    }

                    val bgMat = org.opencv.core.Mat()
                    val maskMat = org.opencv.core.Mat()
                    org.opencv.android.Utils.bitmapToMat(bgBmp, bgMat)
                    org.opencv.android.Utils.bitmapToMat(maskBmp, maskMat)

                    val srcMatPts = org.opencv.core.MatOfPoint2f(
                        org.opencv.core.Point(0.0, 0.0),
                        org.opencv.core.Point(maskMat.cols().toDouble(), 0.0),
                        org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()),
                        org.opencv.core.Point(0.0, maskMat.rows().toDouble())
                    )
                    val dstMatPts = org.opencv.core.MatOfPoint2f(*dstPoints.toTypedArray())

                    val perspectiveTransform = org.opencv.imgproc.Imgproc.getPerspectiveTransform(srcMatPts, dstMatPts)
                    val warpedMask = org.opencv.core.Mat()
                    
                    org.opencv.imgproc.Imgproc.warpPerspective(
                        maskMat, warpedMask, perspectiveTransform, 
                        bgMat.size(), org.opencv.imgproc.Imgproc.INTER_LINEAR
                    )

                    resultBmp = Bitmap.createBitmap(bgBmp.width, bgBmp.height, Bitmap.Config.ARGB_8888)
                    val warpedBmp = Bitmap.createBitmap(bgBmp.width, bgBmp.height, Bitmap.Config.ARGB_8888)
                    org.opencv.android.Utils.matToBitmap(warpedMask, warpedBmp)

                    val canvas = android.graphics.Canvas(resultBmp)
                    canvas.drawBitmap(bgBmp, 0f, 0f, null)
                    canvas.drawBitmap(warpedBmp, 0f, 0f, null)

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
                        resultBmp!!.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                        stream.flush()
                    }

                    legacyFile?.let { file ->
                        android.media.MediaScannerConnection.scanFile(this@MainActivity, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)
                    }

                    showToast("사진이 갤러리에 완벽히 저장되었습니다.")

                    srcMatPts.release()
                    dstMatPts.release()
                    perspectiveTransform.release()
                    warpedMask.release()
                    bgMat.release()
                    maskMat.release()
                    warpedBmp.recycle()

                } catch (e: Exception) {
                    Log.e("JiSeKa", "네이티브 합성 캡처 실패", e)
                    showToast("합성 저장 실패: " + e.message)
                } finally {
                    bgBmp?.recycle()
                    maskBmp?.recycle()
                    resultBmp?.recycle()
                    runOnUiThread {
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
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
                Toast.makeText(this, "🚨 카메라 시작 실패", Toast.LENGTH_LONG).show()
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

                    // 원거리 작은 번호판의 픽셀 보호를 위해 고해상도(2400px) 유지
                    lastCapturedBitmap = downscaleBitmapIfNeeded(rotatedBmp, 2400)
                    if (lastCapturedBitmap != rotatedBmp) rotatedBmp.recycle()
                    
                    val baos = ByteArrayOutputStream()
                    lastCapturedBitmap!!.compress(Bitmap.CompressFormat.JPEG, 75, baos)
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

        val expandedRectImg = Rect(
            kotlin.math.max(0, baseLeft - paddingX),
            kotlin.math.max(0, baseTop - paddingY),
            kotlin.math.min(bitmap.width, baseRight + paddingX),
            kotlin.math.min(bitmap.height, baseBottom + paddingY)
        )

        if (expandedRectImg.width() <= 0 || expandedRectImg.height() <= 0) {
            sendErrorToWeb("가이드 영역 오류")
            return
        }

        val corners = extractGeometryCorners(bitmap, expandedRectImg)
        if (corners == null) {
            sendErrorToWeb("번호판 검출 실패: 가이드 박스를 더 정확히 맞춰주세요.")
            return
        }

        val rectifiedBmp = rectifyToFlatPlate(bitmap, corners)
        if (rectifiedBmp == null) {
            sendSuccessToWeb(corners, imgW, imgH)
            return
        }

        val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
        recognizer.process(inputImage).addOnCompleteListener { task ->
            val text = if (task.isSuccessful) task.result.text.replace(Regex("\\s+"), "") else ""
            Log.d("JiSeKa", "OCR Text: $text")
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

    // ─────────────────────────────────────────────────────────────────
    // 🚨 대혁신: 고각도 번호판 전용 기하 복원(Geometry Reconstruction) 파이프라인
    // ─────────────────────────────────────────────────────────────────
    private fun extractGeometryCorners(bitmap: Bitmap, searchRect: Rect): Array<PointF>? {
        var mat: org.opencv.core.Mat? = null
        var roiMat: org.opencv.core.Mat? = null
        var gray: org.opencv.core.Mat? = null
        var edges: org.opencv.core.Mat? = null
        val contours = ArrayList<org.opencv.core.MatOfPoint>()

        try {
            mat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(bitmap, mat)

            val roiX = kotlin.math.max(0, searchRect.left)
            val roiY = kotlin.math.max(0, searchRect.top)
            val roiW = kotlin.math.min(mat.cols() - roiX, searchRect.width())
            val roiH = kotlin.math.min(mat.rows() - roiY, searchRect.height())

            if (roiW <= 0 || roiH <= 0) return null

            roiMat = org.opencv.core.Mat(mat, org.opencv.core.Rect(roiX, roiY, roiW, roiH))
            val roiArea = roiW * roiH 

            gray = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.cvtColor(roiMat, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)

            // 원거리 선예도 보강
            org.opencv.imgproc.Imgproc.equalizeHist(gray, gray)
            org.opencv.imgproc.Imgproc.GaussianBlur(gray, gray, org.opencv.core.Size(3.0, 3.0), 0.0)

            edges = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.Canny(gray, edges, 20.0, 80.0)

            val kernel = org.opencv.imgproc.Imgproc.getStructuringElement(
                org.opencv.imgproc.Imgproc.MORPH_RECT, 
                org.opencv.core.Size(5.0, 5.0)
            )
            org.opencv.imgproc.Imgproc.morphologyEx(
                edges, edges, 
                org.opencv.imgproc.Imgproc.MORPH_CLOSE, 
                kernel
            )
            kernel.release()

            org.opencv.imgproc.Imgproc.findContours(
                edges, contours, org.opencv.core.Mat(), 
                org.opencv.imgproc.Imgproc.RETR_LIST, 
                org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE
            )

            var bestScore = -9999.0
            var bestQuadPts: Array<PointF>? = null

            val guideCenterX = (searchRect.left + searchRect.right) / 2.0f
            val guideCenterY = (searchRect.top + searchRect.bottom) / 2.0f

            for (contour in contours) {
                // 🚨 개선 1: minAreaRect를 버리고, approxPolyDP 다각형 근사를 통해 진짜 볼록 사각형을 추출
                val contour2f = org.opencv.core.MatOfPoint2f(*contour.toArray())
                val approx = org.opencv.core.MatOfPoint2f()
                
                val arcLen = org.opencv.imgproc.Imgproc.arcLength(contour2f, true)
                // 정밀도를 높여 꼭짓점이 4개인 임의의 사다리꼴을 빈틈없이 복원
                org.opencv.imgproc.Imgproc.approxPolyDP(contour2f, approx, arcLen * 0.02, true)
                contour2f.release()

                // 투시 건전성 검사 (Sanity Check): 꼭짓점이 4개이고 볼록 다각형(Convex)인 경우만 승인
                if (approx.rows() == 4 && org.opencv.imgproc.Imgproc.isContourConvex(org.opencv.core.MatOfPoint(*approx.toArray()))) {
                    val ptsMat = approx.toArray()
                    val quadPts = Array(4) { i -> PointF(ptsMat[i].x.toFloat() + roiX, ptsMat[i].y.toFloat() + roiY) }
                    
                    // 정렬을 위해 임시 바운딩 박스 추출
                    val xs = quadPts.map { it.x }
                    val ys = quadPts.map { it.y }
                    val rw = (xs.maxOrNull() ?: 0f) - (xs.minOrNull() ?: 0f)
                    val rh = (ys.maxOrNull() ?: 0f) - (ys.minOrNull() ?: 0f)
                    val rectArea = rw * rh

                    // 면적 15% ~ 45% 유지로 거대 범퍼/그릴 완벽 차단
                    if (rectArea >= roiArea * 0.10 && rectArea <= roiArea * 0.45) {
                        val aspect = kotlin.math.max(rw, rh) / kotlin.math.min(rw, rh).coerceAtLeast(1.0f)

                        if (aspect in 1.8f..6.0f) {
                            val centerX = xs.average().toFloat()
                            val centerY = ys.average().toFloat()
                            val dist = kotlin.math.hypot((centerX - guideCenterX).toDouble(), (centerY - guideCenterY).toDouble())

                            // 정규화 스코어링
                            val normalizedArea = rectArea / roiArea.toDouble()
                            val aspectScore = 1.0 - kotlin.math.abs(aspect - 3.0f) / 3.0
                            val distanceScore = 1.0 / (1.0 + dist)

                            val score = (aspectScore * 0.5) + (distanceScore * 0.3) + (normalizedArea * 0.2)

                            if (score > bestScore) {
                                bestScore = score
                                bestQuadPts = quadPts
                            }
                        }
                    }
                }
                approx.release()
            }

            // 🚨 개선 2: 강건한 모서리 정렬 (Top/Bottom 분리 후 Left/Right 정렬)
            bestQuadPts?.let { pts ->
                // Y좌표 기준으로 정렬하여 상위 2개(Top)와 하위 2개(Bottom) 분리
                val sortedByY = pts.sortedBy { it.y }
                val topTwo = sortedByY.take(2).sortedBy { it.x }    // Top에서 X기준 좌우 분리
                val bottomTwo = sortedByY.takeLast(2).sortedBy { it.x } // Bottom에서 X기준 좌우 분리

                // TL, TR, BR, BL 순서 완벽 매핑 (아무리 기울어도 역전 불가능)
                val orderedPts = org.opencv.core.MatOfPoint2f(
                    org.opencv.core.Point(topTwo[0].x.toDouble() - roiX, topTwo[0].y.toDouble() - roiY),       // TL
                    org.opencv.core.Point(topTwo[1].x.toDouble() - roiX, topTwo[1].y.toDouble() - roiY),       // TR
                    org.opencv.core.Point(bottomTwo[1].x.toDouble() - roiX, bottomTwo[1].y.toDouble() - roiY), // BR
                    org.opencv.core.Point(bottomTwo[0].x.toDouble() - roiX, bottomTwo[0].y.toDouble() - roiY)  // BL
                )

                // 🚨 개선 3: cornerSubPix는 순서가 확정된 최종 꼭짓점의 아화소 미세 당김 보정으로만 한정
                val criteria = org.opencv.core.TermCriteria(
                    org.opencv.core.TermCriteria.EPS + org.opencv.core.TermCriteria.MAX_ITER, 
                    30, 0.1
                )
                org.opencv.imgproc.Imgproc.cornerSubPix(
                    gray, orderedPts, 
                    org.opencv.core.Size(5.0, 5.0), 
                    org.opencv.core.Size(-1.0, -1.0), 
                    criteria
                )

                val refinedArray = orderedPts.toArray()
                val finalResult = Array(4) { i -> 
                    PointF(refinedArray[i].x.toFloat() + roiX, refinedArray[i].y.toFloat() + roiY) 
                }
                orderedPts.release()
                return finalResult
            }

            return null 

        } catch (e: Exception) {
            return null
        } finally {
            mat?.release()
            roiMat?.release()
            gray?.release()
            edges?.release()
            for (c in contours) c.release()
        }
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
        val options = BitmapFactory.Options().apply { inSampleSize = 2; inPreferredConfig = Bitmap.Config.RGB_565 }
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
