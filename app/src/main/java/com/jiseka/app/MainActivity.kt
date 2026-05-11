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

        // 🚨 파이프라인 전면 개편: 문자 기반 탐색(Text Localization) + 선분 교차 복원(Line Intersection)
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
    // 🚀 Stage 1 ~ Stage 3: 문자 텍스처 탐색 및 선분 교차(Intersection) 정밀 복원 엔진
    // ─────────────────────────────────────────────────────────────────
    private fun extractGeometryCorners(bitmap: Bitmap, searchRect: Rect): Array<PointF>? {
        var mat: org.opencv.core.Mat? = null
        var roiMat: org.opencv.core.Mat? = null
        var gray: org.opencv.core.Mat? = null
        var sobel: org.opencv.core.Mat? = null
        var morph: org.opencv.core.Mat? = null
        var edges: org.opencv.core.Mat? = null

        try {
            mat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(bitmap, mat)

            val roiX = kotlin.math.max(0, searchRect.left)
            val roiY = kotlin.math.top
            val roiW = kotlin.math.min(mat.cols() - roiX, searchRect.width())
            val roiH = kotlin.math.min(mat.rows() - roiY, searchRect.height())

            if (roiW <= 0 || roiH <= 0) return null

            roiMat = org.opencv.core.Mat(mat, org.opencv.core.Rect(roiX, roiY, roiW, roiH))
            gray = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.cvtColor(roiMat, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)

            // 1. CLAHE 대비 극대화
            val clahe = org.opencv.imgproc.Imgproc.createCLAHE(2.0, org.opencv.core.Size(8.0, 8.0))
            clahe.apply(gray, gray)
            clahe.release()

            org.opencv.imgproc.Imgproc.GaussianBlur(gray, gray, org.opencv.core.Size(3.0, 3.0), 0.0)

            // ─────────────────────────────────────────────────────────────
            // 🎯 Stage 1: 문자 구조 기반 탐색 (Text Texture Localization)
            // ─────────────────────────────────────────────────────────────
            // 번호판의 본질인 "세로 에지(Vertical Stroke) 반복 패턴"을 추출합니다.
            sobel = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.Sobel(gray, sobel, org.opencv.core.CvType.CV_8U, 1, 0, 3, 1.0, 0.0, org.opencv.core.Core.BORDER_DEFAULT)
            org.opencv.imgproc.Imgproc.threshold(sobel, sobel, 0.0, 255.0, org.opencv.imgproc.Imgproc.THRESH_BINARY or org.opencv.imgproc.Imgproc.THRESH_OTSU)

            // 수평 커널(25x5)을 적용하여 개별 문자들의 세로 획을 하나의 긴 "번호판 띠"로 융합합니다.
            morph = org.opencv.core.Mat()
            val kernel = org.opencv.imgproc.Imgproc.getStructuringElement(org.opencv.imgproc.Imgproc.MORPH_RECT, org.opencv.core.Size(25.0, 5.0))
            org.opencv.imgproc.Imgproc.morphologyEx(sobel, morph, org.opencv.imgproc.Imgproc.MORPH_CLOSE, kernel)
            kernel.release()

            // 문자 클러스터 탐색을 통해 가장 번호판 영역다운 띠(Hypothesis Bbox)를 생성
            val textContours = ArrayList<org.opencv.core.MatOfPoint>()
            org.opencv.imgproc.Imgproc.findContours(morph, textContours, org.opencv.core.Mat(), org.opencv.imgproc.Imgproc.RETR_EXTERNAL, org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE)

            var plateRegionRect: org.opencv.core.Rect? = null
            var maxTextScore = -9999.0
            val totalRoiArea = roiW * roiH

            for (c in textContours) {
                val r = org.opencv.imgproc.Imgproc.boundingRect(c)
                val area = r.width * r.height
                if (area < totalRoiArea * 0.05 || area > totalRoiArea * 0.60) continue

                val aspect = r.width.toFloat() / r.height.toFloat().coerceAtLeast(1.0f)
                if (aspect in 1.5f..6.0f) {
                    val score = area * aspect // 가로로 길게 뭉친 띠를 최우선으로 선정
                    if (score > maxTextScore) {
                        maxTextScore = score.toDouble()
                        plateRegionRect = r
                    }
                }
            }
            for (c in textContours) c.release()

            // 문자가 탐색되지 않았다면 기존 Strict ROI 전체를 대상으로 대체 탐색
            val safePlateRect = plateRegionRect ?: org.opencv.core.Rect(0, 0, roiW, roiH)

            // ─────────────────────────────────────────────────────────────
            // 🎯 Stage 2 & 3: 서브 ROI 집중 탐색 및 선분 교차 복원 (Line-based Reconstruction)
            // ─────────────────────────────────────────────────────────────
            // 문자 띠 영역 주변에만 집중하여 범퍼나 그릴 간섭을 0%로 제거합니다.
            val subRoiX = safePlateRect.x
            val subRoiY = safePlateRect.y
            val subRoiW = safePlateRect.width
            val subRoiH = safePlateRect.height

            val subGray = org.opencv.core.Mat(gray, org.opencv.core.Rect(subRoiX, subRoiY, subRoiW, subRoiH))
            edges = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.Canny(subGray, edges, 10.0, 50.0)

            // HoughLinesP를 사용하여 번호판 상/하단의 강력한 크롬 및 그림자 에지 선분을 추출
            val lines = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.HoughLinesP(edges, lines, 1.0, kotlin.math.PI / 180.0, 20, subRoiW * 0.3, 10.0)

            val horizontalLines = ArrayList<LineSeg>()
            val verticalLines = ArrayList<LineSeg>()

            for (i in 0 until lines.rows()) {
                val vec = lines.get(i, 0)
                val lx1 = vec[0].toFloat()
                val ly1 = vec[1].toFloat()
                val lx2 = vec[2].toFloat()
                val ly2 = vec[3].toFloat()
                
                val angle = kotlin.math.abs(kotlin.math.atan2((ly2 - ly1).toDouble(), (lx2 - lx1).toDouble()) * 180.0 / kotlin.math.PI)
                val seg = LineSeg(PointF(lx1, ly1), PointF(lx2, ly2))

                if (angle < 35.0 || angle > 145.0) {
                    horizontalLines.add(seg)
                } else if (angle in 55.0..125.0) {
                    verticalLines.add(seg)
                }
            }
            lines.release()
            subGray.release()

            // 선분이 부족하면 기존 다각형 근사 알고리즘으로 안전하게 우회(Fallback)
            if (horizontalLines.size < 2 || verticalLines.size < 2) {
                return fallbackPolyReconstruction(edges, roiX + subRoiX, roiY + subRoiY, subRoiW, subRoiH, gray)
            }

            // 가장 위쪽과 아래쪽에 위치한 수평선(Top/Bottom Edges) 추출
            horizontalLines.sortBy { (it.p1.y + it.p2.y) / 2.0f }
            val topEdge = horizontalLines.first()
            val bottomEdge = horizontalLines.last()

            // 가장 왼쪽과 오른쪽에 위치한 수직선(Left/Right Edges) 추출
            verticalLines.sortBy { (it.p1.x + it.p2.x) / 2.0f }
            val leftEdge = verticalLines.first()
            val rightEdge = verticalLines.last()

            // 4개의 선분을 연장하여 수학적 교차점(Intersection) 계산
            val tl = intersection(topEdge, leftEdge) ?: return null
            val tr = intersection(topEdge, rightEdge) ?: return null
            val br = intersection(bottomEdge, rightEdge) ?: return null
            val bl = intersection(bottomEdge, leftEdge) ?: return null

            // 절대 좌표 매핑 (ROI 및 subROI 오프셋 합산)
            val baseOffset = PointF((roiX + subRoiX).toFloat(), (roiY + subRoiY).toFloat())
            val rawPts = arrayOf(
                PointF(tl.x + baseOffset.x, tl.y + baseOffset.y),
                PointF(tr.x + baseOffset.x, tr.y + baseOffset.y),
                PointF(br.x + baseOffset.x, br.y + baseOffset.y),
                PointF(bl.x + baseOffset.x, bl.y + baseOffset.y)
            )

            // 최종 모서리 Subpix 정밀 보정
            val orderedPts = org.opencv.core.MatOfPoint2f(
                org.opencv.core.Point(rawPts[0].x.toDouble(), rawPts[0].y.toDouble()),
                org.opencv.core.Point(rawPts[1].x.toDouble(), rawPts[1].y.toDouble()),
                org.opencv.core.Point(rawPts[2].x.toDouble(), rawPts[2].y.toDouble()),
                org.opencv.core.Point(rawPts[3].x.toDouble(), rawPts[3].y.toDouble())
            )

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
        }
    }

    // 선분 구조체 및 수학적 교차점 연산 헬퍼
    data class LineSeg(val p1: PointF, val p2: PointF)

    private fun intersection(l1: LineSeg, l2: LineSeg): PointF? {
        val x1 = l1.p1.x; val y1 = l1.p1.y
        val x2 = l1.p2.x; val y2 = l1.p2.y
        val x3 = l2.p1.x; val y3 = l2.p1.y
        val x4 = l2.p2.x; val y4 = l2.p2.y

        val denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (kotlin.math.abs(denom) < 1e-6f) return null // 평행선 방어

        val t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        val px = x1 + t * (x2 - x1)
        val py = y1 + t * (y2 - y1)
        return PointF(px, py)
    }

    // 선분 부족 시 가동되는 기존 안전 다각형 복원 로직
    private fun fallbackPolyReconstruction(edges: org.opencv.core.Mat, absX: Int, absY: Int, sw: Int, sh: Int, gray: org.opencv.core.Mat): Array<PointF>? {
        val contours = ArrayList<org.opencv.core.MatOfPoint>()
        org.opencv.imgproc.Imgproc.findContours(edges, contours, org.opencv.core.Mat(), org.opencv.imgproc.Imgproc.RETR_LIST, org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE)

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

                if (area >= totalArea * 0.10 && area <= totalArea * 0.90) {
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
        for (c in contours) c.release()

        bestQuad?.let { pts ->
            val sortedByY = pts.sortedBy { it.y }
            val topTwo = sortedByY.take(2).sortedBy { it.x }
            val bottomTwo = sortedByY.takeLast(2).sortedBy { it.x }

            val orderedPts = org.opencv.core.MatOfPoint2f(
                org.opencv.core.Point(topTwo[0].x.toDouble(), topTwo[0].y.toDouble()),
                org.opencv.core.Point(topTwo[1].x.toDouble(), topTwo[1].y.toDouble()),
                org.opencv.core.Point(bottomTwo[1].x.toDouble(), bottomTwo[1].y.toDouble()),
                org.opencv.core.Point(bottomTwo[0].x.toDouble(), bottomTwo[0].y.toDouble())
            )
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
