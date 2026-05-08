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
import android.graphics.YuvImage
import android.net.Uri
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.View
import android.webkit.JavascriptInterface
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
import java.util.Collections
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

import com.jiseka.app.BuildConfig
import com.jiseka.app.R

// 🚨 아까 날아갔던 OpenCV 필수 수입(import) 코드들! (에러의 원인)
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var webView: WebView
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) Log.e("JiSeKa", "OpenCV 초기화 실패")

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        webView.setLayerType(View.LAYER_TYPE_HARDWARE, null)
        webView.setBackgroundColor(Color.TRANSPARENT)
        
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            cacheMode = WebSettings.LOAD_DEFAULT
        }

        webView.addJavascriptInterface(WebAppInterface(), "AndroidBridge")
        
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val host = request?.url?.host ?: return true
                val allowedHost = Uri.parse(BuildConfig.VERCEL_URL).host
                return host != allowedHost
            }
        }
        
        webView.loadUrl(BuildConfig.VERCEL_URL)

        if (allPermissionsGranted()) startCamera() 
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun takePicture() { runOnUiThread { captureAndProcess() } }
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

    private fun captureAndProcess() {
        val capture = imageCapture ?: return
        if (!isCapturing.compareAndSet(false, true)) return

        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            @SuppressLint("UnsafeOptInUsageError")
            override fun onCaptureSuccess(imageProxy: ImageProxy) { processEngineBackground(imageProxy) }
            override fun onError(exception: ImageCaptureException) { isCapturing.set(false) }
        })
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processEngineBackground(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmapExt()
        val rotation = imageProxy.imageInfo.rotationDegrees
        imageProxy.close() 

        var originalBmp = bitmap
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated != bitmap) bitmap.recycle()
            originalBmp = rotated
        }

        val safeBmp = downscaleBitmapIfNeeded(originalBmp, 800)
        val viewW = viewFinder.width.toFloat(); val viewH = viewFinder.height.toFloat()
        val imgW = safeBmp.width.toFloat(); val imgH = safeBmp.height.toFloat()
        
        val scale = max(viewW / imgW, viewH / imgH)
        val scaledW = imgW * scale; val scaledH = imgH * scale
        val offsetX = (viewW - scaledW) / 2f; val offsetY = (viewH - scaledH) / 2f

        val guideViewW = viewW * 0.8f; val guideViewH = guideViewW / 3f
        val guideRectImg = Rect(
            ((viewW / 2f - guideViewW / 2f - offsetX) / scale).toInt(),
            ((viewH / 2f - guideViewH / 2f - offsetY) / scale).toInt(),
            ((viewW / 2f + guideViewW / 2f - offsetX) / scale).toInt(),
            ((viewH / 2f + guideViewH / 2f - offsetY) / scale).toInt()
        )

        val corners = extractGeometryCorners(safeBmp, guideRectImg)
        if (corners == null) {
            sendErrorToWeb("번호판을 찾을 수 없습니다.")
            return
        }

        val rectifiedBmp = rectifyToFlatPlate(safeBmp, corners)
        if (rectifiedBmp == null) {
            sendErrorToWeb("평면화에 실패했습니다.")
            return
        }

        val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
        recognizer.process(inputImage).addOnCompleteListener { task ->
            val ocrValid = task.isSuccessful && task.result.text.isNotEmpty()
            if (!ocrValid) {
                sendErrorToWeb("번호판 검증 실패")
                rectifiedBmp.recycle()
                return@addOnCompleteListener
            }
            sendSuccessToWeb(safeBmp, corners)
            rectifiedBmp.recycle()
        }
    }

    private fun sendErrorToWeb(msg: String) {
        runOnUiThread {
            val js = "javascript:window.onNativeError(${JSONObject.quote(msg)})"
            webView.evaluateJavascript(js, null)
            isCapturing.set(false)
        }
    }

    private fun sendSuccessToWeb(bitmap: Bitmap, corners: Array<PointF>) {
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 60, baos)
        val base64Img = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
        
        val payload = JSONObject().apply {
            put("version", 1)
            put("image", "data:image/jpeg;base64,$base64Img")
            put("corners", JSONArray().apply {
                corners.forEach { 
                    put(JSONObject().apply { put("x", it.x); put("y", it.y) })
                }
            })
        }

        runOnUiThread {
            val js = "window.onNativeSuccess(${JSONObject.quote(payload.toString())})"
            webView.evaluateJavascript(js, null)
            isCapturing.set(false)
        }
    }

    private fun downscaleBitmapIfNeeded(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val maxDim = max(bitmap.width, bitmap.height)
        if (maxDim <= maxDimension) return bitmap
        val scale = maxDimension.toFloat() / maxDim
        return Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
    }

    private fun extractGeometryCorners(bitmap: Bitmap, guideRect: Rect): Array<PointF>? {
        var mat: Mat? = null; var roiMat: Mat? = null; var gray: Mat? = null; var edges: Mat? = null
        val contours = ArrayList<MatOfPoint>()
        try {
            mat = Mat(); Utils.bitmapToMat(bitmap, mat)
            val roiX = max(0, guideRect.left); val roiY = max(0, guideRect.top)
            val roiW = min(guideRect.width(), mat.cols() - roiX); val roiH = min(guideRect.height(), mat.rows() - roiY)
            if (roiW <= 0 || roiH <= 0) return null
            roiMat = Mat(mat, org.opencv.core.Rect(roiX, roiY, roiW, roiH))
            gray = Mat(); Imgproc.cvtColor(roiMat, gray, Imgproc.COLOR_RGBA2GRAY)
            edges = Mat(); Imgproc.Canny(gray, edges, 50.0, 150.0)
            Imgproc.findContours(edges, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestScore = 0.0; var bestBox: Array<PointF>? = null
            for (contour in contours) {
                val minRect = Imgproc.minAreaRect(MatOfPoint2f(*contour.toArray()))
                val rw = minRect.size.width; val rh = minRect.size.height
                val aspect = max(rw, rh) / min(rw, rh).coerceAtLeast(1.0)
                if (aspect in 2.0..6.0) {
                    val overlapRatio = (minRect.size.area() / (guideRect.width() * guideRect.height())).coerceIn(0.0, 1.0)
                    val score = overlapRatio + (1.0 / aspect)
                    if (score > bestScore) {
                        bestScore = score
                        val pts = Mat(); Imgproc.boxPoints(minRect, pts)
                        bestBox = Array(4) { i -> PointF(pts.get(i,0)[0].toFloat() + roiX, pts.get(i,1)[0].toFloat() + roiY) }
                        pts.release()
                    }
                }
            }
            bestBox?.let {
                val cx = it.map { p -> p.x }.average().toFloat(); val cy = it.map { p -> p.y }.average().toFloat()
                val sorted = it.sortedBy { p -> atan2(p.y - cy, p.x - cx) }.toMutableList()
                var area = 0f
                for (i in 0..3) {
                    val j = (i + 1) % 4
                    area += sorted[i].x * sorted[j].y - sorted[j].x * sorted[i].y
                }
                if (area < 0) sorted.reverse()
                val tlIdx = sorted.indices.minByOrNull { i -> sorted[i].x + sorted[i].y } ?: 0
                Collections.rotate(sorted, -tlIdx)
                return sorted.toTypedArray()
            }
            return null
        } finally {
            mat?.release(); roiMat?.release(); gray?.release(); edges?.release()
            for (c in contours) c.release()
        }
    }

    private fun rectifyToFlatPlate(bitmap: Bitmap, corners: Array<PointF>): Bitmap? {
        var bgMat: Mat? = null; var dest: Mat? = null
        try {
            bgMat = Mat(); Utils.bitmapToMat(bitmap, bgMat)
            val srcPts = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val dstPts = MatOfPoint2f(Point(0.0, 0.0), Point(400.0, 0.0), Point(400.0, 100.0), Point(0.0, 100.0))
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            dest = Mat(); Imgproc.warpPerspective(bgMat, dest, transform, Size(400.0, 100.0))
            val res = Bitmap.createBitmap(400, 100, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(dest, res)
            return res
        } catch (e: Exception) { return null } 
        finally { bgMat?.release(); dest?.release() }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun ImageProxy.toBitmapExt(): Bitmap {
        val yBuffer = planes[0].buffer; val uBuffer = planes[1].buffer; val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining(); val uSize = uBuffer.remaining(); val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize); vBuffer.get(nv21, ySize, vSize); uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
