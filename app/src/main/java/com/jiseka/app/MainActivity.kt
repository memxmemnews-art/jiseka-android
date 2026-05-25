package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
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
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {

    private var webView: WebView? = null
    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null
    private var displayedBitmap: Bitmap? = null

    private val transformLock = Any()
    private var lastTransformData: CaptureTransformData? = null

    @Volatile private var isProcessing = false
    @Volatile private var isHighResCaptureReady = false
    private var pendingAnalysisMode: String? = null

    private lateinit var previewDir: File
    private lateinit var assetLoader: WebViewAssetLoader

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("MainActivity", "OpenCV initialization failed.")
        }

        viewFinder = findViewById(R.id.viewFinder)
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeGuideView = findViewById(R.id.nativeGuideView)
        webView = findViewById(R.id.webView)

        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()
        previewDir = File(cacheDir, "preview_images").apply { if (!exists()) mkdirs() }

        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()

        setupWebView()

        nativeGuideView?.onGuideDropListener = { mode ->
            val points = nativeGuideView?.getCorners() ?: emptyList()
            val jsPointsArray = points.joinToString(prefix = "[", postfix = "]") { p ->
                "{\"x\":${p.x},\"y\":${p.y}}"
            }
            safeEvaluate("window.onNativeGuideMoved('$mode', $jsPointsArray)")
        }

        if (allPermissionsGranted()) {
            viewFinder?.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView?.apply {
            setBackgroundColor(Color.TRANSPARENT)
            setLayerType(View.LAYER_TYPE_HARDWARE, null)
            
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE

            webViewClient = object : WebViewClientCompat() {
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
                    return assetLoader.shouldInterceptRequest(request.url)
                }
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    safeEvaluate("window.onNativeReady?.()")
                }
            }
            webChromeClient = WebChromeClient()
            addJavascriptInterface(AndroidBridge(), "AndroidBridge")
            loadUrl("https://appassets.androidplatform.net/assets/index.html")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("MainActivity", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    inner class AndroidBridge {
        @JavascriptInterface
        fun takePhoto() {
            // [수정사항 1] 안전하게 메인 스레드로 즉시 전환하여 PreviewView.bitmap 동기화 유도
            runOnUiThread {
                this@MainActivity.takePhoto()
            }
        }

        @JavascriptInterface
        fun changeGuideMode(mode: String) {
            runOnUiThread { nativeGuideView?.setMode(mode) }
        }

        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                if (isVisible) {
                    viewFinder?.visibility = View.VISIBLE
                    nativeBackgroundView?.visibility = View.GONE
                    nativeGuideView?.visibility = View.GONE
                    isProcessing = false
                }
            }
        }

        @JavascriptInterface
        fun requestAnalysis(mode: String) {
            runOnUiThread {
                if (isHighResCaptureReady) {
                    triggerAnalysis(mode)
                } else {
                    pendingAnalysisMode = mode
                }
            }
        }

        @JavascriptInterface
        fun saveFinalImage() {
            runOnUiThread {
                val bitmapToSave = displayedBitmap
                if (bitmapToSave != null) {
                    saveBitmapToGallery(bitmapToSave)
                    safeEvaluate("window.onNativeSaveComplete()")
                } else {
                    Toast.makeText(this@MainActivity, "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
                    safeEvaluate("window.onNativeSaveComplete()")
                }
            }
        }

        @JavascriptInterface
        fun clearResultBitmap() {
            runOnUiThread {
                nativeBackgroundView?.setImageBitmap(null)
                displayedBitmap?.recycle()
                displayedBitmap = null
            }
        }
    }

    private fun takePhoto() {
        if (isProcessing) return
        isProcessing = true
        isHighResCaptureReady = false
        pendingAnalysisMode = null

        val currentCapture = imageCapture ?: run {
            isProcessing = false
            safeEvaluate("window.onNativeError('카메라가 초기화되지 않았습니다.')")
            return
        }

        // UI 스레드 보장 하에 안전하게 렌더링 스냅샷 생성
        val freezeBitmap = viewFinder?.bitmap
        if (freezeBitmap != null) {
            updateNativeBackgroundSafe(freezeBitmap)
            nativeBackgroundView?.visibility = View.VISIBLE
            nativeGuideView?.visibility = View.VISIBLE
            safeEvaluate("window.onNativeScreenFrozen()")
        }

        currentCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    try {
                        val uprightBitmap = imageProxy.toUprightBitmap()
                        val transformData = CaptureTransformData(
                            sensorToBufferMatrix = imageProxy.imageInfo.sensorToBufferTransformMatrix,
                            rotationDegrees = imageProxy.imageInfo.rotationDegrees,
                            bufferWidth = imageProxy.width,
                            bufferHeight = imageProxy.height
                        )

                        synchronized(bitmapLock) {
                            if (lastCapturedBitmap !== uprightBitmap && lastCapturedBitmap?.isRecycled == false) {
                                lastCapturedBitmap?.recycle()
                            }
                            lastCapturedBitmap = uprightBitmap
                        }
                        synchronized(transformLock) { lastTransformData = transformData }

                        isHighResCaptureReady = true

                        runOnUiThread {
                            pendingAnalysisMode?.let { mode ->
                                pendingAnalysisMode = null
                                triggerAnalysis(mode)
                            }
                            safeEvaluate("window.onNativePhotoCaptured()")
                        }
                    } catch (t: Throwable) {
                        // [수정사항 2] Exception 상위인 Throwable(OOM 포함)까지 캐치 처리 보강
                        runOnUiThread {
                            isProcessing = false
                            safeEvaluate("window.onNativeError('이미지 처리 중 오류가 발생했습니다.')")
                        }
                    } finally {
                        // [수정사항 2] 예외 발생 여부와 관계없이 무조건 호출되어 파이프라인 정체를 막음
                        imageProxy.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        isProcessing = false
                        safeEvaluate("window.onNativeError('카메라 캡처에 실패했습니다.')")
                    }
                }
            }
        )
    }

    private fun triggerAnalysis(mode: String) {
        val targetBitmap = synchronized(bitmapLock) { lastCapturedBitmap }
        val transformData = synchronized(transformLock) { lastTransformData }
        val guideView = nativeGuideView

        if (targetBitmap == null || transformData == null || guideView == null) {
            isProcessing = false
            safeEvaluate("window.onNativeError('분석 준비가 되지 않았습니다.')")
            return
        }

        val uiCorners = guideView.getCorners()
        analysisExecutor.execute {
            try {
                val exactCorners = CameraCoordinateConverter.mapUiToExactBitmap(
                    viewFinder!!, transformData, uiCorners, targetBitmap.width, targetBitmap.height
                )

                val resultMat = Mat()
                Utils.bitmapToMat(targetBitmap, resultMat)

                val workingMat = resultMat.copy(CvType.CV_8UC4, true)
                val detectedCorners = PlateDetectionEngine.findPlateCorners(workingMat)
                workingMat.release()

                val finalCorners = detectedCorners ?: exactCorners
                applyMaskToMat(resultMat, finalCorners, mode)

                val resultBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(resultMat, resultBitmap)
                resultMat.release()

                val tempFile = File(previewDir, "temp_result.jpg")
                tempFile.outputStream().use { out ->
                    resultBitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
                }

                runOnUiThread {
                    displayedBitmap?.recycle()
                    displayedBitmap = resultBitmap

                    nativeBackgroundView?.visibility = View.VISIBLE
                    nativeBackgroundView?.setImageBitmap(resultBitmap)
                    nativeGuideView?.visibility = View.GONE

                    val urlParam = "https://appassets.androidplatform.net/assets/preview_images/temp_result.jpg"
                    safeEvaluate("window.onNativeAnalysisComplete('$urlParam')")
                    isProcessing = false
                }
            } catch (e: Exception) {
                runOnUiThread {
                    isProcessing = false
                    safeEvaluate("window.onNativeError('분석 과정 중 예외가 발생했습니다.')")
                }
            }
        }
    }

    private fun applyMaskToMat(mat: Mat, corners: List<PointF>, mode: String) {
        if (corners.size != 4) return
        val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }

        val maskColor = when (mode) {
            "WAX" -> Scalar(0.0, 255.0, 0.0, 255.0)
            "GLOSS" -> Scalar(0.0, 0.0, 255.0, 255.0)
            "CONTAM" -> Scalar(255.0, 0.0, 0.0, 255.0)
            else -> Scalar(0.0, 255.0, 0.0, 255.0)
        }

        val srcPoints = MatOfPoint2f(*pts.toTypedArray())
        val minX = pts.minOf { it.x }; val maxX = pts.maxOf { it.x }
        val minY = pts.minOf { it.y }; val maxY = pts.maxOf { it.y }
        val w = maxX - minX; val h = maxY - minY

        val targetRatio = 3.0 / 1.0
        var newW = w; var newH = h
        if (w / h > targetRatio) { newH = w / targetRatio } else { newW = h * targetRatio }

        val center = Point((minX + maxX) / 2.0, (minY + maxY) / 2.0)
        val dstPoints = MatOfPoint2f(
            Point(center.x - newW / 2.0, center.y - newH / 2.0),
            Point(center.x + newW / 2.0, center.y - newH / 2.0),
            Point(center.x + newW / 2.0, center.y + newH / 2.0),
            Point(center.x - newW / 2.0, center.y + newH / 2.0)
        )

        val transformMatrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
        val transformedMat = Mat()
        Imgproc.warpPerspective(mat, transformedMat, transformMatrix, Size(mat.cols().toDouble(), mat.rows().toDouble()))

        val maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
        val contour = org.opencv.core.MatOfPoint(*dstPoints.toArray().map { Point(it.x, it.y) }.toTypedArray())
        Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))

        val blurredMask = Mat()
        Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)

        val coloredMask = Mat(mat.size(), mat.type(), maskColor)
        val alphaMat = Mat()
        blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)

        val matChannels = ArrayList<Mat>()
        val coloredChannels = ArrayList<Mat>()
        Core.split(mat, matChannels)
        Core.split(coloredMask, coloredChannels)

        for (i in 0 until 3) {
            val mc = matChannels[i]
            val cc = coloredChannels[i]
            val mcF = Mat(); val ccF = Mat()
            mc.convertTo(mcF, CvType.CV_32F)
            cc.convertTo(ccF, CvType.CV_32F)

            val blendedF = Mat()
            Core.multiply(ccF, alphaMat, ccF)
            val invAlpha = Mat()
            Core.subtract(Scalar(1.0), alphaMat, invAlpha)
            Core.multiply(mcF, invAlpha, mcF)
            Core.add(ccF, mcF, blendedF)

            blendedF.convertTo(matChannels[i], CvType.CV_8U)
            mcF.release(); ccF.release(); blendedF.release(); invAlpha.release()
        }

        Core.merge(matChannels, mat)

        srcPoints.release(); dstPoints.release(); transformMatrix.release()
        transformedMat.release(); maskMat.release(); contour.release()
        blurredMask.release(); coloredMask.release(); alphaMat.release()
        matChannels.forEach { it.release() }; coloredChannels.forEach { it.release() }
    }

    private fun updateNativeBackgroundSafe(bitmap: Bitmap) {
        val copy = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        runOnUiThread {
            nativeBackgroundView?.setImageBitmap(copy)
        }
    }

    private fun safeEvaluate(script: String) {
        webView?.post { webView?.evaluateJavascript(script, null) }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/JiSeKa")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        }

        runOnUiThread { Toast.makeText(this, "💾 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) viewFinder?.post { startCamera() } else finish()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        nativeBackgroundView?.setImageBitmap(null)
        displayedBitmap?.recycle()
        displayedBitmap = null

        cameraExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        previewDir.listFiles()?.forEach { it.delete() }
        webView?.apply { stopLoading(); clearHistory(); removeAllViews(); destroy() }
        webView = null
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
