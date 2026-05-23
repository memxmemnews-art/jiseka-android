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
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@OptIn(TransformExperimental::class, androidx.camera.core.ExperimentalGetImage::class)
class MainActivity : AppCompatActivity() {

    private var webView: WebView? = null
    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null
    
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    @Volatile private var isProcessing = false
    @Volatile private var isHighResCaptureReady = false
    private var pendingAnalysisMode: String? = null

    private var displayedBitmap: Bitmap? = null
    private var lastCapturedBitmap: Bitmap? = null
    private var lastTransformData: CaptureTransformData? = null
    
    private val bitmapLock = Any()
    private val transformLock = Any()

    private lateinit var previewDir: File
    private lateinit var assetLoader: WebViewAssetLoader

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (!OpenCVLoader.initDebug()) Log.e("CAMERA_DEBUG", "OpenCV 초기화 실패")

        viewFinder = findViewById(R.id.viewFinder)
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeGuideView = findViewById(R.id.nativeGuideView)
        webView = findViewById(R.id.webView)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newFixedThreadPool(2)

        previewDir = File(filesDir, "preview")
        if (!previewDir.exists()) previewDir.mkdirs()

        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/preview/", WebViewAssetLoader.InternalStoragePathHandler(this, previewDir))
            .build()

        setupWebView()

        // 가이드 박스 드래그 앤 드롭 완료 시 자동 분석 실행
        nativeGuideView?.onGuideDropListener = { mode ->
            runOnUiThread { safeEvaluate("showToast('분석 중입니다...')") }
            triggerAnalysis(mode)
        }

        if (allPermissionsGranted()) {
            viewFinder?.post { startCamera() }
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
            setLayerType(View.LAYER_TYPE_HARDWARE, null)
            setBackgroundColor(Color.TRANSPARENT)
            clearCache(true)
            
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                cacheMode = WebSettings.LOAD_NO_CACHE
            }
            webViewClient = object : WebViewClientCompat() {
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? = 
                    assetLoader.shouldInterceptRequest(request.url)
            }
            webChromeClient = WebChromeClient()
            addJavascriptInterface(AndroidBridge(), "AndroidBridge")
            loadUrl("https://ziseka-app.vercel.app/?v=" + System.currentTimeMillis())
        }
    }

    private fun updateNativeBackgroundSafe(newBitmap: Bitmap?) {
        val oldBitmap = displayedBitmap
        displayedBitmap = newBitmap
        nativeBackgroundView?.setImageBitmap(newBitmap)
        nativeBackgroundView?.post {
            if (oldBitmap != null && oldBitmap !== newBitmap && !oldBitmap.isRecycled) {
                oldBitmap.recycle()
            }
        }
    }

    inner class AndroidBridge {
        @JavascriptInterface fun takePhoto() { this@MainActivity.takePhoto() }
        
        @JavascriptInterface fun changeGuideMode(mode: String) {
            runOnUiThread { nativeGuideView?.setMode(mode) }
        }
        
        @JavascriptInterface fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                if (isVisible) {
                    viewFinder?.visibility = View.VISIBLE
                    nativeBackgroundView?.visibility = View.GONE
                    nativeGuideView?.visibility = View.GONE
                    isProcessing = false
                }
            }
        }

        @JavascriptInterface fun clearResultBitmap() {
            runOnUiThread {
                viewFinder?.visibility = View.VISIBLE
                nativeBackgroundView?.visibility = View.GONE
                nativeGuideView?.visibility = View.GONE
                updateNativeBackgroundSafe(null)
                synchronized(bitmapLock) {
                    lastCapturedBitmap?.recycle()
                    lastCapturedBitmap = null
                }
                isProcessing = false
            }
        }

        @JavascriptInterface fun analyzePlateWithMode(mode: String) {
            triggerAnalysis(mode)
        }

        @JavascriptInterface fun saveFinalImage() {
            analysisExecutor.execute {
                try {
                    val finalBitmapToSave = displayedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
                        ?: throw IllegalStateException("저장할 비트맵이 없습니다.")
                        
                    saveBitmapToGallery(finalBitmapToSave)
                    safeEvaluate("window.onNativeSaveComplete()")
                } catch (e: Exception) {
                    safeEvaluate("window.onNativeError('저장 실패')")
                }
            }
        }
        @JavascriptInterface fun showToast(msg: String) { 
            runOnUiThread { Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() } 
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE

            val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build()
                .also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                cameraProvider.unbindAll()
                val viewPort = viewFinder?.viewPort ?: return@addListener
                val useCaseGroup = androidx.camera.core.UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(imageCapture!!)
                    .build()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, useCaseGroup)
            } catch (e: Exception) { }
        }, ContextCompat.getMainExecutor(this))
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
        
        // 프리뷰 캡처로 화면 즉시 정지
        val freezeBitmap = viewFinder?.bitmap
        if (freezeBitmap != null) {
            runOnUiThread {
                updateNativeBackgroundSafe(freezeBitmap)
                nativeBackgroundView?.visibility = View.VISIBLE
                nativeGuideView?.visibility = View.VISIBLE
                safeEvaluate("window.onNativeScreenFrozen()")
            }
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
                        imageProxy.close()

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
                    } catch (e: Exception) {
                        imageProxy.close()
                        runOnUiThread { 
                            isProcessing = false
                            safeEvaluate("window.onNativeError('이미지 처리 중 오류가 발생했습니다.')")
                        }
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
        if (!isHighResCaptureReady) {
            pendingAnalysisMode = mode
            runOnUiThread { safeEvaluate("showToast('고해상도 원본을 처리 중입니다. 잠시만 대기해주세요 ✨')") }
            return
        }

        analysisExecutor.execute {
            if (isDestroyed || isFinishing) return@execute
            var processingBitmap: Bitmap? = null
            var roiBitmap: Bitmap? = null
            
            try {
                processingBitmap = synchronized(bitmapLock) { lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) } 
                    ?: throw IllegalStateException("캡처된 원본 비트맵이 없습니다.")
                val transformData = synchronized(transformLock) { lastTransformData } 
                    ?: throw IllegalStateException("메타데이터 누락")

                val view = viewFinder ?: throw IllegalStateException("ViewFinder가 초기화되지 않았습니다.")
                val uiCorners = nativeGuideView?.getCorners() ?: throw IllegalStateException("가이드 박스 좌표가 없습니다.")
           
                val exactBitmapPoints = CameraCoordinateConverter.mapUiToExactBitmap(
                    previewView = view, transformData = transformData, uiPoints = uiCorners,
                    uprightBitmapWidth = processingBitmap.width, uprightBitmapHeight = processingBitmap.height
                )

                // 1. ROI 좌표 크롭 영역 계산
                val minX = exactBitmapPoints.minOf { it.x }.toInt().coerceAtLeast(0)
                val minY = exactBitmapPoints.minOf { it.y }.toInt().coerceAtLeast(0)
                val maxX = exactBitmapPoints.maxOf { it.x }.toInt().coerceAtMost(processingBitmap.width)
                val maxY = exactBitmapPoints.maxOf { it.y }.toInt().coerceAtMost(processingBitmap.height)
                val roiWidth = maxX - minX
                val roiHeight = maxY - minY

                if (roiWidth <= 0 || roiHeight <= 0) throw IllegalStateException("유효하지 않은 가이드 박스 영역입니다.")

                roiBitmap = Bitmap.createBitmap(processingBitmap, minX, minY, roiWidth, roiHeight)

                // 2. 엔진 분석 실행
                val localCorners = PlateDetectionEngine.findPlateCorners(roiBitmap)

                if (localCorners != null) {
                    // 로컬 좌표 -> 전체 픽셀 좌표 역산 변환
                    val globalCorners = localCorners.map { PointF(it.x + minX, it.y + minY) }
                    
                    // 가림막 합성
                    val resultBitmap = applyPrivacyMask(processingBitmap, globalCorners)
                    
                    runOnUiThread {
                        updateNativeBackgroundSafe(resultBitmap)
                        nativeGuideView?.visibility = View.GONE
                        safeEvaluate("window.onNativeSuccess()")
                    }
                } else {
                    runOnUiThread { safeEvaluate("window.onNativeError('번호판을 찾지 못했습니다. 가이드 박스를 번호판 크기에 맞게 조절해주세요.')") }
                }

            } catch (e: Exception) { 
                Log.e("CAMERA_DEBUG", "분석 실패", e)
                safeEvaluate("window.onNativeError('분석 중 오류가 발생했습니다.')")
            } finally {
                processingBitmap?.recycle()
                roiBitmap?.recycle()
                isProcessing = false 
            }
        }
    }

    // OpenCV 기반 가림막 합성 방어 코드 (OOM 방지 완벽 해제)
    private fun applyPrivacyMask(source: Bitmap, targetCorners: List<PointF>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val mat = Mat()
        val maskMat = Mat()
        val warpedMask = Mat()
        val alphaMask = Mat()
        val inv = Mat()
        val final = Mat()
        val warpedChannels32F = mutableListOf<Mat>()
        val targetChannels = mutableListOf<Mat>()
        val warpedChannels = mutableListOf<Mat>()
        var srcPts: MatOfPoint2f? = null
        var dstPts: MatOfPoint2f? = null
        var transform: Mat? = null

        try {
            Utils.bitmapToMat(result, mat)
            val maskBmp = Bitmap.createBitmap(600, 150, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.LTGRAY) }
            Utils.bitmapToMat(maskBmp, maskMat)
            maskBmp.recycle()

            srcPts = MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0),
                org.opencv.core.Point(maskMat.cols().toDouble(), 0.0),
                org.opencv.core.Point(maskMat.cols().toDouble(), maskMat.rows().toDouble()),
                org.opencv.core.Point(0.0, maskMat.rows().toDouble())
            )
            
            // 좌표 정렬 보정
            val sortedByX = targetCorners.sortedBy { it.x }
            val left = sortedByX.take(2).sortedBy { it.y }
            val right = sortedByX.takeLast(2).sortedBy { it.y }
            val orderedCorners = listOf(left[0], right[0], right[1], left[1])
            
            dstPts = MatOfPoint2f(*orderedCorners.map { org.opencv.core.Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())

            transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(maskMat, warpedMask, transform, mat.size(), Imgproc.INTER_LINEAR)

            Core.split(warpedMask, warpedChannels)
            if (warpedChannels.size >= 4) {
                warpedChannels[3].copyTo(alphaMask)
                Imgproc.threshold(alphaMask, alphaMask, 1.0, 255.0, Imgproc.THRESH_BINARY)
                Core.bitwise_not(alphaMask, inv)

                warpedMask.convertTo(warpedMask, CvType.CV_32FC4)
                mat.convertTo(mat, CvType.CV_32FC4)
                alphaMask.convertTo(alphaMask, CvType.CV_32FC1, 1.0 / 255.0)
                inv.convertTo(inv, CvType.CV_32FC1, 1.0 / 255.0)

                Core.split(warpedMask, warpedChannels32F)
                Core.split(mat, targetChannels)

                for (i in 0 until 3) {
                    Core.multiply(warpedChannels32F[i], alphaMask, warpedChannels32F[i])
                    Core.multiply(targetChannels[i], inv, targetChannels[i])
                    Core.add(warpedChannels32F[i], targetChannels[i], warpedChannels32F[i])
                }
                if (warpedChannels32F.size > 3 && targetChannels.size > 3) targetChannels[3].copyTo(warpedChannels32F[3])

                Core.merge(warpedChannels32F, final)
                final.convertTo(final, CvType.CV_8UC4)
                Utils.matToBitmap(final, result)
            }
        } catch (e: Exception) {
            Log.e("CAMERA_DEBUG", "Warp 가림막 합성 실패", e)
        } finally {
            mat.release(); maskMat.release(); warpedMask.release(); alphaMask.release()
            inv.release(); final.release()
            warpedChannels.forEach { it.release() }
            warpedChannels32F.forEach { it.release() }
            targetChannels.forEach { it.release() }
            srcPts?.release(); dstPts?.release(); transform?.release()
        }
        return result
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "JiSeKa_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/JiSeKa")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        runOnUiThread { Toast.makeText(this, "💾 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) viewFinder?.post { startCamera() } else finish()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        nativeBackgroundView?.setImageBitmap(null)
        displayedBitmap?.recycle()
        displayedBitmap = null
        
        cameraExecutor.shutdownNow(); analysisExecutor.shutdownNow()
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
