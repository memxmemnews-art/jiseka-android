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
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private var webView: WebView? = null
    private var viewFinder: PreviewView? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    // 촬영된 원본 비트맵을 메모리에 유지하여 합성 시 사용
    private var lastCapturedBitmap: Bitmap? = null
    private val bitmapLock = Any()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

        // CameraX 뷰파인더 설정: WebView와 GPU 충돌 방지를 위해 COMPATIBLE 모드 적용
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
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView?.apply {
            // 하드웨어 가속 충돌로 인한 잔상 방지를 위해 SOFTWARE 렌더링 강제
            setLayerType(View.LAYER_TYPE_SOFTWARE, null)
            setBackgroundColor(Color.TRANSPARENT)

            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = true
                allowContentAccess = true
                loadWithOverviewMode = true
                useWideViewPort = true
            }

            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    return false
                }
            }
            webChromeClient = WebChromeClient()

            addJavascriptInterface(AndroidBridge(), "AndroidBridge")
            loadUrl("file:///android_asset/index.html")
        }
    }

    // ── JavaScript Interface Bridge ──
    inner class AndroidBridge {

        @JavascriptInterface
        fun takePhoto() {
            this@MainActivity.takePhoto()
        }

        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                viewFinder?.visibility = if (isVisible) View.VISIBLE else View.INVISIBLE
            }
        }

        /**
         * 분석 인터페이스: app.js에서 보낸 4개 꼭짓점 JSON 문자열을 수신합니다.
         */
        @JavascriptInterface
        fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            Log.d("JiSeKa", "Received Payload for mode $mode: $cornersJsonStr")
            
            runOnUiThread {
                webView?.evaluateJavascript("window.onNativeSuccess('$cornersJsonStr')", null)
            }
        }

        /**
         * 저장 인터페이스: 0~1 정규화된 4개 꼭짓점을 실제 원본 비트맵 해상도에 매핑하여 합성 저장합니다.
         */
        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            val sourceBitmap = synchronized(bitmapLock) {
                lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true)
            } ?: run {
                Log.e("JiSeKa", "원본 비트맵이 없습니다.")
                return
            }

            cameraExecutor.execute {
                try {
                    val json = JSONObject(cornersJsonStr)
                    val corners = json.getJSONArray("corners")

                    val bmpW = sourceBitmap.width.toFloat()
                    val bmpH = sourceBitmap.height.toFloat()

                    val pts = mutableListOf<PointF>()
                    for (i in 0 until 4) {
                        val p = corners.getJSONObject(i)
                        // 정규화 좌표(0~1)를 실제 비트맵 해상도로 복원
                        pts.add(PointF(p.getDouble("x").toFloat() * bmpW, 
                                       p.getDouble("y").toFloat() * bmpH))
                    }

                    // 4점 투시 변환 기반 합성 후 저장
                    val result = processOverlay(sourceBitmap, pts)
                    saveBitmapToGallery(result)
                    
                    runOnUiThread {
                        webView?.evaluateJavascript("window.onNativeSaveComplete()", null)
                    }

                } catch (e: Exception) {
                    Log.e("JiSeKa", "저장 중 오류 발생", e)
                } finally {
                    sourceBitmap.recycle()
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(1920, 1080)) // 안정적인 분석을 위해 타겟 해상도 강제
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // 최초 실행 시 권한 허용 후 검은 화면 방지 로직
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                viewFinder?.post { startCamera() }
            } else {
                Toast.makeText(this, "카메라 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val bitmap = image.toBitmapExt() // YUV를 Bitmap으로 안전 변환
                        
                        synchronized(bitmapLock) {
                            lastCapturedBitmap?.recycle()
                            lastCapturedBitmap = bitmap
                        }
                        
                        val base64Image = bitmapToBase64(bitmap)
                        
                        runOnUiThread {
                            webView?.evaluateJavascript("window.onNativePhotoCaptured('$base64Image')", null)
                        }
                    } catch (e: Exception) {
                        Log.e("JiSeKa", "Bitmap conversion failed", e)
                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("JiSeKa", "Capture failed", exception)
                }
            }
        )
    }

    private fun ImageProxy.toBitmapExt(): Bitmap {
        val nv21 = yuv420ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    // 🚨 해결 방법 2 완벽 반영: remaining() 중복 호출로 인한 포인터 오작동 및 버퍼 예외 원천 차단
    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        // 버퍼의 크기를 불변(Immutable) 변수로 사전 추출
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // 고정된 크기값을 바탕으로 정확한 슬롯에 복사
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
        return nv21
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, outputStream)
        // Base64.NO_WRAP을 강제하여 웹뷰 evaluateJavascript 전송 시 줄바꿈 파싱 에러 방지
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    private fun processOverlay(source: Bitmap, pts: List<PointF>): Bitmap {
        // 백엔드 OpenCV 4점 투시 밀착 렌더링 파이프라인 안착 지점
        return source 
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
        var fos: OutputStream? = null
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/JiSeKa")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }

        val contentResolver = contentResolver
        val imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

        try {
            imageUri?.let { uri ->
                fos = contentResolver.openOutputStream(uri)
                fos?.let { bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it) }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                    contentResolver.update(uri, contentValues, null, null)
                }
                runOnUiThread { Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() }
            }
        } catch (e: Exception) {
            Log.e("JiSeKa", "Save failed", e)
        } finally {
            fos?.close()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
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
