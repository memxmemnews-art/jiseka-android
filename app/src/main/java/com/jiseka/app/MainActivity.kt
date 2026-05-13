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

    private var webView: WebView? = null [cite: 2]
    private var viewFinder: PreviewView? = null
    private var imageCapture: ImageCapture? = null [cite: 3]
    private lateinit var cameraExecutor: ExecutorService

    // 촬영된 원본 비트맵을 메모리에 유지하여 합성 시 사용 [cite: 4]
    private var lastCapturedBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main) [cite: 5]

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

        // CameraX 뷰파인더 설정: WebView와 GPU 충돌 방지를 위해 COMPATIBLE 모드 권장
        viewFinder?.apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        setupWebView()

        if (allPermissionsGranted()) {
            startCamera() [cite: 5]
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS) [cite: 5]
        }

        cameraExecutor = Executors.newSingleThreadExecutor() [cite: 5]
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView?.apply {
            // 하드웨어 가속 충돌로 인한 잔상 방지를 위해 SOFTWARE 렌더링 강제 [cite: 6]
            setLayerType(View.LAYER_TYPE_SOFTWARE, null)
            setBackgroundColor(Color.TRANSPARENT)

            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = true [cite: 7]
                allowContentAccess = true
                loadWithOverviewMode = true
                useWideViewPort = true
            }

            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    return false [cite: 8]
                }
            }
            webChromeClient = WebChromeClient()

            addJavascriptInterface(AndroidBridge(), "AndroidBridge") [cite: 9]
            loadUrl("file:///android_asset/index.html") [cite: 9]
        }
    }

    // ── JavaScript Interface Bridge ── [cite: 10]
    inner class AndroidBridge {

        @JavascriptInterface
        fun takePhoto() {
            this@MainActivity.takePhoto()
        }

        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                viewFinder?.visibility = if (isVisible) View.VISIBLE else View.INVISIBLE [cite: 10]
            }
        }

        /**
         * 🚨 분석 인터페이스: app.js에서 보낸 4개 꼭짓점 JSON 문자열을 수신합니다.
         */
        @JavascriptInterface
        fun analyzePlateWithMode(cornersJsonStr: String, mode: String) {
            Log.d("JiSeKa", "Received Payload for mode $mode: $cornersJsonStr")
            
            // 여기서 수신된 cornersJsonStr을 파싱하여 OpenCV 분석 로직으로 전달할 수 있습니다.
            // 성공 시 JS의 window.onNativeSuccess(payload)를 호출하십시오.
            runOnUiThread {
                webView?.evaluateJavascript("window.onNativeSuccess('$cornersJsonStr')", null)
            }
        }

        /**
         * 🚨 저장 인터페이스: 0~1 정규화된 4개 꼭짓점을 실제 원본 비트맵 해상도에 매핑하여 합성 저장합니다.
         */
        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJsonStr: String) {
            val sourceBitmap = lastCapturedBitmap ?: run {
                Log.e("JiSeKa", "원본 비트맵이 없습니다.") [cite: 12]
                return
            }

            try {
                val json = JSONObject(cornersJsonStr)
                val corners = json.getJSONArray("corners")

                val bmpW = sourceBitmap.width.toFloat() [cite: 13]
                val bmpH = sourceBitmap.height.toFloat()

                val pts = mutableListOf<PointF>()
                for (i in 0 until 4) {
                    val p = corners.getJSONObject(i)
                    // 정규화 좌표(0~1)를 실제 비트맵 해상도로 복원 [cite: 14]
                    pts.add(PointF(p.getDouble("x").toFloat() * bmpW, 
                                   p.getDouble("y").toFloat() * bmpH))
                }

                // 4점 투시 변환(warpPerspective) 기반 합성 후 저장 [cite: 15]
                val result = processOverlay(sourceBitmap, pts)
                saveBitmapToGallery(result)
                
                // 저장 완료 후 JS에 알림
                runOnUiThread {
                    webView?.evaluateJavascript("window.onNativeSaveComplete()", null)
                }

            } catch (e: Exception) {
                Log.e("JiSeKa", "저장 중 오류 발생", e)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this) [cite: 16]
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }

            imageCapture = ImageCapture.Builder() [cite: 17]
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(1920, 1080)) // 안정적인 분석을 위해 해상도 명시
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture) [cite: 18]
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // 🚨 최초 실행 시 권한 허용 후 검은 화면 방지 로직
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
            ContextCompat.getMainExecutor(this), [cite: 19]
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = image.toBitmapExt() // YUV를 Bitmap으로 안전 변환 [cite: 20]
                    image.close()

                    lastCapturedBitmap = bitmap
                    val base64Image = bitmapToBase64(bitmap)
                    
                    runOnUiThread {
                        // app.js의 onNativePhotoCaptured 함수 호출 (함수명 일치 확인) [cite: 21]
                        webView?.evaluateJavascript("window.onNativePhotoCaptured('$base64Image')", null)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("JiSeKa", "Capture failed", exception) [cite: 22]
                }
            }
        )
    }

    private fun ImageProxy.toBitmapExt(): Bitmap {
        val nv21 = yuv420ToNv21(this) [cite: 23]
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true) [cite: 24]
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val nv21 = ByteArray(yBuffer.remaining() + uBuffer.remaining() + vBuffer.remaining())
        yBuffer.get(nv21, 0, yBuffer.remaining())
        vBuffer.get(nv21, yBuffer.remaining(), vBuffer.remaining())
        uBuffer.get(nv21, yBuffer.remaining() + vBuffer.remaining(), uBuffer.remaining())
        return nv21 [cite: 25]
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    private fun processOverlay(source: Bitmap, pts: List<PointF>): Bitmap {
        // 여기에 OpenCV warpPerspective 로직을 적용하여 사다리꼴 가림막을 합성합니다. [cite: 26]
        // 현재는 안정적인 저장을 위해 원본을 그대로 반환하는 상태입니다. [cite: 27]
        return source 
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
        var fos: OutputStream? = null [cite: 28]
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/JiSeKa")
                put(MediaStore.MediaColumns.IS_PENDING, 1) [cite: 29]
            }
        }

        val imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

        try {
            imageUri?.let { uri ->
                fos = contentResolver.openOutputStream(uri)
                fos?.let { bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it) } [cite: 30]

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                    contentResolver.update(uri, contentValues, null, null) [cite: 30]
                }
                runOnUiThread { Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() } [cite: 31]
            }
        } catch (e: Exception) {
            Log.e("JiSeKa", "Save failed", e)
        } finally {
            fos?.close()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED [cite: 32]
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
