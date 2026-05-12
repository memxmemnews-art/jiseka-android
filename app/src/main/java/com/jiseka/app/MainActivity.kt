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

    // 촬영된 원본 비트맵을 저장 시 사용하기 위해 메모리에 유지
    private var lastCapturedBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

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
            // 🚨 핵심 수정: 특정 기기에서 반투명 레이어 잔상이 남는 하드웨어 가속 충돌을 방지합니다.
            // HARDWARE -> SOFTWARE로 변경하여 렌더링 안정성을 확보합니다.
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

            // JavaScript 브릿지 연결
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

        /**
         * 결과 UI 화면에서 카메라 실시간 화면이 비치는 것을 방지하기 위한 제어 함수
         */
        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                viewFinder?.visibility = if (isVisible) View.VISIBLE else View.INVISIBLE
                Log.d("JiSeKa", "Camera Visibility: $isVisible")
            }
        }

        /**
         * 🚨 정밀 합성 및 저장 (0~1 정규화 좌표를 실제 비트맵 크기에 매핑)
         */
        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJson: String) {
            val sourceBitmap = lastCapturedBitmap ?: run {
                Log.e("JiSeKa", "저장할 원본 비트맵이 없습니다.")
                return
            }

            try {
                val json = JSONObject(cornersJson)
                val corners = json.getJSONArray("corners")

                val bmpW = sourceBitmap.width.toFloat()
                val bmpH = sourceBitmap.height.toFloat()

                val pts = mutableListOf<PointF>()
                for (i in 0 until 4) {
                    val p = corners.getJSONObject(i)
                    // 이중 스케일링 방지: 정규화 좌표(0~1) * 실제 비트맵 해상도
                    pts.add(PointF(p.getDouble("x").toFloat() * bmpW, 
                                   p.getDouble("y").toFloat() * bmpH))
                }

                // 합성 가공 후 갤러리에 저장
                val result = processOverlay(sourceBitmap, pts)
                saveBitmapToGallery(result)

            } catch (e: Exception) {
                Log.e("JiSeKa", "저장 중 오류 발생", e)
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
                .setTargetRotation(viewFinder!!.display.rotation)
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

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    // YUV 포맷의 사진을 안전하게 Bitmap으로 변환 (검은 화면 방지)
                    val bitmap = image.toBitmapExt()
                    image.close()

                    lastCapturedBitmap = bitmap
                    val base64Image = bitmapToBase64(bitmap)
                    
                    runOnUiThread {
                        webView?.evaluateJavascript("window.onNativePhotoCaptured('$base64Image')", null)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("JiSeKa", "Capture failed: ${exception.message}", exception)
                }
            }
        )
    }

    // 🚨 핵심: YUV_420_888 포맷을 안전하게 Bitmap으로 디코딩하는 함수
    private fun ImageProxy.toBitmapExt(): Bitmap {
        val nv21 = yuv420ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // 센서 회전에 따른 보정
        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        return nv21
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    private fun processOverlay(source: Bitmap, pts: List<PointF>): Bitmap {
        // 여기에 OpenCV warpPerspective 로직이 들어갑니다.
        // 현재는 안전한 저장을 확인하기 위해 원본 source를 그대로 반환하거나 가공합니다.
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
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
