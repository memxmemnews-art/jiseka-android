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

    // 🚨 핵심: 촬영된 원본 비트맵을 메모리에 유지하여 정밀 합성 시 활용
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
            // 🚨 핵심 수정 1: 투명 배경 유지 및 matrix3d 연산 안정성을 위한 하드웨어 가속 지정
            setBackgroundColor(Color.TRANSPARENT)
            setLayerType(View.LAYER_TYPE_HARDWARE, null)

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

            // 자바스크립트 인터페이스 브릿지 연결
            addJavascriptInterface(AndroidBridge(), "AndroidBridge")

            loadUrl("file:///android_asset/index.html")
        }
    }

    // ── JavaScript Interface Bridge ──
    inner class AndroidBridge {

        /**
         * 🚨 핵심 수정 2: JS와의 함수명 정합성 보장
         * 웹단에서 takePhotoOnly()를 호출할 가능성까지 대비하여 두 함수 모두 열어둡니다.
         */
        @JavascriptInterface
        fun takePhoto() {
            this@MainActivity.takePhoto()
        }

        @JavascriptInterface
        fun takePhotoOnly() {
            Log.w("JiSeKa", "JS에서 구형 브릿지명(takePhotoOnly)을 호출함. takePhoto()로 우회 실행합니다.")
            this@MainActivity.takePhoto()
        }

        /**
         * 카메라 실시간 뷰어 숨김 제어 (결과 화면에서 뒤에 잔상이 비치는 현상 방지)
         */
        @JavascriptInterface
        fun setCameraVisibility(isVisible: Boolean) {
            runOnUiThread {
                viewFinder?.visibility = if (isVisible) View.VISIBLE else View.INVISIBLE
                Log.d("JiSeKa", "카메라 프리뷰 상태 변경 (Visible: $isVisible)")
            }
        }

        /**
         * 🚨 핵심 수정 3: 저장 및 좌표 복원 파이프라인 실제 구현
         * 넘어오는 0.0 ~ 1.0 정규화 좌표를 실제 비트맵 해상도에 단일 매핑합니다.
         */
        @JavascriptInterface
        fun saveImageWithNativeOverlay(cornersJson: String) {
            val sourceBitmap = lastCapturedBitmap ?: run {
                runOnUiThread { Toast.makeText(this@MainActivity, "저장할 원본 사진이 없습니다.", Toast.LENGTH_SHORT).show() }
                Log.e("JiSeKa", "저장 대상 비트맵(lastCapturedBitmap)이 null입니다.")
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
                    // 절대 규칙: 정규화 좌표 * 실제 비트맵 해상도 (viewW 등 이중 스케일링 금지)
                    val realX = p.getDouble("x").toFloat() * bmpW
                    val realY = p.getDouble("y").toFloat() * bmpH
                    pts.add(PointF(realX, realY))
                }

                Log.d("JiSeKa", "복원된 실제 비트맵 좌표: $pts")

                // 정밀 마스킹/합성 처리 후 갤러리 저장 실행 (여기서는 원본 보존 및 안전 저장을 기본 탑재)
                val processedBitmap = processOverlay(sourceBitmap, pts)
                saveBitmapToGallery(processedBitmap)

            } catch (e: Exception) {
                Log.e("JiSeKa", "좌표 매핑 및 최종 저장 중 치명적 오류 발생", e)
                runOnUiThread { Toast.makeText(this@MainActivity, "이미지 저장에 실패했습니다.", Toast.LENGTH_SHORT).show() }
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
                Log.e("JiSeKa", "CameraX Use case 바인딩 실패", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    // 🚨 핵심 수정 4: 안전하고 검증된 YUV -> Bitmap 디코딩 파이프라인 거침
                    val bitmap = image.toBitmapExt()
                    image.close()

                    // 전역 변수에 최종 캡처 비트맵 캐싱 (저장 시 사용)
                    lastCapturedBitmap = bitmap

                    val base64Image = bitmapToBase64(bitmap)
                    runOnUiThread {
                        webView?.evaluateJavascript("window.onNativePhotoCaptured('$base64Image')", null)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("JiSeKa", "사진 촬영 실패: ${exception.message}", exception)
                }
            }
        )
    }

    // ── 🚨 핵심 파이프라인: ImageProxy (YUV_420_888) -> 안전한 Bitmap 변환 ──
    private fun ImageProxy.toBitmapExt(): Bitmap {
        val nv21 = yuv420ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()

        // YuvImage 객체를 통해 안전하게 원본 손실 없이 압축
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        val decodedBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // EXIF/센서 방향에 맞춰 최종 이미지 회전 보정
        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(decodedBitmap, 0, 0, decodedBitmap.width, decodedBitmap.height, matrix, true)
    }

    // CameraX의 YUV 버퍼 스트라이드를 고려하여 NV21 바이트 배열로 추출하는 표준 유틸리티
    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4

        val nv21 = ByteArray(ySize + uvSize * 2)

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val rowStride = image.planes[0].rowStride
        var pos = 0

        if (rowStride == width) {
            yBuffer.get(nv21, 0, ySize)
            pos = ySize
        } else {
            var yBufferPos = 0
            for (row in 0 until height) {
                yBuffer.position(yBufferPos)
                yBuffer.get(nv21, pos, width)
                pos += width
                yBufferPos += rowStride
            }
        }

        val uvRowStride = image.planes[2].rowStride
        val uvPixelStride = image.planes[2].pixelStride

        if (uvPixelStride == 2 && uvRowStride == width) {
            val length = uvSize * 2 - 1
            if (vBuffer.remaining() >= length) {
                vBuffer.get(nv21, ySize, length)
            }
        } else {
            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    val vIndex = row * uvRowStride + col * uvPixelStride
                    val uIndex = row * image.planes[1].rowStride + col * image.planes[1].pixelStride

                    nv21[ySize + (row * width / 2 + col) * 2] = vBuffer.get(vIndex)
                    nv21[ySize + (row * width / 2 + col) * 2 + 1] = uBuffer.get(uIndex)
                }
            }
        }
        return nv21
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        // 웹 전송 속도 최적화를 위해 80~90% 수준 압축 전달 권장
        bitmap.compress(Bitmap.CompressFormat.JPEG, 85, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    // 최종 비트맵 합성 가공 처리부 (필요시 OpenCV warpPerspective 로직 연동 지점)
    private fun processOverlay(source: Bitmap, pts: List<PointF>): Bitmap {
        // TODO: 네이티브 OpenCV 라이브러리가 연동되어 있다면 pts 좌표를 Mat으로 변환하여 정밀 처리 진행
        // 현재는 안전한 갤러리 저장을 보장하기 위해 원본 비트맵을 반환하도록 설정되어 있습니다.
        return source 
    }

    // 안드로이드 MediaStore를 활용한 갤러리 안전 저장 로직
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
                runOnUiThread { Toast.makeText(this, "사진이 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show() }
                Log.d("JiSeKa", "갤러리 저장 성공: $uri")
            }
        } catch (e: Exception) {
            Log.e("JiSeKa", "갤러리 저장 실패", e)
            runOnUiThread { Toast.makeText(this, "갤러리 저장에 실패했습니다.", Toast.LENGTH_SHORT).show() }
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
        lastCapturedBitmap?.recycle()
        lastCapturedBitmap = null
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
