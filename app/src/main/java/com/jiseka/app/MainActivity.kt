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
            Toast.makeText(this, "🚨 [치명적 에러] activity_main.xml 파일에 문제가 있습니다.", Toast.LENGTH_LONG).show()
            return
        }

        // 🚨 방어벽 1: OpenCV가 실패해도 앱을 끄지 않고 경고만 띄웁니다.
        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            Log.e("JiSeKa", "OpenCV 초기화 실패")
            Toast.makeText(this, "🚨 [경고] OpenCV 로드 실패! (PC 에뮬레이터 환경인지 확인하세요)", Toast.LENGTH_LONG).show()
        }

        // 🚨 방어벽 2: XML ID 매칭 실패로 인한 즉사 방지
        viewFinder = findViewById<PreviewView>(R.id.viewFinder)
        webView = findViewById<WebView>(R.id.webView)

        if (viewFinder == null || webView == null) {
            Toast.makeText(this, "🚨 [경고] XML에서 viewFinder나 webView를 찾을 수 없습니다!", Toast.LENGTH_LONG).show()
            return
        }

        viewFinder?.scaleType = PreviewView.ScaleType.FILL_CENTER
        cameraExecutor = Executors.newSingleThreadExecutor()

        webView?.apply {
            clearCache(true)
            setLayerType(View.LAYER_TYPE_HARDWARE, null)
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
            
            // 실제 웹사이트 로딩
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
                        processPlateAnalysis(safeBitmap, RectF(left, top, right, bottom))
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

        // ... (이하 saveImageToGallery 코드는 기존과 100% 동일하므로 생략 없이 기존 코드 그대로 유지하시면 됩니다) ...
        @JavascriptInterface
        fun saveImageToGallery(base64Data: String) {
            Thread {
                var bitmap: Bitmap? = null
                try {
                    val base64Image = base64Data.substringAfter(",")
                    val imageBytes = Base64.decode(base64Image, Base64.DEFAULT)
                    bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size) ?: throw IllegalStateException("Bitmap decode failed")

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
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream)
                        stream.flush()
                    }

                    legacyFile?.let { file ->
                        android.media.MediaScannerConnection.scanFile(this@MainActivity, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)
                    }
                    showToast("사진이 갤러리에 저장되었습니다.")
                } catch (e: Exception) {
                    showToast("사진 저장 실패")
                } finally {
                    bitmap?.let { if (!it.isRecycled) it.recycle() }
                }
            }.start()
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
            
            // 🚨 방어벽 3: 에뮬레이터 카메라 바인딩 에러 방지
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
                Toast.makeText(this, "🚨 [경고] 후면 카메라를 바인딩할 수 없습니다.", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ... (이하 capturePhoto, processPlateAnalysis, downscaleBitmapIfNeeded 등 나머지 함수는 오류가 없으므로 이전 코드 그대로 복사하시면 됩니다) ...
    // 응답 길이 제한을 방지하기 위해 중복 코드를 생략했습니다. 방어 로직이 모두 들어간 위쪽 절반만 확실하게 덮어써 주시면 됩니다.
