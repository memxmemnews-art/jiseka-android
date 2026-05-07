package com.jiseka.app

import android.annotation.SuppressLint
import android.graphics.Color
import android.graphics.PointF
import android.os.Bundle
import android.util.Log
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var viewFinder: PreviewView
    private lateinit var cameraExecutor: ExecutorService

    // Throttling: 초당 약 5~6회 OCR 및 JS 호출
    private var lastAnalysisTime = 0L
    private val analysisInterval = 180L

    // EMA Smoothing 및 Persistence
    private var smoothedPoints = Array(4) { PointF(0f, 0f) }
    private val alpha = 0.25f 
    private var isFirstDetection = true
    private var lastDetectionTime = 0L
    private val persistenceTimeout = 600L

    // ML Kit 인식기
    private val recognizer by lazy {
        TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 🚨 이 부분이 추가되어야 에러가 발생하지 않습니다!
        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

        webView.setBackgroundColor(Color.TRANSPARENT)
        webView.settings.javaScriptEnabled = true
        // 🚨 Vercel 주소가 맞는지 꼭 확인하세요!
        webView.loadUrl("https://your-vercel-app-url.vercel.app") 

        cameraExecutor = Executors.newSingleThreadExecutor()
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastAnalysisTime >= analysisInterval) {
                            processImageProxy(imageProxy)
                            lastAnalysisTime = currentTime
                        } else {
                            imageProxy.close() 
                        }
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processImageProxy(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val rotation = imageProxy.imageInfo.rotationDegrees
        val image = InputImage.fromMediaImage(mediaImage, rotation)

        recognizer.process(image)
            .addOnSuccessListener { result ->
                var bestBlock: Text.TextBlock? = null
                var bestScore = Float.MAX_VALUE

                val isPortrait = rotation == 90 || rotation == 270
                val imgW = if (isPortrait) imageProxy.height else imageProxy.width
                val imgH = if (isPortrait) imageProxy.width else imageProxy.height
                val centerX = imgW / 2f
                val centerY = imgH / 2f

                for (block in result.textBlocks) {
                    val box = block.boundingBox ?: continue
                    
                    val blockCenterX = box.exactCenterX()
                    val blockCenterY = box.exactCenterY()
                    val distToCenter = abs(centerX - blockCenterX) + abs(centerY - blockCenterY)
                    
                    val area = box.width() * box.height()
                    val score = distToCenter - (area * 0.1f) 

                    if (score < bestScore) {
                        bestScore = score
                        bestBlock = block
                    }
                }

                val points = bestBlock?.cornerPoints

                if (points != null && points.size == 4) {
                    lastDetectionTime = System.currentTimeMillis()

                    val viewWidth = viewFinder.width.toFloat()
                    val viewHeight = viewFinder.height.toFloat()

                    val scale = maxOf(viewWidth / imgW, viewHeight / imgH)
                    val offsetX = (viewWidth - imgW * scale) / 2f
                    val offsetY = (viewHeight - imgH * scale) / 2f
                    val density = resources.displayMetrics.density

                    val currentPoints = Array(4) { i ->
                        PointF(
                            ((points[i].x * scale) + offsetX) / density,
                            ((points[i].y * scale) + offsetY) / density
                        )
                    }

                    if (isFirstDetection) {
                        for (i in 0..3) {
                            smoothedPoints[i] = PointF(currentPoints[i].x, currentPoints[i].y)
                        }
                        isFirstDetection = false
                    } else {
                        for (i in 0..3) {
                            smoothedPoints[i].x = (currentPoints[i].x * alpha) + (smoothedPoints[i].x * (1 - alpha))
                            smoothedPoints[i].y = (currentPoints[i].y * alpha) + (smoothedPoints[i].y * (1 - alpha))
                        }
                    }

                    pushToWebView(smoothedPoints)

                } else {
                    if (System.currentTimeMillis() - lastDetectionTime > persistenceTimeout) {
                        isFirstDetection = true 
                        pushToWebView(null)
                    }
                }
            }
            .addOnFailureListener {
                imageProxy.close() 
            }
            .addOnCompleteListener {
                imageProxy.close() 
            }
    }

    private fun pushToWebView(points: Array<PointF>?) {
        runOnUiThread {
            if (points == null) {
                webView.evaluateJavascript("javascript:if(window.drawCarbonMask) window.drawCarbonMask([]);", null)
            } else {
                val jsonCoords = """[
                    {"x": ${points[0].x}, "y": ${points[0].y}},
                    {"x": ${points[1].x}, "y": ${points[1].y}},
                    {"x": ${points[2].x}, "y": ${points[2].y}},
                    {"x": ${points[3].x}, "y": ${points[3].y}}
                ]""".trimIndent()
                
                webView.evaluateJavascript("javascript:if(window.drawCarbonMask) window.drawCarbonMask($jsonCoords);", null)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        recognizer.close() 
    }
}
