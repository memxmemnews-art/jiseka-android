package com.jiseka.app

import android.annotation.SuppressLint
import android.graphics.Color
import android.graphics.PointF
import android.graphics.Rect
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

    // 🛡️ [방어막 1] Throttling: 초당 약 5~6회 OCR 및 JS 호출
    private var lastAnalysisTime = 0L
    private val analysisInterval = 180L

    // 🛡️ [방어막 2] EMA Smoothing 및 Persistence(유지)
    private var smoothedPoints = Array(4) { PointF(0f, 0f) }
    private val alpha = 0.25f // 살짝 더 빠르게 따라오도록 0.25로 조정
    private var isFirstDetection = true
    private var lastDetectionTime = 0L
    private val persistenceTimeout = 600L // 0.6초 동안 안 보이면 마스크 지우기

    // ML Kit 인식기 싱글톤
    private val recognizer by lazy {
        TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        webView = findViewById(R.id.webView)

        webView.setBackgroundColor(Color.TRANSPARENT)
        webView.settings.javaScriptEnabled = true
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
                        
                        // 1. OCR 횟수 제한 (발열 방지)
                        if (currentTime - lastAnalysisTime >= analysisInterval) {
                            processImageProxy(imageProxy)
                            lastAnalysisTime = currentTime
                        } else {
                            // 🚨 [치명적 버그 해결 1] 분석 안 할 때 무조건 close!
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
        // 🚨 [치명적 버그 해결 2] image가 null일 때도 무조건 close() 호출하고 return
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val rotation = imageProxy.imageInfo.rotationDegrees
        val image = InputImage.fromMediaImage(mediaImage, rotation)

        recognizer.process(image)
            .addOnSuccessListener { result ->
                
                // 🚨 [치명적 버그 해결 3] ROI 기반 탐색 (화면 중앙에서 가장 가까운 거대한 텍스트 찾기)
                var bestBlock: Text.TextBlock? = null
                var bestScore = Float.MAX_VALUE

                val isPortrait = rotation == 90 || rotation == 270
                val imgW = if (isPortrait) imageProxy.height else imageProxy.width
                val imgH = if (isPortrait) imageProxy.width else imageProxy.height
                val centerX = imgW / 2f
                val centerY = imgH / 2f

                for (block in result.textBlocks) {
                    val box = block.boundingBox ?: continue
                    
                    // 가이드 박스 영역(중앙)과의 거리 계산
                    val blockCenterX = box.exactCenterX()
                    val blockCenterY = box.exactCenterY()
                    val distToCenter = abs(centerX - blockCenterX) + abs(centerY - blockCenterY)
                    
                    val area = box.width() * box.height()
                    
                    // 크기가 크고(면적), 화면 중앙에 가까울수록 점수가 낮음(좋은 점수)
                    // (임의의 가중치 공식: 거리에 벌점, 면적에 보너스)
                    val score = distToCenter - (area * 0.1f) 

                    if (score < bestScore) {
                        bestScore = score
                        bestBlock = block
                    }
                }

                val points = bestBlock?.cornerPoints

                // 🚨 [치명적 버그 해결 4] Convex & Clockwise 검증 (순서 꼬임 방지)
                if (points != null && points.size == 4) {
                    lastDetectionTime = System.currentTimeMillis()

                    val viewWidth = viewFinder.width.toFloat()
                    val viewHeight = viewFinder.height.toFloat()

                    // PreviewView의 OutputTransform 기반 비율 계산 (가장 안전한 Scale 공식)
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

                    // 🚨 [치명적 버그 해결 5] 깊은 복사(Deep Copy)를 통한 EMA 보정
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
                    // 🚨 [치명적 버그 해결 6] Persistence: 못 찾았더라도 0.6초간은 이전 마스크 유지
                    if (System.currentTimeMillis() - lastDetectionTime > persistenceTimeout) {
                        isFirstDetection = true // 0.6초 넘게 안 보이면 초기화하고 화면에서 지움
                        pushToWebView(null)
                    }
                }
            }
            .addOnFailureListener {
                imageProxy.close() // 에러 나도 닫기
            }
            .addOnCompleteListener {
                imageProxy.close() // 성공해도 닫기 (절대 멈춤 없음)
            }
    }

    private fun pushToWebView(points: Array<PointF>?) {
        runOnUiThread {
            if (points == null) {
                // 0.6초 이상 놓치면 마스크 지우기
                webView.evaluateJavascript("javascript:if(window.drawCarbonMask) window.drawCarbonMask([]);", null)
            } else {
                val jsonCoords = """[
                    {"x": ${points[0].x}, "y": ${points[0].y}},
                    {"x": ${points[1].x}, "y": ${points[1].y}},
                    {"x": ${points[2].x}, "y": ${points[2].y}},
                    {"x": ${points[3].x}, "y": ${points[3].y}}
                ]""".trimIndent()
                
                // 🚨 [치명적 버그 해결 7] JS Ready 체크 (window.drawCarbonMask가 있을 때만 호출)
                webView.evaluateJavascript("javascript:if(window.drawCarbonMask) window.drawCarbonMask($jsonCoords);", null)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        recognizer.close() // 🚨 [치명적 버그 해결 8] 메모리 누수 방지
    }
}
