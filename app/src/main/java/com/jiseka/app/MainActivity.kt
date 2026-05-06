package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private var tflite: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 🚨 1. OpenCV 엔진 가동
        if (OpenCVLoader.initLocal()) {
            Log.d("JiSeKa", "OpenCV 엔진 가동 성공!")
        } else {
            Log.e("JiSeKa", "OpenCV 엔진 가동 실패!")
        }

        // 🚨 2. AI 모델 로드
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "fairscan-segmentation-model.tflite")
            tflite = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("JiSeKa", "TFLite 모델 로드 실패: ${e.message}")
        }

        // 🚨 3. 웹뷰 세팅 (캐시 완벽 차단)
        webView = findViewById(R.id.webView)
        webView.settings.apply {
            javaScriptEnabled = true
            mediaPlaybackRequiresUserGesture = false
            domStorageEnabled = true
            cacheMode = WebSettings.LOAD_NO_CACHE
        }
        
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                request.grant(request.resources)
            }
        }

        webView.addJavascriptInterface(WebAppInterface(), "JiSeKaNative")
        val cacheBusterUrl = "https://ziseka-app.vercel.app?refresh=" + System.currentTimeMillis()
        webView.loadUrl(cacheBusterUrl)
    }

    // ==========================================
    // 🌐 웹 ↔ 안드로이드 스트리밍 통신 브릿지
    // ==========================================
    inner class WebAppInterface {
        private var imageBuffer = StringBuilder()

        @JavascriptInterface
        fun startImageStream() {
            imageBuffer.clear()
        }

        @JavascriptInterface
        fun appendImageChunk(chunk: String) {
            imageBuffer.append(chunk)
        }

        @JavascriptInterface
        fun finishImageStream() {
            val base64Str = imageBuffer.toString()
            Thread {
                try {
                    val decodedByteArray = Base64.decode(base64Str, Base64.DEFAULT)
                    val bitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.size) 
                        ?: throw Exception("비트맵 변환 실패")

                    val cornersJson = runInference(bitmap)

                    runOnUiThread {
                        webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
                    }
                } catch (e: Throwable) { 
                    val safeMsg = e.message?.replace(Regex("[^a-zA-Z0-9가-힣 ]"), "_") ?: "Error"
                    runOnUiThread {
                        webView.evaluateJavascript("alert('AI 마스킹 오류: ${safeMsg}'); window.receiveAICorners('[]');", null)
                    }
                }
            }.start()
        }
    }

    // ==========================================
    // 🧠 AI 연산 및 OpenCV 정밀 마스킹 (궁극의 튜닝 버전)
    // ==========================================
    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) throw Exception("TFLite 누락")

        val inputSize = 256
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val v = intValues[pixel++]
                inputBuffer.putFloat(((v shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((v shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((v and 0xFF) / 255f)
            }
        }

        val outputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize)
        outputBuffer.order(ByteOrder.nativeOrder())

        tflite?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val maskMat = Mat(inputSize, inputSize, CvType.CV_8UC1)
        val hierarchy = Mat()
        var contour2f: MatOfPoint2f? = null
        var approx2f: MatOfPoint2f? = null
        var box2f: MatOfPoint2f? = null

        try {
            // 🔥 1. Binary mask 생성 (어두운 주차장 대응: 0.25f)
            val maskData = ByteArray(inputSize * inputSize)
            var idx = 0
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val conf = outputBuffer.float
                    maskData[idx++] = if (conf > 0.25f) 255.toByte() else 0.toByte()
                }
            }
            maskMat.put(0, 0, maskData)

            // 🔥 2. 노이즈 제거 (디테일 뭉개짐 방지: 3x3 커널)
            Imgproc.medianBlur(maskMat, maskMat, 3)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            Imgproc.morphologyEx(maskMat, maskMat, Imgproc.MORPH_CLOSE, kernel)

            // 🔥 3. Contour 추출
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(
                maskMat, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) return "[]"

            // 🔥 4. 비율 필터링: 가로가 세로보다 1.5배~7배 긴 "번호판 모양"만 통과!
            val filteredContours = contours.filter {
                val rect = Imgproc.boundingRect(it)
                val ratio = rect.width.toFloat() / rect.height.toFloat()
                ratio in 1.5f..7.0f 
            }

            // 필터링 통과한 것 중 가장 큰 영역 선택 (없으면 전체 중 가장 큰 것)
            val targetContour = if (filteredContours.isNotEmpty()) {
                filteredContours.maxByOrNull { Imgproc.contourArea(it) }!!
            } else {
                contours.maxByOrNull { Imgproc.contourArea(it) }!!
            }

            // 🔥 5. approxPolyDP로 예리하게 4점 깎아내기 (가장 정확한 윤곽선)
            contour2f = MatOfPoint2f(*targetContour.toArray())
            val peri = Imgproc.arcLength(contour2f, true)
            approx2f = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx2f, 0.02 * peri, true)

            var points = approx2f.toArray()

            // 🔥 6. 4점이 안 나올 경우의 최후의 보루: minAreaRect 자동 방어
            if (points.size != 4) {
                val rect = Imgproc.minAreaRect(contour2f)
                box2f = MatOfPoint2f()
                Imgproc.boxPoints(rect, box2f)
                points = box2f.toArray()
            }

            // 🔥 7. 꼬임 원천 차단: 물리적 4점 정렬 (atan2 제거, 순수 위치 기반)
            val sortedByY = points.sortedBy { it.y }
            // 상단 2점: X가 작은 게 좌상, 큰 게 우상
            val top = sortedByY.take(2).sortedBy { it.x } 
            // 하단 2점: X가 큰 게 우하, 작은 게 좌하
            val bottom = sortedByY.takeLast(2).sortedByDescending { it.x } 
            
            // OpenCV.js가 요구하는 완벽한 순서: [좌상, 우상, 우하, 좌하]
            val orderedPoints = listOf(top[0], top[1], bottom[0], bottom[1])

            // 🔥 8. 원본 해상도 스케일로 복원 및 JSON 직렬화
            val scaleX = bitmap.width.toFloat() / inputSize
            val scaleY = bitmap.height.toFloat() / inputSize

            return orderedPoints.joinToString(
                separator = ", ",
                prefix = "[",
                postfix = "]"
            ) {
                "{\"x\": ${it.x * scaleX}, \"y\": ${it.y * scaleY}}"
            }

        } finally {
            // 메모리 누수 완벽 차단
            maskMat.release()
            hierarchy.release()
            contour2f?.release()
            approx2f?.release()
            box2f?.release()
        }
    }
}
