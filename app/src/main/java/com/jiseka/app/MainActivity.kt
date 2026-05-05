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
    // 🧠 AI 연산 및 OpenCV 정밀 마스킹 (ChatGPT 피드백 완벽 반영)
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
        var box2f: MatOfPoint2f? = null

        try {
            // 🔥 1. Binary mask 생성 (어두운 곳 대응을 위해 0.25f로 파격 인하)
            val maskData = ByteArray(inputSize * inputSize)
            var idx = 0
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val conf = outputBuffer.float
                    maskData[idx++] = if (conf > 0.25f) 255.toByte() else 0.toByte()
                }
            }
            maskMat.put(0, 0, maskData)

            // 🔥 2. 노이즈 제거 (ChatGPT 권고: 5x5 커널 ➔ 3x3 커널로 축소하여 디테일 보존)
            Imgproc.medianBlur(maskMat, maskMat, 3)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            Imgproc.morphologyEx(maskMat, maskMat, Imgproc.MORPH_CLOSE, kernel)

            // 🔥 3. Contour 추출
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(
                maskMat, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) {
                Log.d("JiSeKa", "Contour 0개 검출")
                return "[]"
            }

            // 🔥 4. 비율 필터링: 무조건 큰 게 아니라, 번호판처럼 "길쭉한" 녀석들만 남긴다!
            val filteredContours = contours.filter {
                val rect = Imgproc.boundingRect(it)
                // 한국 번호판은 보통 가로가 세로보다 2~6배 길다
                val ratio = rect.width.toFloat() / rect.height.toFloat()
                ratio in 1.5f..7.0f 
            }

            // 필터링 후 남은 것들 중 가장 큰 것을 선택 (다 걸러졌으면 그냥 제일 큰 거 선택)
            val targetContour = if (filteredContours.isNotEmpty()) {
                filteredContours.maxByOrNull { Imgproc.contourArea(it) }!!
            } else {
                contours.maxByOrNull { Imgproc.contourArea(it) }!!
            }

            // 🚨 5. minAreaRect (기울어진 사각형)
            contour2f = MatOfPoint2f(*targetContour.toArray())
            val rect = Imgproc.minAreaRect(contour2f)
            
            box2f = MatOfPoint2f()
            Imgproc.boxPoints(rect, box2f)
            val points = box2f.toArray()

            // 🔥 6. 가장 치명적이었던 꼬임 버그 해결: 확실한 물리적 좌표 정렬 (ChatGPT 방식)
            val sortedByY = points.sortedBy { it.y }
            // 위쪽 2개 점 중 x가 작은게 좌상, 큰게 우상
            val top = sortedByY.take(2).sortedBy { it.x } 
            // 아래쪽 2개 점 중 x가 큰게 우하, 작은게 좌하
            val bottom = sortedByY.takeLast(2).sortedByDescending { it.x } 
            
            // OpenCV.js가 사랑하는 완벽한 순서: [좌상, 우상, 우하, 좌하]
            val orderedPoints = listOf(top[0], top[1], bottom[0], bottom[1])

            // 🔥 7. 원본 크기 좌표로 복원 및 JSON 변환
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
            maskMat.release()
            hierarchy.release()
            contour2f?.release()
            box2f?.release()
        }
    }
}
