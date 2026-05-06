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

        // 1. OpenCV 엔진 가동
        if (OpenCVLoader.initLocal()) {
            Log.d("JiSeKa", "OpenCV 엔진 가동 성공!")
        } else {
            Log.e("JiSeKa", "OpenCV 엔진 가동 실패!")
        }

        // 2. AI 모델(TFLite) 로드
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "fairscan-segmentation-model.tflite")
            tflite = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("JiSeKa", "TFLite 모델 로드 실패: ${e.message}")
        }

        // 3. 웹뷰 세팅 (캐시 완벽 차단 및 성능 최적화)
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
        
        // Vercel 웹앱 호출 (항상 최신 버전 강제 로딩)
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
                    // Base64 문자열을 바이너리로 복원
                    val decodedByteArray = Base64.decode(base64Str, Base64.DEFAULT)
                    val bitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.size) 
                        ?: throw Exception("비트맵 변환 실패")

                    // AI 및 OpenCV 연산 실행
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
    // 🧠 AI 연산 및 OpenCV 정밀 마스킹
    // ==========================================
    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) throw Exception("TFLite 누락")

        val inputSize = 256
        
        // 1. 비율 왜곡 방지를 위한 패딩 리사이즈 (정확도 향상의 핵심)
        val resizedBitmap = resizeWithPadding(bitmap, inputSize, inputSize)

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
            // 2. Threshold (그릴 오인식 방지를 위해 0.45f로 상향 튜닝)
            val maskData = ByteArray(inputSize * inputSize)
            var idx = 0
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val conf = outputBuffer.float
                    maskData[idx++] = if (conf > 0.45f) 255.toByte() else 0.toByte()
                }
            }
            maskMat.put(0, 0, maskData)

            // 3. 노이즈 제거 및 형태 보정
            Imgproc.medianBlur(maskMat, maskMat, 3)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            Imgproc.morphologyEx(maskMat, maskMat, Imgproc.MORPH_CLOSE, kernel)

            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(maskMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            if (contours.isEmpty()) return "[]"

            // 번호판 비율 필터링 (가로가 세로보다 1.5~7배 긴 것)
            val filteredContours = contours.filter {
                val rect = Imgproc.boundingRect(it)
                val ratio = rect.width.toFloat() / rect.height.toFloat()
                ratio in 1.5f..7.0f 
            }

            val targetContour = if (filteredContours.isNotEmpty()) {
                filteredContours.maxByOrNull { Imgproc.contourArea(it) }!!
            } else {
                contours.maxByOrNull { Imgproc.contourArea(it) }!!
            }

            // 4. approxPolyDP 정밀 추출 (0.015 계수 적용)
            contour2f = MatOfPoint2f(*targetContour.toArray())
            val peri = Imgproc.arcLength(contour2f, true)
            approx2f = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx2f, 0.015 * peri, true)

            var points = approx2f.toArray()

            // 5. 4점이 안 나올 경우 fallback (minAreaRect로 보정)
            if (points.size != 4) {
                val rect = Imgproc.minAreaRect(contour2f)
                box2f = MatOfPoint2f()
                Imgproc.boxPoints(rect, box2f)
                points = box2f.toArray()
            }

            // 6. 물리적 위치 기반 정렬 (좌상 -> 우상 -> 우하 -> 좌하)
            val sortedByY = points.sortedBy { it.y }
            val top = sortedByY.take(2).sortedBy { it.x } 
            val bottom = sortedByY.takeLast(2).sortedByDescending { it.x } 
            val orderedPoints = listOf(top[0], top[1], bottom[0], bottom[1])

            // 7. 패딩을 감안한 원본 좌표 스케일 복원
            val scale = Math.max(bitmap.width.toFloat() / inputSize, bitmap.height.toFloat() / inputSize)
            val padX = (inputSize - (bitmap.width / scale)) / 2f
            val padY = (inputSize - (bitmap.height / scale)) / 2f

            return orderedPoints.joinToString(separator = ", ", prefix = "[", postfix = "]") {
                "{\"x\": ${(it.x - padX) * scale}, \"y\": ${(it.y - padY) * scale}}"
            }

        } finally {
            // 메모리 누수 방지
            maskMat.release()
            hierarchy.release()
            contour2f?.release()
            approx2f?.release()
            box2f?.release()
        }
    }

    // 비율 왜곡을 방지하면서 검은색 여백을 채우는 리사이즈 함수
    private fun resizeWithPadding(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        val scale = Math.min(width.toFloat() / bitmap.width, height.toFloat() / bitmap.height)
        val scaledWidth = Math.round(scale * bitmap.width)
        val scaledHeight = Math.round(scale * bitmap.height)

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val paddedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        val canvas = android.graphics.Canvas(paddedBitmap)
        canvas.drawColor(android.graphics.Color.BLACK) 
        canvas.drawBitmap(scaledBitmap, (width - scaledWidth) / 2f, (height - scaledHeight) / 2f, null)

        return paddedBitmap
    }
}
