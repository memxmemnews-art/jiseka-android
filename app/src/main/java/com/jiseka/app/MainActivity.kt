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
        }

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
                        webView.evaluateJavascript("alert('AI 연산 오류: $safeMsg'); window.receiveAICorners('[]');", null)
                    }
                }
            }.start()
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) throw Exception("TFLite 모델 누락")

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

        // OpenCV Mat 객체들 선언 (메모리 해제를 위해 밖으로 분리)
        val maskMat = Mat(inputSize, inputSize, CvType.CV_8UC1)
        val hierarchy = Mat()
        var contour2f: MatOfPoint2f? = null
        var box2f: MatOfPoint2f? = null

        try {
            // 🔥 1. 마스크 → Mat 변환
            val maskData = ByteArray(inputSize * inputSize)
            var idx = 0
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val conf = outputBuffer.float
                    maskData[idx++] = if (conf > 0.45f) 255.toByte() else 0.toByte()
                }
            }
            maskMat.put(0, 0, maskData)

            // 🔥 2. 노이즈 제거 (ChatGPT 추천 반영)
            Imgproc.medianBlur(maskMat, maskMat, 5)

            // 🔥 3. Contour 추출
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(
                maskMat,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) return "[]"

            // 🔥 4. 가장 큰 contour 선택
            val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return "[]"
            contour2f = MatOfPoint2f(*largestContour.toArray())

            // 🔥 5. minAreaRect (ChatGPT 핵심 로직 반영)
            val rect = Imgproc.minAreaRect(contour2f)
            box2f = MatOfPoint2f()
            Imgproc.boxPoints(rect, box2f)
            val points = box2f.toArray()

            // 🚨 제미나이 보완: 텍스처 꼬임 방지를 위한 기하학적 정렬 (atan2)
            val center = points.reduce { acc, p -> Point(acc.x + p.x, acc.y + p.y) }
                .let { Point(it.x / 4, it.y / 4) }

            val ordered = points.sortedBy {
                kotlin.math.atan2(it.y - center.y, it.x - center.x)
            }

            // 🔥 6. 원본 크기로 스케일 복원 및 JSON 직렬화
            val scaleX = bitmap.width.toFloat() / inputSize
            val scaleY = bitmap.height.toFloat() / inputSize

            return ordered.joinToString(
                separator = ", ",
                prefix = "[",
                postfix = "]"
            ) {
                "{\"x\": ${it.x * scaleX}, \"y\": ${it.y * scaleY}}"
            }

        } finally {
            // 🚨 제미나이 보완: 네이티브 메모리 누수 완벽 방어 (매우 중요)
            maskMat.release()
            hierarchy.release()
            contour2f?.release()
            box2f?.release()
        }
    }
}
