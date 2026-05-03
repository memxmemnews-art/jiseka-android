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

        // 🚨 1. OpenCV 엔진 가동 (상용 앱 퀄리티의 핵심)
        if (OpenCVLoader.initLocal()) {
            Log.d("JiSeKa", "OpenCV 엔진 가동 성공!")
        } else {
            Log.e("JiSeKa", "OpenCV 엔진 가동 실패!")
        }

        // 🚨 2. AI 모델(TFLite) 로드
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

        // 🚨 4. 웹과 안드로이드를 연결하는 브릿지 장착
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
            
            // UI가 멈추지 않도록 무거운 연산은 백그라운드 스레드에서 실행
            Thread {
                try {
                    val decodedByteArray = Base64.decode(base64Str, Base64.DEFAULT)
                    val bitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.size) 
                        ?: throw Exception("비트맵 변환 실패")

                    // AI 마스킹 연산 시작!
                    val cornersJson = runInference(bitmap)
                    
                    // 연산 결과를 웹으로 돌려주기 (UI 스레드에서 실행 필수)
                    runOnUiThread {
                        webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
                    }
                } catch (e: Throwable) { 
                    val safeMsg = e.message?.replace(Regex("[^a-zA-Z0-9가-힣 ]"), "_") ?: "Error"
                    runOnUiThread {
                        webView.evaluateJavascript("alert('AI 마스킹 오류: $safeMsg'); window.receiveAICorners('[]');", null)
                    }
                }
            }.start()
        }
    }

    // ==========================================
    // 🧠 AI 연산 및 OpenCV 정밀 마스킹 (Masterpiece)
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

        val maskMat = org.opencv.core.Mat(inputSize, inputSize, org.opencv.core.CvType.CV_8UC1)
        val hierarchy = org.opencv.core.Mat()
        var contour2f: org.opencv.core.MatOfPoint2f? = null

        try {
            // 🔥 1. Binary mask 생성
            val maskData = ByteArray(inputSize * inputSize)
            var idx = 0
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val conf = outputBuffer.float
                    maskData[idx++] = if (conf > 0.45f) 255.toByte() else 0
                }
            }
            maskMat.put(0, 0, maskData)

            // 🔥 2. 노이즈 제거 + 형태 보정
            org.opencv.imgproc.Imgproc.medianBlur(maskMat, maskMat, 5)

            val kernel = org.opencv.imgproc.Imgproc.getStructuringElement(
                org.opencv.imgproc.Imgproc.MORPH_RECT,
                org.opencv.core.Size(5.0, 5.0)
            )
            org.opencv.imgproc.Imgproc.morphologyEx(
                maskMat, maskMat,
                org.opencv.imgproc.Imgproc.MORPH_CLOSE,
                kernel
            )

            // 🔥 3. Contour 추출
            val contours = ArrayList<org.opencv.core.MatOfPoint>()
            org.opencv.imgproc.Imgproc.findContours(
                maskMat,
                contours,
                hierarchy,
                org.opencv.imgproc.Imgproc.RETR_EXTERNAL,
                org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) return "[]"

            // 🔥 4. 가장 큰 contour 선택
            val largest = contours.maxByOrNull {
                org.opencv.imgproc.Imgproc.contourArea(it)
            } ?: return "[]"

            val area = org.opencv.imgproc.Imgproc.contourArea(largest)
            if (area < 400) return "[]"

            // 🔥 5. 비율 필터 (번호판 형태 확인)
            val rect = org.opencv.imgproc.Imgproc.boundingRect(largest)
            val ratio = rect.width.toFloat() / rect.height.toFloat()
            if (ratio < 2.0f || ratio > 6.0f) return "[]"

            // 🔥 6. convex hull로 contour 안정화 (잔굴곡 제거)
            val hull = org.opencv.core.MatOfInt()
            org.opencv.imgproc.Imgproc.convexHull(largest, hull)

            val hullPoints = mutableListOf<org.opencv.core.Point>()
            for (i in 0 until hull.rows()) {
                val index = hull.get(i, 0)[0].toInt()
                hullPoints.add(largest.toArray()[index])
            }

            contour2f = org.opencv.core.MatOfPoint2f(*hullPoints.toTypedArray())

            // 🔥 7. epsilon 자동 튜닝 (4점이 나올 때까지 루프)
            val peri = org.opencv.imgproc.Imgproc.arcLength(contour2f, true)
            var finalPoints: Array<org.opencv.core.Point>? = null

            for (ratioEps in listOf(0.01, 0.02, 0.03, 0.04)) {
                val approx = org.opencv.core.MatOfPoint2f()
                org.opencv.imgproc.Imgproc.approxPolyDP(
                    contour2f, approx, ratioEps * peri, true
                )

                if (approx.total() == 4L) {
                    finalPoints = approx.toArray()
                    approx.release()
                    break
                }
                approx.release()
            }

            val scaleX = bitmap.width.toFloat() / inputSize
            val scaleY = bitmap.height.toFloat() / inputSize

            // 🔥 8. fallback (끝내 4점을 찾지 못한 경우 기본 사각형 반환)
            if (finalPoints == null) {
                val minX = rect.x * scaleX
                val minY = rect.y * scaleY
                val maxX = (rect.x + rect.width) * scaleX
                val maxY = (rect.y + rect.height) * scaleY

                return "[{\"x\": $minX, \"y\": $minY}, {\"x\": $maxX, \"y\": $minY}, {\"x\": $maxX, \"y\": $maxY}, {\"x\": $minX, \"y\": $maxY}]"
            }

            // 🔥 9. robust point ordering (각도 기반으로 좌상/우상/우하/좌하 정렬)
            val center = finalPoints.reduce { acc, p ->
                org.opencv.core.Point(acc.x + p.x, acc.y + p.y)
            }.let {
                org.opencv.core.Point(it.x / 4, it.y / 4)
            }

            val ordered = finalPoints.sortedBy {
                kotlin.math.atan2(it.y - center.y, it.x - center.x)
            }

            // 🔥 10. 원본 사진 크기로 좌표 변환 후 JSON 변환
            return ordered.joinToString(
                separator = ", ",
                prefix = "[",
                postfix = "]"
            ) {
                "{\"x\": ${it.x * scaleX}, \"y\": ${it.y * scaleY}}"
            }

        } finally {
            // 메모리 누수 방지를 위한 자원 해제 (매우 중요)
            maskMat.release()
            hierarchy.release()
            contour2f?.release()
        }
    }
}
