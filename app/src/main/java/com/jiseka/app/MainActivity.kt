package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
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

        // 1. AI 모델 로드 (fairscan-segmentation-model.tflite)
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "fairscan-segmentation-model.tflite")
            tflite = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        webView = findViewById(R.id.webView)
        webView.settings.javaScriptEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false
        
        // 카메라 권한 승인 로직 (이중 보안문 해제)
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                request.grant(request.resources)
            }
        }

        webView.addJavascriptInterface(WebAppInterface(), "JiSeKaNative")
        webView.loadUrl("https://ziseka-app.vercel.app")
    }

   inner class WebAppInterface {
        @JavascriptInterface
        fun sendImageData(base64Str: String) {
            try {
                // 웹에서 온 Base64 이미지를 비트맵으로 변환
                val pureBase64 = base64Str.substringAfter(",")
                val decodedString = Base64.decode(pureBase64, Base64.DEFAULT)
                val bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size)

                // AI 추론 실행
                val cornersJson = runInference(bitmap)

                // 결과 웹으로 쏘기
                runOnUiThread {
                    webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
                }
            } catch (e: Exception) {
                // 🚨 AI 분석 중 에러가 발생하면 폰 화면에 팝업 띄우기
                val errorMsg = e.message?.replace("'", "\\'") ?: "Unknown Error"
                runOnUiThread {
                    webView.evaluateJavascript("alert('AI 에러 발생: $errorMsg');", null)
                    webView.evaluateJavascript("window.receiveAICorners('[]');", null) // UI 원상복구
                }
            }
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) return "[]"

        val inputSize = 256
        
        // 1. 입력 이미지 전처리 (256x256 리사이징)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val valInt = intValues[pixel++]
                // RGB 값을 0.0 ~ 1.0 사이의 Float로 변환하여 주입
                inputBuffer.putFloat(((valInt shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((valInt shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((valInt and 0xFF) / 255.0f)
            }
        }

        // 2. 출력 배열 준비 (1 x 256 x 256 x 1)
        val outputArray = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(1) } } }
        
        // 3. AI 모델 실행
        tflite?.run(inputBuffer, outputArray)

        // 4. 후처리: 마스크(도면)에서 번호판 위치(Bounding Box) 찾기
        var minX = inputSize
        var minY = inputSize
        var maxX = -1
        var maxY = -1

        val threshold = 0.5f // 50% 이상의 확신이 있는 픽셀만 번호판으로 간주

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val confidence = outputArray[0][y][x][0]
                if (confidence > threshold) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        // 번호판을 아예 찾지 못한 경우 (하얀색 픽셀이 없음)
        if (minX > maxX || minY > maxY) {
            return "[]"
        }

        // 5. 좌표 복원: 256x256 기준의 좌표를 원래 사진 크기 비율에 맞춰 확대
        val scaleX = bitmap.width.toFloat() / inputSize
        val scaleY = bitmap.height.toFloat() / inputSize

        val realMinX = minX * scaleX
        val realMaxX = maxX * scaleX
        val realMinY = minY * scaleY
        val realMaxY = maxY * scaleY

        // 6. JSON 형태로 4개 꼭짓점 반환 (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        return """
            [
              {"x": $realMinX, "y": $realMinY},
              {"x": $realMaxX, "y": $realMinY},
              {"x": $realMaxX, "y": $realMaxY},
              {"x": $realMinX, "y": $realMaxY}
            ]
        """.trimIndent()
    }
}
