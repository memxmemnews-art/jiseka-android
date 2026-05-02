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

        // 1. AI 모델 로드
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "fairscan-segmentation-model.tflite")
            tflite = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        webView = findViewById(R.id.webView)
        webView.settings.javaScriptEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false
        webView.settings.domStorageEnabled = true
        
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
                val pureBase64 = base64Str.substringAfter(",")
                val decodedString = Base64.decode(pureBase64, Base64.DEFAULT)
                val bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size) 
                    ?: throw Exception("비트맵 변환 실패")

                val cornersJson = runInference(bitmap)

                // 정상 결과 반환
                runOnUiThread {
                    webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
                }
            } catch (e: Exception) {
                // 🚨 에러 메시지 안에 섞인 특수문자나 줄바꿈 때문에 JS가 또 뻗는 것을 방지
                val safeMsg = e.message?.replace(Regex("[^a-zA-Z0-9가-힣 ]"), "_") ?: "Unknown Error"
                runOnUiThread {
                    webView.evaluateJavascript("alert('AI 에러 발생: $safeMsg'); window.receiveAICorners('[]');", null)
                }
            }
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) throw Exception("TFLite 모델을 찾을 수 없습니다")

        val inputSize = 256
        
        // 1. 입력 이미지 리사이징 (256x256)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val valInt = intValues[pixel++]
                inputBuffer.putFloat(((valInt shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((valInt shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((valInt and 0xFF) / 255.0f)
            }
        }

        // 2. 출력 버퍼 (모델의 배열 모양에 상관없이 데이터를 강제로 쏟아붓게 만드는 마법의 ByteBuffer)
        val outputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 1)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // 3. AI 실행
        tflite?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind() // 다 읽은 후 처음으로 되감기

        // 4. 좌표 추출
        var minX = inputSize
        var minY = inputSize
        var maxX = -1
        var maxY = -1

        val threshold = 0.5f 

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val confidence = outputBuffer.float
                if (confidence > threshold) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        // 번호판을 못 찾은 경우
        if (minX > maxX || minY > maxY) {
            return "[]"
        }

        // 5. 원본 비율로 확대
        val scaleX = bitmap.width.toFloat() / inputSize
        val scaleY = bitmap.height.toFloat() / inputSize

        val realMinX = minX * scaleX
        val realMaxX = maxX * scaleX
        val realMinY = minY * scaleY
        val realMaxY = maxY * scaleY

        // 6. JSON 반환 (🚨 절대 엔터키를 치지 않고 한 줄로 반환해야 JS 에러가 안 납니다!)
        return "[{\"x\": $realMinX, \"y\": $realMinY}, {\"x\": $realMaxX, \"y\": $realMinY}, {\"x\": $realMaxX, \"y\": $realMaxY}, {\"x\": $realMinX, \"y\": $realMaxY}]"
    }
}
