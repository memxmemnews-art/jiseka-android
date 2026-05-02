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
            // 🚨 [핵심 수술] 무거운 AI 연산을 백그라운드 스레드로 던져서 화면 멈춤(Freeze) 원천 차단!
            Thread {
                try {
                    val pureBase64 = base64Str.substringAfter(",")
                    val decodedByteArray = Base64.decode(pureBase64, Base64.DEFAULT)
                    
                    // 웹에서 가이드 박스 크기만큼 잘라서 보내주므로, 화질 저하 없이 원본 그대로 압축 해제
                    val bitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.size) 
                        ?: throw Exception("비트맵 변환 실패")

                    val cornersJson = runInference(bitmap)

                    // UI(화면) 업데이트는 다시 메인 스레드에서 안전하게 실행
                    runOnUiThread {
                        webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
                    }
                } catch (e: Throwable) { 
                    val safeMsg = e.message?.replace(Regex("[^a-zA-Z0-9가-힣 ]"), "_") ?: "Fatal Error"
                    runOnUiThread {
                        webView.evaluateJavascript("alert('AI 에러 발생: $safeMsg'); window.receiveAICorners('[]');", null)
                    }
                }
            }.start() // 스레드 실행
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) throw Exception("TFLite 모델을 찾을 수 없습니다")

        val inputSize = 256
        
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

        val outputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 1)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        tflite?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind() 

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

        if (minX > maxX || minY > maxY) {
            return "[]"
        }

        val scaleX = bitmap.width.toFloat() / inputSize.toFloat()
        val scaleY = bitmap.height.toFloat() / inputSize.toFloat()

        val realMinX = minX * scaleX
        val realMaxX = maxX * scaleX
        val realMinY = minY * scaleY
        val realMaxY = maxY * scaleY

        // 🚨 줄바꿈(엔터) 없이 한 줄의 JSON으로 반환하여 웹 통신 에러 방지
        return "[{\"x\": $realMinX, \"y\": $realMinY}, {\"x\": $realMaxX, \"y\": $realMinY}, {\"x\": $realMaxX, \"y\": $realMaxY}, {\"x\": $realMinX, \"y\": $realMaxY}]"
    }
}
