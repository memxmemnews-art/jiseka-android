package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private var tflite: Interpreter? = null

    // 🚨 [핵심 변경] 대용량 조각 데이터를 안전하게 모으기 위한 동기화 보장 버퍼 (Thread-Safe)
    private val base64Buffer = StringBuffer()

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
        webView.settings.domStorageEnabled = true
        
        // 캐시 방지
        webView.settings.cacheMode = WebSettings.LOAD_NO_CACHE
        
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
        
        // 🚨 [신규] 1. 스트리밍 시작 (버퍼 초기화)
        @JavascriptInterface
        fun startImageStream() {
            base64Buffer.setLength(0)
        }

        // 🚨 [신규] 2. 데이터 조각 수신 (차곡차곡 모으기)
        @JavascriptInterface
        fun appendImageChunk(chunk: String) {
            base64Buffer.append(chunk)
        }

        // 🚨 [신규] 3. 스트리밍 종료 및 AI 추론 시작
        @JavascriptInterface
        fun finishImageStream() {
            Thread {
                try {
                    // 순수 Base64만 넘어오도록 JS에서 조치했으므로 substringAfter(",") 불필요
                    val base64Str = base64Buffer.toString()
                    val decoded = Base64.decode(base64Str, Base64.DEFAULT)
                    
                    val bitmap = BitmapFactory.decodeByteArray(decoded, 0, decoded.size) 
                        ?: throw Exception("비트맵 변환 실패")

                    val cornersJson = runInference(bitmap)

                    runOnUiThread {
                        webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
                    }
                } catch (e: Throwable) { 
                    val safeMsg = e.message?.replace(Regex("[^a-zA-Z0-9가-힣 ]"), "_") ?: "Fatal Error"
                    runOnUiThread {
                        webView.evaluateJavascript("alert('AI 에러 발생: $safeMsg'); window.receiveAICorners('[]');", null)
                    }
                }
            }.start()
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

        return "[{\"x\": $realMinX, \"y\": $realMinY}, {\"x\": $realMaxX, \"y\": $realMinY}, {\"x\": $realMaxX, \"y\": $realMaxY}, {\"x\": $realMinX, \"y\": $realMaxY}]"
    }
}
