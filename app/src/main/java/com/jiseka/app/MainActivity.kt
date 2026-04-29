package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

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
            e.printStackTrace() // 모델 로드 실패 시 로그 출력
        }

        webView = findViewById(R.id.webview)
        webView.settings.javaScriptEnabled = true
        webView.addJavascriptInterface(WebAppInterface(), "JiSeKaNative")
        webView.loadUrl("https://ziseka-app.vercel.app")
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun sendImageData(base64Str: String) {
            // 2. 웹에서 넘어온 Base64 데이터를 진짜 사진(Bitmap)으로 변환
            val pureBase64 = base64Str.substringAfter(",")
            val decodedString = Base64.decode(pureBase64, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size)

            // 3. AI 분석 실행 
            val cornersJson = runInference(bitmap)

            // 4. 찾은 좌표를 다시 Vercel 웹으로 쏘아주기!
            runOnUiThread {
                webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
            }
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) {
            return "[]" // 모델이 없으면 빈 배열 반환
        }
        
        // TODO: segmentation 모델의 출력 배열을 분석하여 4개 꼭짓점(x, y)으로 변환하는 세부 로직 필요.
        // 현재는 통신(브릿지)이 완벽하게 성공했는지 확인하기 위한 임시 가짜 응답입니다.
        return "[{\"x\":150, \"y\":200}, {\"x\":450, \"y\":200}, {\"x\":450, \"y\":300}, {\"x\":150, \"y\":300}]"
    }
}
