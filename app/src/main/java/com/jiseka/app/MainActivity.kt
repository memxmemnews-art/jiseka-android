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

        webView = findViewById(R.id.webView) // 아까 수정한 대문자 W 유지
        webView.settings.javaScriptEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false // 카메라 자동 재생 허용
        
        // 🚨 [핵심 추가] 웹뷰 내부 카메라 권한 뚫어주기 (두 번째 보안문 해제)
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                // Vercel 웹사이트가 카메라 권한을 요청하면 무조건 승인!
                request.grant(request.resources)
            }
        }

        webView.addJavascriptInterface(WebAppInterface(), "JiSeKaNative")
        webView.loadUrl("https://ziseka-app.vercel.app")
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun sendImageData(base64Str: String) {
            val pureBase64 = base64Str.substringAfter(",")
            val decodedString = Base64.decode(pureBase64, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size)

            val cornersJson = runInference(bitmap)

            runOnUiThread {
                webView.evaluateJavascript("window.receiveAICorners('$cornersJson')", null)
            }
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        if (tflite == null) return "[]"
        
        // 현재는 통신 테스트를 위한 임시 가짜 좌표 응답입니다.
        return "[{\"x\":150, \"y\":200}, {\"x\":450, \"y\":200}, {\"x\":450, \"y\":300}, {\"x\":150, \"y\":300}]"
    }
}
