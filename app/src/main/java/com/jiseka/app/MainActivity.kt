package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }

        webView = findViewById(R.id.webView)
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                request?.grant(request.resources)
            }
        }
        webView.webViewClient = WebViewClient()
        webView.addJavascriptInterface(JiSeKaBridge(this, webView), "JiSeKaNative")
        
        // 사용자님의 실제 Vercel 주소
        webView.loadUrl("https://ziseka-app.vercel.app")
    }
}

class JiSeKaBridge(private val context: Context, private val webView: WebView) {
    @JavascriptInterface
    fun processImage(base64Data: String, bboxJSON: String) {
        Log.d("JiSeKaNative", "사진 수신 완료, 가짜 좌표 반환 대기 중...")
        val fakeCorners = "[{\"x\":100,\"y\":100},{\"x\":200,\"y\":100},{\"x\":200,\"y\":150},{\"x\":100,\"y\":150}]"
        Handler(Looper.getMainLooper()).postDelayed({
            webView.evaluateJavascript("window.receiveAICorners('$fakeCorners');", null)
        }, 1000)
    }
}
