package com.jiseka.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors

class SplashActivity : AppCompatActivity() {

    private val splashDelay = 2000L
    private val startTime = System.currentTimeMillis()
    
    // 💡 Race Condition 방지용 플래그 추가
    private var alreadyProceeding = false 

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        Executors.newSingleThreadExecutor().execute {
            // 💡 OpenCV 초기화 실패 시 즉시 종료되도록 수정
            if (OpenCVLoader.initDebug()) {
                Log.d("SPLASH_DEBUG", "OpenCV initialization succeeded.")
                
                runOnUiThread {
                    if (checkCameraPermission()) {
                        proceedToMainWithDelay()
                    }
                }
            } else {
                Log.e("SPLASH_DEBUG", "OpenCV initialization failed.")
                runOnUiThread {
                    Toast.makeText(this@SplashActivity, "핵심 모듈(OpenCV) 초기화에 실패하여 앱을 종료합니다.", Toast.LENGTH_LONG).show()
                    finish()
                }
            }
        }

        if (!checkCameraPermission()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA
            )
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun proceedToMainWithDelay() {
        // 💡 중복 실행 방지 방어 코드 적용
        if (alreadyProceeding) return
        alreadyProceeding = true

        val elapsedTime = System.currentTimeMillis() - startTime
        val remainingTime = splashDelay - elapsedTime

        Handler(Looper.getMainLooper()).postDelayed({
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }, if (remainingTime > 0) remainingTime else 0)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                proceedToMainWithDelay()
            } else {
                Toast.makeText(this, "카메라 권한이 필수적으로 필요합니다.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    companion object {
        private const val REQUEST_CODE_CAMERA = 1001
    }
}
