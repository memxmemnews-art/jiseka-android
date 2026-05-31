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

    private val splashDelay = 2000L // 최소 스플래시 대기 시간 (2초)
    private val startTime = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        // 1. 백그라운드 스레드에서 OpenCV 미리 초기화
        Executors.newSingleThreadExecutor().execute {
            if (OpenCVLoader.initDebug()) {
                Log.d("SPLASH_DEBUG", "OpenCV initialization succeeded.")
            } else {
                Log.e("SPLASH_DEBUG", "OpenCV initialization failed.")
            }
            
            // 2. 초기화 완료 후 메인 스레드에서 권한 확인
            runOnUiThread {
                if (checkCameraPermission()) {
                    proceedToMainWithDelay()
                }
            }
        }

        // 3. 카메라 권한이 없다면 요청
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
