package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.Collections
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

// 🚨 중요: 프로젝트 패키지에 맞는 R 클래스를 명시적으로 임포트합니다.
import com.jiseka.app.R

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var resultImageView: ImageView
    private lateinit var cameraExecutor: ExecutorService
    
    // 🌐 웹에서 불러올 가림막 텍스처 (Nullable)
    private var plateTextureBmp: Bitmap? = null
    
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        private const val TARGET_MAX_DIMENSION = 1920 
        // 테스트용 가림막 이미지 URL (Vercel이나 본인 서버 주소로 교체하세요)
        private const val TEXTURE_URL = "https://your-project.vercel.app/plate_sample.png"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 1. UI 연결
        viewFinder = findViewById(R.id.viewFinder)
        resultImageView = findViewById(R.id.resultImageView)
        val shutterBtn = findViewById<Button>(R.id.shutterBtn)
        
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 2. 가림막 이미지 웹에서 비동기 로드
        loadTextureFromWeb(TEXTURE_URL)

        // 3. 촬영 버튼 리스너
        shutterBtn.setOnClickListener { captureAndProcess() }

        // 4. 권한 체크 및 카메라 시작
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        }
    }

    // ── 🌐 웹 텍스처 로딩 로직 ──
    private fun loadTextureFromWeb(urlString: String) {
        Executors.newSingleThreadExecutor().execute {
            try {
                val url = java.net.URL(urlString)
                val connection = url.openConnection() as java.net.HttpURLConnection
                connection.doInput = true
                connection.connectTimeout = 5000
                connection.connect()

                val inputStream = connection.inputStream
                val bitmap = BitmapFactory.decodeStream(inputStream)
                plateTextureBmp = bitmap
                
                runOnUiThread { Log.d("JiSeKa", "가림막 이미지 로드 완료") }
            } catch (e: Exception) {
                Log.e("JiSeKa", "웹 이미지 로드 실패, 로컬 리소스 확인 중", e)
                // 실패 시 로컬 drawable에서 fallback (파일명: plate_sample.png)
                try {
                    plateTextureBmp = BitmapFactory.decodeResource(resources, R.drawable.plate_sample)
                } catch (resEx: Exception) {
                    Log.e("JiSeKa", "로컬 리소스도 없습니다.")
                }
            }
        }
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder.surfaceProvider) }
            
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
                
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "카메라 바인딩 실패", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndProcess() {
        val capture = imageCapture ?: return
        if (plateTextureBmp == null) {
            Toast.makeText(this, "가림막 이미지를 불러오는 중입니다. 잠시만 기다려주세요.", Toast.LENGTH_SHORT).show()
            return
        }
        
        if (!isCapturing.compareAndSet(false, true)) return

        capture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                @SuppressLint("UnsafeOptInUsageError")
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    processEngineBackground(imageProxy)
                }
                override fun onError(exception: ImageCaptureException) {
                    isCapturing.set(false)
                }
            }
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processEngineBackground(imageProxy: ImageProxy) {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        imageProxy.close() 

        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        var originalBmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) ?: return

        // 회전 보정
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            originalBmp = Bitmap.createBitmap(originalBmp, 0, 0, originalBmp.width, originalBmp.height, matrix, true)
        }

        // 가이드 영역 계산 (화면 중앙 80%)
        val viewW = viewFinder.width.toFloat(); val viewH = viewFinder.height.toFloat()
        val imgW = originalBmp.width.toFloat(); val imgH = originalBmp.height.toFloat()
        val scale = maxOf(viewW / imgW, viewH / imgH)
        val offsetX = (viewW - imgW * scale) / 2f; val offsetY = (viewH - imgH * scale) / 2f

        val guideViewW = viewW * 0.8f; val guideViewH = guideViewW / 3f
        val guideRectImg = Rect(
            (((viewW - guideViewW) / 2f - offsetX) / scale).toInt(),
            (((viewH - guideViewH) / 2f - offsetY) / scale).toInt(),
            (((viewW + guideViewW) / 2f - offsetX) / scale).toInt(),
            (((viewH + guideViewH) / 2f - offsetY) / scale).toInt()
        )

        // 1. 번호판 코너 검출 (Overlap 스코어링 적용)
        val corners = extractGeometryCorners(originalBmp, guideRectImg)

        if (corners == null) {
            runOnUiThread { Toast.makeText(this, "번호판을 찾을 수 없습니다.", Toast.LENGTH_SHORT).show(); isCapturing.set(false) }
            return
        }

        // 2. OCR 평면화 검증
        val rectifiedBmp = rectifyToFlatPlate(originalBmp, corners)
        if (rectifiedBmp != null) {
            val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
            recognizer.process(inputImage).addOnCompleteListener { task ->
                val ocrValid = task.isSuccessful && task.result.text.isNotEmpty()
                
                // 3. 최종 합성 (조명 전사 렌더링)
                cameraExecutor.execute {
                    val finalImage = overlayTextureWithLighting(originalBmp, corners)
                    runOnUiThread {
                        resultImageView.setImageBitmap(finalImage)
                        resultImageView.visibility = View.VISIBLE
                        isCapturing.set(false)
                        rectifiedBmp.recycle() // 🚨 메모리 해제
                    }
                }
            }
        }
    }

    private fun extractGeometryCorners(bitmap: Bitmap, guideRect: Rect): Array<PointF>? {
        val mat = Mat(); Utils.bitmapToMat(bitmap, mat)
        val roiMat = Mat(mat, org.opencv.core.Rect(guideRect.left, guideRect.top, guideRect.width(), guideRect.height()))
        val gray = Mat(); Imgproc.cvtColor(roiMat, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Canny(gray, gray, 50.0, 150.0)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(gray, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var bestScore = 0.0
        var bestBox: Array<PointF>? = null

        for (contour in contours) {
            val minRect = Imgproc.minAreaRect(MatOfPoint2f(*contour.toArray()))
            val rw = minRect.size.width; val rh = minRect.size.height
            val aspect = max(rw, rh) / min(rw, rh)
            
            if (aspect in 2.0..6.0) {
                // 🚨 Overlap Ratio 계산
                val candBound = minRect.boundingRect()
                val overlapRatio = (minRect.size.area() / (guideRect.width() * guideRect.height())).coerceIn(0.0, 1.0)
                val score = overlapRatio + (1.0 / aspect)

                if (score > bestScore) {
                    bestScore = score
                    val pts = Mat(); Imgproc.boxPoints(minRect, pts)
                    bestBox = Array(4) { i -> 
                        PointF(pts.get(i,0)[0].toFloat() + guideRect.left, pts.get(i,1)[0].toFloat() + guideRect.top) 
                    }
                }
            }
        }

        // 🚨 코너 정렬 (Signed Area로 X자 꼬임 방지)
        bestBox?.let {
            val cx = it.map { p -> p.x }.average().toFloat()
            val cy = it.map { p -> p.y }.average().toFloat()
            val sorted = it.sortedBy { p -> atan2(p.y - cy, p.x - cx) }.toMutableList()
            
            var area = 0f
            for (i in 0..3) {
                val j = (i + 1) % 4
                area += sorted[i].x * sorted[j].y - sorted[j].x * sorted[i].y
            }
            if (area < 0) sorted.reverse() // 시계방향 강제
            
            val tlIdx = sorted.indices.minByOrNull { i -> sorted[i].x + sorted[i].y } ?: 0
            Collections.rotate(sorted, -tlIdx)
            return sorted.toTypedArray()
        }

        return null
    }

    private fun rectifyToFlatPlate(bitmap: Bitmap, corners: Array<PointF>): Bitmap? {
        val bgMat = Mat(); Utils.bitmapToMat(bitmap, bgMat)
        val srcPts = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        val dstPts = MatOfPoint2f(Point(0.0, 0.0), Point(400.0, 0.0), Point(400.0, 100.0), Point(0.0, 100.0))
        val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val dest = Mat(); Imgproc.warpPerspective(bgMat, dest, transform, Size(400.0, 100.0))
        val res = Bitmap.createBitmap(400, 100, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dest, res)
        return res
    }

    // ── 🚨 핵심: 원본 조명 전사(Luminance Transfer) 합성 ──
    private fun overlayTextureWithLighting(background: Bitmap, corners: Array<PointF>): Bitmap {
        val tex = plateTextureBmp ?: return background
        val bgMat = Mat(); Utils.bitmapToMat(background, bgMat); Imgproc.cvtColor(bgMat, bgMat, Imgproc.COLOR_RGBA2RGB)
        val texMat = Mat(); Utils.bitmapToMat(tex, texMat); Imgproc.cvtColor(texMat, texMat, Imgproc.COLOR_RGBA2RGB)

        // 1. Perspective Warp
        val srcPts = MatOfPoint2f(Point(0.0, 0.0), Point(texMat.cols().toDouble(), 0.0), Point(texMat.cols().toDouble(), texMat.rows().toDouble()), Point(0.0, texMat.rows().toDouble()))
        val dstPts = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val warpedTex = Mat(bgMat.size(), bgMat.type()); Imgproc.warpPerspective(texMat, warpedTex, transform, bgMat.size())

        // 2. 조명 전사 (Lighting Transfer)
        val bgGray = Mat(); Imgproc.cvtColor(bgMat, bgGray, Imgproc.COLOR_RGB2GRAY)
        val bgGrayFloat = Mat(); bgGray.convertTo(bgGrayFloat, CvType.CV_32FC1, 1.0/255.0)
        val bgGray3Ch = Mat(); Imgproc.cvtColor(bgGrayFloat, bgGray3Ch, Imgproc.COLOR_GRAY2RGB)
        
        val texFloat = Mat(); warpedTex.convertTo(texFloat, CvType.CV_32FC3)
        Core.multiply(texFloat, bgGray3Ch, texFloat) // 텍스처 색상 * 원본 명암
        
        // 3. 알파 블렌딩
        val mask = Mat.zeros(bgMat.size(), CvType.CV_8UC1)
        Imgproc.fillConvexPoly(mask, MatOfPoint(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()), Scalar(255.0))
        Imgproc.GaussianBlur(mask, mask, Size(5.0, 5.0), 0.0) // 경계 부드럽게
        
        val maskF = Mat(); mask.convertTo(maskF, CvType.CV_32FC1, 1.0/255.0)
        val mask3Ch = Mat(); Imgproc.cvtColor(maskF, mask3Ch, Imgproc.COLOR_GRAY2RGB)
        
        val bgF = Mat(); bgMat.convertTo(bgF, CvType.CV_32FC3)
        val invMask = Mat(); Core.subtract(Scalar(1.0, 1.0, 1.0), mask3Ch, invMask)
        
        val finalF = Mat()
        Core.multiply(texFloat, mask3Ch, texFloat)
        Core.multiply(bgF, invMask, bgF)
        Core.add(texFloat, bgF, finalF)

        val resMat = Mat(); finalF.convertTo(resMat, CvType.CV_8UC3)
        val resBmp = Bitmap.createBitmap(resMat.cols(), resMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resMat, resBmp)
        
        // 메모리 해제
        bgMat.release(); texMat.release(); warpedTex.release(); bgGray.release(); mask.release()
        return resBmp
    }
}
