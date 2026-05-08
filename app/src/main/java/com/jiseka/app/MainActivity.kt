package com.jiseka.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
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
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.util.Collections
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var resultImageView: ImageView
    private lateinit var cameraExecutor: ExecutorService
    
    // 🚨 개선: 멀티스레드 Race Condition 방지를 위한 Volatile 적용 및 OOM 방지 구조 
    @Volatile
    private var plateTextureBmp: Bitmap? = null
    
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        // 🚨 개선: 캐싱 우회를 위한 버전 파라미터 적용 (실서비스 구조)
        private const val TEXTURE_URL = "https://your-project.vercel.app/plate_sample.png?v=1"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("JiSeKa", "OpenCV 초기화 실패")
            Toast.makeText(this, "엔진 초기화에 실패했습니다.", Toast.LENGTH_LONG).show()
        } else {
            Log.d("JiSeKa", "OpenCV 초기화 성공")
        }

        viewFinder = findViewById(R.id.viewFinder)
        resultImageView = findViewById(R.id.resultImageView)
        val shutterBtn = findViewById<Button>(R.id.shutterBtn)
        
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 앱 실행 시 즉시 웹에서 텍스처 최신화
        loadTextureFromWeb(TEXTURE_URL)

        shutterBtn.setOnClickListener { captureAndProcess() }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        }
    }

    // ── 🚨 개선된 웹 텍스처 다운로드 시스템 (안정화) ──
    private fun loadTextureFromWeb(urlString: String) {
        Executors.newSingleThreadExecutor().execute {
            try {
                val connection = java.net.URL(urlString).openConnection() as java.net.HttpURLConnection
                connection.connectTimeout = 5000
                connection.readTimeout = 5000 
                connection.doInput = true
                connection.useCaches = false // 최신 버전 강제 확보
                connection.connect()

                // OOM 방어 및 해상도 최적화
                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                val bitmap = BitmapFactory.decodeStream(connection.inputStream, null, options)
                if (bitmap == null) {
                    Log.e("JiSeKa", "Bitmap decode 실패")
                    return@execute
                }

                // 🚨 기존 Bitmap 메모리 누수(Leak) 방어 및 안전한 교체
                val old = plateTextureBmp
                plateTextureBmp = downscaleBitmapIfNeeded(bitmap, 1024) 
                
                old?.let {
                    if (!it.isRecycled) it.recycle()
                }

                Log.d("JiSeKa", "웹 가림막 로드 성공")
            } catch (e: Exception) {
                // 🚨 로컬 폴백(R.drawable...) 완벽 제거. 철저히 웹 CDN 의존.
                Log.e("JiSeKa", "웹 가림막 다운로드 실패", e)
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "가림막을 불러오지 못했습니다. 네트워크를 확인하세요.", Toast.LENGTH_SHORT).show()
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
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("JiSeKa", "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndProcess() {
        val capture = imageCapture ?: return
        if (plateTextureBmp == null) {
            Toast.makeText(this, "가림막 리소스를 준비 중입니다.", Toast.LENGTH_SHORT).show()
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
        val bitmap = imageProxy.toBitmapExt()
        val rotation = imageProxy.imageInfo.rotationDegrees
        imageProxy.close() 

        var originalBmp = bitmap
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated != bitmap) bitmap.recycle()
            originalBmp = rotated
        }

        originalBmp = downscaleBitmapIfNeeded(originalBmp, 1920)

        val viewW = viewFinder.width.toFloat(); val viewH = viewFinder.height.toFloat()
        val scale = maxOf(viewW / originalBmp.width, viewH / originalBmp.height)
        val guideViewW = viewW * 0.8f; val guideViewH = guideViewW / 3f
        
        val guideRectImg = Rect(
            (((viewW - guideViewW) / 2f - (viewW - originalBmp.width * scale) / 2f) / scale).toInt(),
            (((viewH - guideViewH) / 2f - (viewH - originalBmp.height * scale) / 2f) / scale).toInt(),
            (((viewW + guideViewW) / 2f - (viewW - originalBmp.width * scale) / 2f) / scale).toInt(),
            (((viewH + guideViewH) / 2f - (viewH - originalBmp.height * scale) / 2f) / scale).toInt()
        )

        val corners = extractGeometryCorners(originalBmp, guideRectImg)
        if (corners == null) {
            runOnUiThread { Toast.makeText(this, "번호판을 찾을 수 없습니다.", Toast.LENGTH_SHORT).show(); isCapturing.set(false) }
            return
        }

        val rectifiedBmp = rectifyToFlatPlate(originalBmp, corners)
        if (rectifiedBmp == null) {
            runOnUiThread { Toast.makeText(this, "평면화에 실패했습니다.", Toast.LENGTH_SHORT).show(); isCapturing.set(false) }
            return
        }

        val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
        recognizer.process(inputImage).addOnCompleteListener { task ->
            val ocrValid = task.isSuccessful && task.result.text.isNotEmpty()
            if (!ocrValid) {
                runOnUiThread { Toast.makeText(this, "번호판 검증 실패", Toast.LENGTH_SHORT).show(); isCapturing.set(false) }
                rectifiedBmp.recycle()
                return@addOnCompleteListener
            }
            
            cameraExecutor.execute {
                val finalImage = overlayTextureWithLighting(originalBmp, corners)
                runOnUiThread {
                    resultImageView.setImageBitmap(finalImage)
                    resultImageView.visibility = View.VISIBLE
                    isCapturing.set(false)
                    rectifiedBmp.recycle()
                }
            }
        }
    }

    private fun downscaleBitmapIfNeeded(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val maxDim = max(bitmap.width, bitmap.height)
        if (maxDim <= maxDimension) return bitmap
        val scale = maxDimension.toFloat() / maxDim
        val scaled = Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
        if (scaled != bitmap) bitmap.recycle()
        return scaled
    }

    private fun extractGeometryCorners(bitmap: Bitmap, guideRect: Rect): Array<PointF>? {
        var mat: Mat? = null; var roiMat: Mat? = null; var gray: Mat? = null; var edges: Mat? = null
        val contours = ArrayList<MatOfPoint>()

        try {
            mat = Mat(); Utils.bitmapToMat(bitmap, mat)
            
            val roiX = max(0, guideRect.left); val roiY = max(0, guideRect.top)
            val roiW = min(guideRect.width(), mat.cols() - roiX); val roiH = min(guideRect.height(), mat.rows() - roiY)
            if (roiW <= 0 || roiH <= 0) return null

            roiMat = Mat(mat, org.opencv.core.Rect(roiX, roiY, roiW, roiH))
            gray = Mat(); Imgproc.cvtColor(roiMat, gray, Imgproc.COLOR_RGBA2GRAY)
            edges = Mat(); Imgproc.Canny(gray, edges, 50.0, 150.0)

            Imgproc.findContours(edges, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestScore = 0.0
            var bestBox: Array<PointF>? = null

            for (contour in contours) {
                val minRect = Imgproc.minAreaRect(MatOfPoint2f(*contour.toArray()))
                val rw = minRect.size.width; val rh = minRect.size.height
                val aspect = max(rw, rh) / min(rw, rh).coerceAtLeast(1.0)
                
                if (aspect in 2.0..6.0) {
                    val overlapRatio = (minRect.size.area() / (guideRect.width() * guideRect.height())).coerceIn(0.0, 1.0)
                    val score = overlapRatio + (1.0 / aspect)
                    if (score > bestScore) {
                        bestScore = score
                        val pts = Mat(); Imgproc.boxPoints(minRect, pts)
                        bestBox = Array(4) { i -> PointF(pts.get(i,0)[0].toFloat() + roiX, pts.get(i,1)[0].toFloat() + roiY) }
                        pts.release()
                    }
                }
            }

            bestBox?.let {
                val cx = it.map { p -> p.x }.average().toFloat()
                val cy = it.map { p -> p.y }.average().toFloat()
                val sorted = it.sortedBy { p -> atan2(p.y - cy, p.x - cx) }.toMutableList()
                var area = 0f
                for (i in 0..3) {
                    val j = (i + 1) % 4
                    area += sorted[i].x * sorted[j].y - sorted[j].x * sorted[i].y
                }
                if (area < 0) sorted.reverse()
                val tlIdx = sorted.indices.minByOrNull { i -> sorted[i].x + sorted[i].y } ?: 0
                Collections.rotate(sorted, -tlIdx)
                return sorted.toTypedArray()
            }
            return null
        } finally {
            mat?.release(); roiMat?.release(); gray?.release(); edges?.release()
            for (c in contours) c.release()
        }
    }

    private fun rectifyToFlatPlate(bitmap: Bitmap, corners: Array<PointF>): Bitmap? {
        var bgMat: Mat? = null; var dest: Mat? = null
        try {
            bgMat = Mat(); Utils.bitmapToMat(bitmap, bgMat)
            val srcPts = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val dstPts = MatOfPoint2f(Point(0.0, 0.0), Point(400.0, 0.0), Point(400.0, 100.0), Point(0.0, 100.0))
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            dest = Mat(); Imgproc.warpPerspective(bgMat, dest, transform, Size(400.0, 100.0))
            val res = Bitmap.createBitmap(400, 100, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(dest, res)
            return res
        } catch (e: Exception) {
            return null
        } finally {
            bgMat?.release(); dest?.release()
        }
    }

    private fun overlayTextureWithLighting(background: Bitmap, corners: Array<PointF>): Bitmap {
        val tex = plateTextureBmp ?: return background
        var bgMat: Mat? = null; var texMat: Mat? = null; var warpedTex: Mat? = null
        var bgGray: Mat? = null; var bgGrayFloat: Mat? = null; var bgGray3Ch: Mat? = null
        var texFloat: Mat? = null; var mask: Mat? = null; var maskF: Mat? = null
        var mask3Ch: Mat? = null; var bgF: Mat? = null; var ones: Mat? = null
        var invMask: Mat? = null; var finalF: Mat? = null; var resMat: Mat? = null

        try {
            bgMat = Mat(); Utils.bitmapToMat(background, bgMat); Imgproc.cvtColor(bgMat, bgMat, Imgproc.COLOR_RGBA2RGB)
            texMat = Mat(); Utils.bitmapToMat(tex, texMat); Imgproc.cvtColor(texMat, texMat, Imgproc.COLOR_RGBA2RGB)

            val srcPts = MatOfPoint2f(Point(0.0, 0.0), Point(texMat.cols().toDouble(), 0.0), Point(texMat.cols().toDouble(), texMat.rows().toDouble()), Point(0.0, texMat.rows().toDouble()))
            val dstPts = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            warpedTex = Mat(bgMat.size(), bgMat.type())
            Imgproc.warpPerspective(texMat, warpedTex, transform, bgMat.size())

            bgGray = Mat(); Imgproc.cvtColor(bgMat, bgGray, Imgproc.COLOR_RGB2GRAY)
            bgGrayFloat = Mat(); bgGray.convertTo(bgGrayFloat, CvType.CV_32FC1, 1.0/255.0)
            bgGray3Ch = Mat(); Imgproc.cvtColor(bgGrayFloat, bgGray3Ch, Imgproc.COLOR_GRAY2RGB)
            
            texFloat = Mat(); warpedTex.convertTo(texFloat, CvType.CV_32FC3)
            Core.multiply(texFloat, bgGray3Ch, texFloat)
            
            mask = Mat.zeros(bgMat.size(), CvType.CV_8UC1)
            Imgproc.fillConvexPoly(mask, MatOfPoint(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()), Scalar(255.0))
            Imgproc.GaussianBlur(mask, mask, Size(5.0, 5.0), 0.0)
            
            maskF = Mat(); mask.convertTo(maskF, CvType.CV_32FC1, 1.0/255.0)
            mask3Ch = Mat(); Imgproc.cvtColor(maskF, mask3Ch, Imgproc.COLOR_GRAY2RGB)
            bgF = Mat(); bgMat.convertTo(bgF, CvType.CV_32FC3)
            
            // 🚨 안전한 OpenCV Core.subtract 연산 유지
            ones = Mat(bgMat.size(), mask3Ch.type(), Scalar(1.0, 1.0, 1.0))
            invMask = Mat()
            Core.subtract(ones, mask3Ch, invMask)
            
            finalF = Mat()
            Core.multiply(texFloat, mask3Ch, texFloat)
            Core.multiply(bgF, invMask, bgF)
            Core.add(texFloat, bgF, finalF)

            resMat = Mat(); finalF.convertTo(resMat, CvType.CV_8UC3)
            val resBmp = Bitmap.createBitmap(resMat.cols(), resMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(resMat, resBmp)
            
            return resBmp
        } finally {
            bgMat?.release(); texMat?.release(); warpedTex?.release()
            bgGray?.release(); bgGrayFloat?.release(); bgGray3Ch?.release()
            texFloat?.release(); mask?.release(); maskF?.release()
            mask3Ch?.release(); bgF?.release(); ones?.release()
            invMask?.release(); finalF?.release(); resMat?.release()
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun ImageProxy.toBitmapExt(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
