package com.jiseka.app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {

    // UI 컴포넌트
    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null
    private var modeSelectionLayout: LinearLayout? = null
    private var resultActionLayout: LinearLayout? = null
    private var btnCapture: Button? = null
    private var progressBar: ProgressBar? = null

    // 모드 버튼
    private var btnModePassenger: Button? = null
    private var btnModeFront: Button? = null
    private var btnModeDriver: Button? = null

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null
    private var displayedBitmap: Bitmap? = null
    private val transformLock = Any()
    private var lastTransformData: CaptureTransformData? = null

    @Volatile private var isProcessing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("CAMERA_DEBUG", "OpenCV initialization failed.")
        }

        // 뷰 바인딩
        viewFinder = findViewById(R.id.viewFinder)
        
        // [수정 1] 기기 파편화로 인한 SurfaceView 강제 돌출 및 Z-Order 버그(검은 화면)를 막기 위해 COMPATIBLE(TextureView) 강제 설정
        viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeGuideView = findViewById(R.id.nativeGuideView)
        modeSelectionLayout = findViewById(R.id.modeSelectionLayout)
        resultActionLayout = findViewById(R.id.resultActionLayout)
        btnCapture = findViewById(R.id.btnCapture)
        progressBar = findViewById(R.id.progressBar)
        
        btnModePassenger = findViewById(R.id.btnModePassenger)
        btnModeFront = findViewById(R.id.btnModeFront)
        btnModeDriver = findViewById(R.id.btnModeDriver)

        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()

        setupUIListeners()

        if (allPermissionsGranted()) {
            viewFinder?.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun setupUIListeners() {
        btnModePassenger?.setOnClickListener { setMode("PASSENGER", it as Button) }
        btnModeFront?.setOnClickListener { setMode("FRONT", it as Button) }
        btnModeDriver?.setOnClickListener { setMode("DRIVER", it as Button) }

        btnCapture?.setOnClickListener { takePhoto() }

        findViewById<Button>(R.id.btnRetry).setOnClickListener {
            resetToLiveMode()
        }

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            displayedBitmap?.let { bmp ->
                saveBitmapToGallery(bmp)
            } ?: Toast.makeText(this, "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setMode(mode: String, clickedButton: Button) {
        if (isProcessing) return
        
        val defaultBg = Color.parseColor("#80000000")
        val activeBg = Color.parseColor("#00FF00")
        val defaultText = Color.WHITE
        val activeText = Color.BLACK

        btnModePassenger?.apply { setBackgroundColor(defaultBg); setTextColor(defaultText) }
        btnModeFront?.apply { setBackgroundColor(defaultBg); setTextColor(defaultText) }
        btnModeDriver?.apply { setBackgroundColor(defaultBg); setTextColor(defaultText) }

        clickedButton.setBackgroundColor(activeBg)
        clickedButton.setTextColor(activeText)

        nativeGuideView?.setMode(mode)
    }

    private fun resetToLiveMode() {
        isProcessing = false
        btnCapture?.isEnabled = true

        nativeBackgroundView?.setImageBitmap(null)
        displayedBitmap?.recycle()
        displayedBitmap = null

        // [수정 6] 재촬영 시 백그라운드에 남아있는 비트맵 찌꺼기까지 완벽하게 소거하여 OOM 방지
        synchronized(bitmapLock) {
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null
        }

        viewFinder?.visibility = View.VISIBLE
        nativeGuideView?.visibility = View.VISIBLE
        modeSelectionLayout?.visibility = View.VISIBLE
        btnCapture?.visibility = View.VISIBLE
        
        nativeBackgroundView?.visibility = View.GONE
        resultActionLayout?.visibility = View.GONE
        progressBar?.visibility = View.GONE
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(viewFinder?.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("CAMERA_DEBUG", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        if (isProcessing) return
        isProcessing = true
        
        // [수정 5] 분석 중 연타로 인한 스레드 큐 적체 현상을 막기 위해 즉시 버튼 비활성화
        btnCapture?.isEnabled = false
        
        progressBar?.visibility = View.VISIBLE
        btnCapture?.visibility = View.GONE
        modeSelectionLayout?.visibility = View.GONE

        val currentCapture = imageCapture ?: run {
            Toast.makeText(this, "카메라 초기화 실패", Toast.LENGTH_SHORT).show()
            resetToLiveMode()
            return
        }

        // [수정 2] 무거운 원본 비트맵 대신 RGB_565로 다운샘플링하여 스냅샷을 렌더링하고 원본은 즉시 회수
        viewFinder?.bitmap?.let { original ->
            val snapshot = original.copy(Bitmap.Config.RGB_565, false)
            nativeBackgroundView?.setImageBitmap(snapshot)
            nativeBackgroundView?.visibility = View.VISIBLE
            original.recycle()
        }

        currentCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    try {
                        val uprightBitmap = imageProxy.toUprightBitmap()
                        val transformData = CaptureTransformData(
                            sensorToBufferMatrix = imageProxy.imageInfo.sensorToBufferTransformMatrix,
                            rotationDegrees = imageProxy.imageInfo.rotationDegrees,
                            bufferWidth = imageProxy.width,
                            bufferHeight = imageProxy.height
                        )

                        synchronized(bitmapLock) {
                            if (lastCapturedBitmap !== uprightBitmap && lastCapturedBitmap?.isRecycled == false) {
                                lastCapturedBitmap?.recycle()
                            }
                            lastCapturedBitmap = uprightBitmap
                        }
                        synchronized(transformLock) { lastTransformData = transformData }

                        val currentMode = nativeGuideView?.currentMode ?: "FRONT"
                        triggerAnalysis(currentMode)
                        
                    } catch (t: Throwable) {
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "이미지 처리 오류", Toast.LENGTH_SHORT).show()
                            resetToLiveMode()
                        }
                    } finally {
                        imageProxy.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "카메라 캡처 실패", Toast.LENGTH_SHORT).show()
                        resetToLiveMode()
                    }
                }
            }
        )
    }

    private fun triggerAnalysis(mode: String) {
        val targetBitmap = synchronized(bitmapLock) { lastCapturedBitmap }
        val transformData = synchronized(transformLock) { lastTransformData }
        val guideView = nativeGuideView

        // [수정 7] 비동기 콜백 시점에서 뷰파인더가 날아갔을 경우를 대비한 NPE 철벽 방어
        val vf = viewFinder
        if (targetBitmap == null || transformData == null || guideView == null || vf == null) {
            runOnUiThread { resetToLiveMode() }
            return
        }

        val uiCorners = guideView.getCorners()
        analysisExecutor.execute {
            try {
                val exactCorners = CameraCoordinateConverter.mapUiToExactBitmap(
                    vf, transformData, uiCorners, targetBitmap.width, targetBitmap.height
                )

                val detectedCorners = PlateDetectionEngine.findPlateCorners(targetBitmap)
                val finalCorners = detectedCorners ?: exactCorners

                val resultMat = Mat()
                Utils.bitmapToMat(targetBitmap, resultMat)
                
                applyMaskToMat(resultMat, finalCorners, mode)

                val resultBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(resultMat, resultBitmap)
                resultMat.release()

                runOnUiThread {
                    displayedBitmap?.recycle()
                    displayedBitmap = resultBitmap

                    viewFinder?.visibility = View.GONE
                    nativeGuideView?.visibility = View.GONE
                    progressBar?.visibility = View.GONE
                    
                    nativeBackgroundView?.setImageBitmap(resultBitmap)
                    nativeBackgroundView?.visibility = View.VISIBLE
                    resultActionLayout?.visibility = View.VISIBLE
                    isProcessing = false
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "분석 중 오류 발생", Toast.LENGTH_SHORT).show()
                    resetToLiveMode()
                }
            }
        }
    }

    // [수정 4] 예외 발생 시 메모리에 남아있는 모든 OpenCV Mat 인스턴스를 무조건 해제(recycle)하는 finally 구조 도입
    private fun applyMaskToMat(mat: Mat, corners: List<PointF>, mode: String) {
        if (corners.size != 4) return
        
        var srcPoints: MatOfPoint2f? = null
        var dstPoints: MatOfPoint2f? = null
        var transformMatrix: Mat? = null
        var transformedMat: Mat? = null
        var maskMat: Mat? = null
        var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null
        var coloredMask: Mat? = null
        var alphaMat: Mat? = null
        val matChannels = ArrayList<Mat>()
        val coloredChannels = ArrayList<Mat>()
        
        try {
            val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }
            val maskColor = Scalar(0.0, 255.0, 0.0, 255.0)

            srcPoints = MatOfPoint2f(*pts.toTypedArray())
            val minX = pts.minOf { it.x }; val maxX = pts.maxOf { it.x }
            val minY = pts.minOf { it.y }; val maxY = pts.maxOf { it.y }
            val w = maxX - minX; val h = maxY - minY

            val targetRatio = 3.0 / 1.0
            var newW = w; var newH = h
            if (w / h > targetRatio) { newH = w / targetRatio } else { newW = h * targetRatio }

            val center = Point((minX + maxX) / 2.0, (minY + maxY) / 2.0)
            dstPoints = MatOfPoint2f(
                Point(center.x - newW / 2.0, center.y - newH / 2.0),
                Point(center.x + newW / 2.0, center.y - newH / 2.0),
                Point(center.x + newW / 2.0, center.y + newH / 2.0),
                Point(center.x - newW / 2.0, center.y + newH / 2.0)
            )

            transformMatrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
            transformedMat = Mat()
            Imgproc.warpPerspective(mat, transformedMat, transformMatrix, Size(mat.cols().toDouble(), mat.rows().toDouble()))

            maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
            contour = org.opencv.core.MatOfPoint(*dstPoints.toArray().map { Point(it.x, it.y) }.toTypedArray())
            Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))

            blurredMask = Mat()
            Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)

            coloredMask = Mat(mat.size(), mat.type(), maskColor)
            alphaMat = Mat()
            blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)

            Core.split(mat, matChannels)
            Core.split(coloredMask, coloredChannels)

            for (i in 0 until 3) {
                var mcF: Mat? = null
                var ccF: Mat? = null
                var blendedF: Mat? = null
                var invAlpha: Mat? = null
                var scalarMat: Mat? = null
                try {
                    val mc = matChannels[i]
                    val cc = coloredChannels[i]
                    mcF = Mat(); ccF = Mat()
                    mc.convertTo(mcF, CvType.CV_32F)
                    cc.convertTo(ccF, CvType.CV_32F)

                    blendedF = Mat()
                    Core.multiply(ccF, alphaMat, ccF)

                    invAlpha = Mat()
                    scalarMat = Mat(alphaMat.size(), alphaMat.type(), Scalar(1.0))
                    Core.subtract(scalarMat, alphaMat, invAlpha)

                    Core.multiply(mcF, invAlpha, mcF)
                    Core.add(ccF, mcF, blendedF)

                    blendedF.convertTo(matChannels[i], CvType.CV_8U)
                } finally {
                    mcF?.release(); ccF?.release(); blendedF?.release()
                    invAlpha?.release(); scalarMat?.release()
                }
            }
            Core.merge(matChannels, mat)

        } finally {
            srcPoints?.release(); dstPoints?.release(); transformMatrix?.release()
            transformedMat?.release(); maskMat?.release(); contour?.release()
            blurredMask?.release(); coloredMask?.release(); alphaMat?.release()
            matChannels.forEach { it.release() }; coloredChannels.forEach { it.release() }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        try {
            val filename = "JiSeKa_${System.currentTimeMillis()}.jpg"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/JiSeKa")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
            contentResolver.openOutputStream(uri)?.use { 
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) 
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
            }

            Toast.makeText(this, "💾 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
            resetToLiveMode() 
        } catch (e: Exception) {
            Log.e("CAMERA_DEBUG", "Failed to save image", e)
            Toast.makeText(this, "이미지 저장에 실패했습니다.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) viewFinder?.post { startCamera() } else finish()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        synchronized(bitmapLock) { lastCapturedBitmap?.recycle(); lastCapturedBitmap = null }
        displayedBitmap?.recycle(); displayedBitmap = null
        cameraExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
