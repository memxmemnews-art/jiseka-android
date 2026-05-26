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
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit

@OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {

    private var viewFinder: PreviewView? = null
    private var nativeBackgroundView: ImageView? = null
    private var nativeGuideView: NativeGuideView? = null
    private var modeSelectionLayout: LinearLayout? = null
    private var resultActionLayout: LinearLayout? = null
    private var btnCapture: Button? = null
    private var progressBar: ProgressBar? = null

    private var btnModePassenger: Button? = null
    private var btnModeFront: Button? = null
    private var btnModeDriver: Button? = null

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ThreadPoolExecutor

    private val bitmapLock = Any()
    private var lastCapturedBitmap: Bitmap? = null 
    private var displayedBitmap: Bitmap? = null    

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("CAMERA_DEBUG", "OpenCV initialization failed.")
        }

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder?.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
     
        nativeBackgroundView = findViewById(R.id.nativeBackgroundView)
        nativeBackgroundView?.scaleType = ImageView.ScaleType.CENTER_CROP
        
        nativeGuideView = findViewById(R.id.nativeGuideView)
        modeSelectionLayout = findViewById(R.id.modeSelectionLayout)
        resultActionLayout = findViewById(R.id.resultActionLayout)
        btnCapture = findViewById(R.id.btnCapture)
        progressBar = findViewById(R.id.progressBar)
   
        btnModePassenger = findViewById(R.id.btnModePassenger)
        btnModeFront = findViewById(R.id.btnModeFront)
        btnModeDriver = findViewById(R.id.btnModeDriver)

        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = ThreadPoolExecutor(
            1, 1, 0L, TimeUnit.MILLISECONDS,
            ArrayBlockingQueue(1), ThreadPoolExecutor.DiscardOldestPolicy()
        )

        setupUIListeners()
        resetToLiveMode()

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
        
        findViewById<Button>(R.id.btnRetry).setOnClickListener { resetToLiveMode() }
        findViewById<Button>(R.id.btnSave).setOnClickListener {
            displayedBitmap?.let { bmp -> saveBitmapToGallery(bmp) } 
                ?: Toast.makeText(this, "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
        }

        nativeGuideView?.onGuideDropListener = { mode ->
            triggerAnalysis(mode)
        }
    }

    private fun setMode(mode: String, clickedButton: Button) {
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
        
        synchronized(bitmapLock) {
            lastCapturedBitmap?.let {
                nativeBackgroundView?.setImageBitmap(it)
            }
        }
    }

    private fun resetToLiveMode() {
        btnCapture?.isEnabled = true

        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle()
        displayedBitmap = null

        synchronized(bitmapLock) {
            lastCapturedBitmap?.recycle()
            lastCapturedBitmap = null
        }

        viewFinder?.visibility = View.VISIBLE
        btnCapture?.visibility = View.VISIBLE
        
        btnModeFront?.let { setMode("FRONT", it) }
        
        nativeGuideView?.visibility = View.GONE
        modeSelectionLayout?.visibility = View.GONE
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
        btnCapture?.isEnabled = false
        progressBar?.visibility = View.VISIBLE
        btnCapture?.visibility = View.GONE

        val currentCapture = imageCapture ?: run {
            Toast.makeText(this, "카메라 초기화 실패", Toast.LENGTH_SHORT).show()
            resetToLiveMode()
            return
        }

        currentCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    try {
                        val rawBitmap = imageProxy.toUprightBitmap()
                        
                        val maxDim = 1920f
                        val scale = minOf(1f, maxDim / maxOf(rawBitmap.width, rawBitmap.height))
                        
                        val resizedBitmap = if (scale < 1f) {
                            Bitmap.createScaledBitmap(rawBitmap, (rawBitmap.width * scale).toInt(), (rawBitmap.height * scale).toInt(), true)
                        } else rawBitmap

                        val uprightBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
                        
                        if (resizedBitmap !== rawBitmap) resizedBitmap.recycle()
                        if (rawBitmap !== uprightBitmap) rawBitmap.recycle() 

                        synchronized(bitmapLock) {
                            lastCapturedBitmap?.recycle()
                            lastCapturedBitmap = uprightBitmap
                        }

                        runOnUiThread {
                            if (isFinishing || isDestroyed) return@runOnUiThread

                            viewFinder?.visibility = View.GONE
                            nativeBackgroundView?.setImageBitmap(uprightBitmap)
                            nativeBackgroundView?.visibility = View.VISIBLE
                            
                            nativeGuideView?.visibility = View.VISIBLE
                            modeSelectionLayout?.visibility = View.VISIBLE
                            
                            // 요청 반영: 촬영 직후에는 다시 촬영/다운로드 버튼을 노출하지 않음
                            resultActionLayout?.visibility = View.GONE
                            progressBar?.visibility = View.GONE
                            
                            // 🛠️ 자동 1회 분석 트리거 제거됨 (사용자가 드래그하기 전까지 원본 대기)
                        }
                    } catch (t: Throwable) {
                        runOnUiThread {
                            if (isFinishing || isDestroyed) return@runOnUiThread
                            Toast.makeText(this@MainActivity, "이미지 처리 오류", Toast.LENGTH_SHORT).show()
                            resetToLiveMode()
                        }
                    } finally {
                        imageProxy.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        if (isFinishing || isDestroyed) return@runOnUiThread
                        Toast.makeText(this@MainActivity, "카메라 캡처 실패", Toast.LENGTH_SHORT).show()
                        resetToLiveMode()
                    }
                }
            }
        )
    }

    private fun triggerAnalysis(mode: String) {
        val safeTargetBitmap = synchronized(bitmapLock) { 
            lastCapturedBitmap?.copy(Bitmap.Config.ARGB_8888, true) 
        } ?: return

        val guideView = nativeGuideView
        val viewW = nativeBackgroundView?.width ?: 1
        val viewH = nativeBackgroundView?.height ?: 1

        if (guideView == null || viewW <= 1) {
            safeTargetBitmap.recycle()
            return
        }

        runOnUiThread {
            if (!isFinishing && !isDestroyed) {
                progressBar?.visibility = View.VISIBLE
                resultActionLayout?.visibility = View.GONE 
            }
        }

        val uiCorners = guideView.getCorners()
        
        analysisExecutor.execute {
            try {
                var bmpCorners = mapUiToBitmap(uiCorners, viewW, viewH, safeTargetBitmap.width, safeTargetBitmap.height)

                val padding = 30
                val minX = maxOf(0, bmpCorners.minOf { it.x }.toInt() - padding)
                val minY = maxOf(0, bmpCorners.minOf { it.y }.toInt() - padding)
                val maxX = minOf(safeTargetBitmap.width, bmpCorners.maxOf { it.x }.toInt() + padding)
                val maxY = minOf(safeTargetBitmap.height, bmpCorners.maxOf { it.y }.toInt() + padding)
                
                val cropWidth = maxX - minX
                val cropHeight = maxY - minY

                if (cropWidth > 0 && cropHeight > 0) {
                    val croppedBitmap = Bitmap.createBitmap(safeTargetBitmap, minX, minY, cropWidth, cropHeight)

                    try {
                        val detected = PlateDetectionEngine.findPlateCorners(croppedBitmap)
                  
                        if (detected != null) {
                            bmpCorners = detected.map { PointF(it.x + minX, it.y + minY) }
                            
                            val newUiCorners = mapBitmapToUi(bmpCorners, viewW, viewH, safeTargetBitmap.width, safeTargetBitmap.height)
                            runOnUiThread {
                                if (isFinishing || isDestroyed) return@runOnUiThread
                                nativeGuideView?.setCorners(newUiCorners)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("CAMERA_DEBUG", "ML Kit / Engine 인식 실패", e)
                    } finally {
                        croppedBitmap.recycle()
                    }
                }

                val orderedCorners = orderCorners(bmpCorners)

                val resultMat = Mat()
                Utils.bitmapToMat(safeTargetBitmap, resultMat)
                
                applyMaskToMat(resultMat, orderedCorners)

                val resultBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(resultMat, resultBitmap)
                resultMat.release()

                runOnUiThread {
                    if (isFinishing || isDestroyed) {
                        resultBitmap.recycle()
                        return@runOnUiThread
                    }

                    nativeBackgroundView?.setImageDrawable(null)
                    displayedBitmap?.recycle()
                    displayedBitmap = resultBitmap
                    nativeBackgroundView?.setImageBitmap(resultBitmap)
                    
                    progressBar?.visibility = View.GONE
                    
                    // 요청 반영: 가림막 합성이 완벽히 끝난 시점에 버튼들 활성화
                    resultActionLayout?.visibility = View.VISIBLE
                }
            } catch (t: Throwable) {
                Log.e("CAMERA_DEBUG", "Analysis Error", t)
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    Toast.makeText(this@MainActivity, "가림막 처리 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                    progressBar?.visibility = View.GONE
                    resultActionLayout?.visibility = View.VISIBLE 
                }
            } finally {
                safeTargetBitmap.recycle()
            }
        }
    }

    private fun mapUiToBitmap(uiPts: List<PointF>, vw: Int, vh: Int, bw: Int, bh: Int): List<PointF> {
        val scale = Math.max(vw.toFloat() / bw, vh.toFloat() / bh)
        val dx = (vw - bw * scale) / 2f
        val dy = (vh - bh * scale) / 2f
        
        return uiPts.map { 
            val safeX = ((it.x - dx) / scale).coerceIn(0f, (bw - 1).toFloat())
            val safeY = ((it.y - dy) / scale).coerceIn(0f, (bh - 1).toFloat())
            PointF(safeX, safeY) 
        }
    }

    private fun mapBitmapToUi(bmpPts: List<PointF>, vw: Int, vh: Int, bw: Int, bh: Int): List<PointF> {
        val scale = Math.max(vw.toFloat() / bw, vh.toFloat() / bh)
        val dx = (vw - bw * scale) / 2f
        val dy = (vh - bh * scale) / 2f
        return bmpPts.map { PointF(it.x * scale + dx, it.y * scale + dy) }
    }

    private fun orderCorners(corners: List<PointF>): List<PointF> {
        if (corners.size != 4) return corners
        val tl = corners.minByOrNull { it.x + it.y } ?: corners[0]
        val br = corners.maxByOrNull { it.x + it.y } ?: corners[2]
        val tr = corners.minByOrNull { it.y - it.x } ?: corners[1]
        val bl = corners.maxByOrNull { it.y - it.x } ?: corners[3]
        return listOf(tl, tr, br, bl)
    }

    private fun applyMaskToMat(mat: Mat, corners: List<PointF>) {
        if (corners.size != 4) return
        var maskMat: Mat? = null; var contour: org.opencv.core.MatOfPoint? = null
        var blurredMask: Mat? = null; var coloredMask: Mat? = null; var alphaMat: Mat? = null
        val matChannels = ArrayList<Mat>(); val coloredChannels = ArrayList<Mat>()
        
        try {
            val pts = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }
            val maskColor = Scalar(0.0, 255.0, 0.0, 255.0) 

            maskMat = Mat.zeros(mat.size(), CvType.CV_8UC1)
            contour = org.opencv.core.MatOfPoint(*pts.toTypedArray())
            
            Imgproc.fillPoly(maskMat, listOf(contour), Scalar(255.0))

            blurredMask = Mat()
            Imgproc.GaussianBlur(maskMat, blurredMask, Size(15.0, 15.0), 5.0)

            coloredMask = Mat(mat.size(), mat.type(), maskColor)
            alphaMat = Mat()
            blurredMask.convertTo(alphaMat, CvType.CV_32F, 1.0 / 255.0)

            Core.split(mat, matChannels); Core.split(coloredMask, coloredChannels)

            for (i in 0 until 3) {
                var mcF: Mat? = null; var ccF: Mat? = null; var blendedF: Mat? = null
                var invAlpha: Mat? = null; var scalarMat: Mat? = null
                try {
                    val mc = matChannels[i]; val cc = coloredChannels[i]
                    mcF = Mat(); ccF = Mat()
                    mc.convertTo(mcF, CvType.CV_32F); cc.convertTo(ccF, CvType.CV_32F)

                    blendedF = Mat()
                    Core.multiply(ccF, alphaMat, ccF)

                    invAlpha = Mat(); scalarMat = Mat(alphaMat.size(), alphaMat.type(), Scalar(1.0))
                    Core.subtract(scalarMat, alphaMat, invAlpha)

                    Core.multiply(mcF, invAlpha, mcF); Core.add(ccF, mcF, blendedF)
                    blendedF.convertTo(matChannels[i], CvType.CV_8U)
                } finally {
                    mcF?.release(); ccF?.release(); blendedF?.release()
                    invAlpha?.release(); scalarMat?.release()
                }
            }
            Core.merge(matChannels, mat)
        } finally {
            maskMat?.release(); contour?.release()
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
            contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
            }

            Toast.makeText(this, "💾 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
            resetToLiveMode() 
        } catch (e: Exception) {
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
        nativeBackgroundView?.setImageDrawable(null)
        displayedBitmap?.recycle(); displayedBitmap = null
        cameraExecutor.shutdownNow(); analysisExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
