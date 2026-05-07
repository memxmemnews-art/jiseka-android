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

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var resultImageView: ImageView
    private lateinit var cameraExecutor: ExecutorService
    
    private lateinit var plateTextureBmp: Bitmap
    
    private var imageCapture: ImageCapture? = null
    private val isCapturing = AtomicBoolean(false)
    private val recognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        private const val TARGET_MAX_DIMENSION = 1920 
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        resultImageView = findViewById(R.id.resultImageView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        plateTextureBmp = BitmapFactory.decodeResource(resources, R.drawable.plate_sample, options)

        findViewById<Button>(R.id.shutterBtn).setOnClickListener { captureAndProcess() }

        if (allPermissionsGranted()) startCamera() else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
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
                Log.e("JiSeKa", "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndProcess() {
        val capture = imageCapture ?: return
        if (!isCapturing.compareAndSet(false, true)) {
            Toast.makeText(this, "처리 중입니다.", Toast.LENGTH_SHORT).show()
            return
        }

        capture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                @SuppressLint("UnsafeOptInUsageError")
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    processEngineBackground(imageProxy)
                }
                override fun onError(exception: ImageCaptureException) {
                    isCapturing.set(false)
                    Log.e("JiSeKa", "Capture failed", exception)
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

        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
        options.inSampleSize = calculateInSampleSize(options, TARGET_MAX_DIMENSION, TARGET_MAX_DIMENSION)
        options.inJustDecodeBounds = false
        options.inPreferredConfig = Bitmap.Config.ARGB_8888

        var originalBmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) ?: run {
            rejectCaptureUI("이미지 디코딩에 실패했습니다.")
            return
        }

        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotatedBmp = Bitmap.createBitmap(originalBmp, 0, 0, originalBmp.width, originalBmp.height, matrix, true)
            if (rotatedBmp != originalBmp) originalBmp.recycle()
            originalBmp = rotatedBmp
        }

        val viewW = viewFinder.width.toFloat()
        val viewH = viewFinder.height.toFloat()
        val imgW = originalBmp.width.toFloat()
        val imgH = originalBmp.height.toFloat()
        val scale = maxOf(viewW / imgW, viewH / imgH)
        val offsetX = (viewW - imgW * scale) / 2f
        val offsetY = (viewH - imgH * scale) / 2f

        val guideViewW = viewW * 0.8f
        val guideViewH = guideViewW / 3f
        
        val guideRectImg = Rect(
            (((viewW - guideViewW) / 2f - offsetX) / scale).toInt(),
            (((viewH - guideViewH) / 2f - offsetY) / scale).toInt(),
            (((viewW + guideViewW) / 2f - offsetX) / scale).toInt(),
            (((viewH + guideViewH) / 2f - offsetY) / scale).toInt()
        )

        val corners = extractGeometryCorners(originalBmp, guideRectImg)

        if (corners == null || !validatePerspectiveStability(corners)) {
            rejectCaptureUI("안정적인 번호판 평면을 찾지 못했습니다.")
            return
        }

        val rectifiedBmp = rectifyToFlatPlate(originalBmp, corners)
        if (rectifiedBmp == null) {
            rejectCaptureUI("원근 보정 중 오류가 발생했습니다.")
            return
        }

        val inputImage = InputImage.fromBitmap(rectifiedBmp, 0)
        recognizer.process(inputImage)
            .addOnCompleteListener { task ->
                if (isDestroyed || isFinishing) {
                    rectifiedBmp.recycle() // 🚨 안전한 해제
                    return@addOnCompleteListener
                }

                var ocrValid = false
                if (task.isSuccessful) {
                    val text = task.result.text.replace(Regex("[^가-힣0-9]"), "")
                    ocrValid = text.length >= 3 
                }

                val finalScore = calculateStabilityScore(corners) + if (ocrValid) 0.2f else 0.0f

                if (finalScore >= 0.70f) {
                    cameraExecutor.execute {
                        val finalImage = overlayTextureWithLighting(originalBmp, corners)
                        runOnUiThread {
                            resultImageView.setImageBitmap(finalImage)
                            resultImageView.visibility = View.VISIBLE
                            isCapturing.set(false)
                        }
                    }
                } else {
                    rejectCaptureUI("평면 신뢰도가 부족합니다. (Score: $finalScore)")
                }
                
                // 🚨 메모리 누수 방지
                rectifiedBmp.recycle()
            }
    }

    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val (height: Int, width: Int) = options.outHeight to options.outWidth
        var inSampleSize = 1
        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }

    private fun rejectCaptureUI(msg: String) {
        runOnUiThread { 
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() 
            isCapturing.set(false)
        }
    }

    // ── 🚨 STEP 1: Overlap Ratio 반영 스코어링 및 완벽 정렬 ──
    private fun extractGeometryCorners(bitmap: Bitmap, guideRect: Rect): Array<PointF>? {
        var mat: Mat? = null; var roiMat: Mat? = null; var gray: Mat? = null
        var edges: Mat? = null; var kernel: Mat? = null; val contours = ArrayList<MatOfPoint>()
        
        try {
            mat = Mat()
            Utils.bitmapToMat(bitmap, mat)
            
            val margin = 30
            val roiX = max(0, guideRect.left - margin)
            val roiY = max(0, guideRect.top - margin)
            val roiW = min(guideRect.width() + margin * 2, mat.cols() - roiX)
            val roiH = min(guideRect.height() + margin * 2, mat.rows() - roiY)
            val roiRect = org.opencv.core.Rect(roiX, roiY, roiW, roiH)
            
            roiMat = Mat(mat, roiRect)
            gray = Mat()
            Imgproc.cvtColor(roiMat, gray, Imgproc.COLOR_RGBA2GRAY)

            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
            edges = Mat()
            Imgproc.Canny(gray, edges, 50.0, 150.0)
            
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

            val hierarchy = Mat()
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            hierarchy.release()

            val guideCenterX = guideRect.exactCenterX()
            val guideCenterY = guideRect.exactCenterY()
            val guideArea = guideRect.width() * guideRect.height()
            val maxValidDist = guideRect.width() * 0.4f

            var bestBox: Array<PointF>? = null
            var highestScore = 0.0

            for (contour in contours) {
                val contour2f = MatOfPoint2f(*contour.toArray())
                val minRect = Imgproc.minAreaRect(contour2f)
                
                val globalCenterX = minRect.center.x + roiX
                val globalCenterY = minRect.center.y + roiY
                val distToCenter = sqrt((globalCenterX - guideCenterX).pow(2) + (globalCenterY - guideCenterY).pow(2))
                if (distToCenter > maxValidDist) continue 

                val rw = max(minRect.size.width, minRect.size.height)
                val rh = min(minRect.size.width, minRect.size.height)
                if (rh == 0.0) continue
                
                val aspect = rw / rh
                if (aspect !in 2.0..6.5) continue 

                val rectArea = rw * rh
                val actualArea = Imgproc.contourArea(contour)
                
                // 🚨 겹침 비율(Overlap Ratio) 계산 (범퍼, 그릴 오검출 강력 차단)
                val candBound = minRect.boundingRect()
                val candRect = Rect(candBound.x + roiX, candBound.y + roiY, candBound.x + roiX + candBound.width, candBound.y + roiY + candBound.height)
                val intersect = Rect()
                intersect.setIntersect(guideRect, candRect)
                val overlapArea = max(0, intersect.width()) * max(0, intersect.height())
                val overlapRatio = (overlapArea.toDouble() / rectArea).coerceIn(0.0, 1.0)
                
                val areaScore = (actualArea / guideArea).coerceIn(0.0, 1.0)
                val centerScore = max(0.0, 1.0 - (distToCenter / maxValidDist))
                val aspectScore = if (aspect in 2.8..5.5) 1.0 else 0.6
                val rectangularityScore = (actualArea / rectArea).coerceIn(0.0, 1.0) 

                // Overlap Ratio에 0.25의 높은 가중치 부여
                val score = (areaScore * 0.20) + (centerScore * 0.25) + (overlapRatio * 0.25) + (aspectScore * 0.15) + (rectangularityScore * 0.15)

                if (score > highestScore && actualArea > (guideArea * 0.08)) {
                    highestScore = score
                    val boxPts = Mat()
                    Imgproc.boxPoints(minRect, boxPts)
                    bestBox = Array(4) { i ->
                        PointF(boxPts.get(i, 0)[0].toFloat() + roiX, boxPts.get(i, 1)[0].toFloat() + roiY)
                    }
                    boxPts.release()
                }
            }

            if (bestBox == null) return null

            // 🚨 CW/CCW 꼬임 방지를 위한 완벽한 정렬 (Signed Area)
            val cx = bestBox.map { it.x }.average().toFloat()
            val cy = bestBox.map { it.y }.average().toFloat()
            val sortedList = bestBox.sortedBy { pt -> atan2((pt.y - cy).toDouble(), (pt.x - cx).toDouble()) }.toMutableList()

            var signedArea = 0f
            for (i in 0..3) {
                val j = (i + 1) % 4
                signedArea += sortedList[i].x * sortedList[j].y - sortedList[j].x * sortedList[i].y
            }
            
            // Screen 좌표계(y가 아래로 증가) 기준 외적이 음수면 반시계 방향 -> 뒤집음
            if (signedArea < 0) {
                sortedList.reverse()
            }

            val tlIndex = sortedList.indices.minByOrNull { sortedList[it].x + sortedList[it].y } ?: 0
            Collections.rotate(sortedList, -tlIndex)

            return sortedList.toTypedArray()

        } finally {
            mat?.release(); roiMat?.release(); gray?.release(); edges?.release(); kernel?.release()
            contours.forEach { it.release() }
        }
    }

    private fun validatePerspectiveStability(pts: Array<PointF>): Boolean {
        val w1 = distance(pts[0], pts[1]); val w2 = distance(pts[3], pts[2])
        val h1 = distance(pts[0], pts[3]); val h2 = distance(pts[1], pts[2])
        val widthRatio = min(w1, w2) / max(w1, w2)
        val heightRatio = min(h1, h2) / max(h1, h2)
        if (widthRatio < 0.25f || heightRatio < 0.25f) return false
        return true
    }

    private fun calculateStabilityScore(pts: Array<PointF>): Float {
        val w1 = distance(pts[0], pts[1]); val w2 = distance(pts[3], pts[2])
        val widthRatio = min(w1, w2) / max(w1, w2)
        return ((widthRatio - 0.25f) / 0.55f).coerceIn(0f, 1f) * 0.8f
    }

    private fun distance(p1: PointF, p2: PointF): Float = sqrt((p1.x - p2.x).pow(2) + (p1.y - p2.y).pow(2))

    // ── STEP 2: 정면화 보정 ──
    private fun rectifyToFlatPlate(bitmap: Bitmap, corners: Array<PointF>): Bitmap? {
        var bgMat: Mat? = null; var rectifiedMat: Mat? = null
        try {
            bgMat = Mat()
            Utils.bitmapToMat(bitmap, bgMat)

            val srcPts = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
            val targetW = 400.0; val targetH = 100.0 
            val dstPts = MatOfPoint2f(Point(0.0, 0.0), Point(targetW, 0.0), Point(targetW, targetH), Point(0.0, targetH))

            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            rectifiedMat = Mat()
            Imgproc.warpPerspective(bgMat, rectifiedMat, transform, Size(targetW, targetH), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)

            val rectifiedBitmap = Bitmap.createBitmap(targetW.toInt(), targetH.toInt(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rectifiedMat, rectifiedBitmap)
            return rectifiedBitmap

        } catch (e: Exception) { return null
        } finally { bgMat?.release(); rectifiedMat?.release() }
    }

    // ── 🚨 STEP 3: 원본 조명 전사(Luminance Transfer) 및 합성 ──
    private fun overlayTextureWithLighting(background: Bitmap, corners: Array<PointF>): Bitmap {
        var bgMat: Mat? = null; var texMat: Mat? = null; var warpedTex: Mat? = null
        var mask: Mat? = null; var bgFloat: Mat? = null; var texFloat: Mat? = null
        var maskFloat: Mat? = null; var maskFloat3: Mat? = null; var invMaskFloat3: Mat? = null
        var blendedTex: Mat? = null; var blendedBg: Mat? = null; var finalFloat: Mat? = null
        var finalResultMat: Mat? = null; var erodeElement: Mat? = null
        
        var bgGray: Mat? = null; var bgGrayFloat: Mat? = null; var bgGrayFloatColor: Mat? = null

        try {
            bgMat = Mat()
            Utils.bitmapToMat(background, bgMat)
            Imgproc.cvtColor(bgMat, bgMat, Imgproc.COLOR_RGBA2RGB)

            texMat = Mat()
            Utils.bitmapToMat(plateTextureBmp, texMat)
            Imgproc.cvtColor(texMat, texMat, Imgproc.COLOR_RGBA2RGB)

            val texW = texMat.cols().toDouble(); val texH = texMat.rows().toDouble()
            val srcPts = MatOfPoint2f(Point(0.0, 0.0), Point(texW, 0.0), Point(texW, texH), Point(0.0, texH))
            val dstPtsList = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()
            val dstPts = MatOfPoint2f(*dstPtsList)

            val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            warpedTex = Mat(bgMat.size(), CvType.CV_8UC3, Scalar(0.0, 0.0, 0.0))
            Imgproc.warpPerspective(texMat, warpedTex, transform, bgMat.size(), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)

            // 🚨 자연스러움을 더하는 미세 블러 (CG 티 제거)
            Imgproc.GaussianBlur(warpedTex, warpedTex, Size(3.0, 3.0), 0.0)

            mask = Mat(bgMat.size(), CvType.CV_8UC1, Scalar(0.0))
            Imgproc.fillConvexPoly(mask, MatOfPoint(*dstPtsList), Scalar(255.0))
            
            erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.erode(mask, mask, erodeElement)

            val quadWidth = distance(corners[0], corners[1])
            var blurSize = (quadWidth / 40.0).toInt().coerceIn(5, 21)
            if (blurSize % 2 == 0) blurSize += 1 
            Imgproc.GaussianBlur(mask, mask, Size(blurSize.toDouble(), blurSize.toDouble()), 0.0)

            bgFloat = Mat(); texFloat = Mat(); maskFloat = Mat()
            bgMat.convertTo(bgFloat, CvType.CV_32FC3)
            warpedTex.convertTo(texFloat, CvType.CV_32FC3)
            mask.convertTo(maskFloat, CvType.CV_32FC1, 1.0 / 255.0)

            // 🚨 원본 조명 전사(Luminance Transfer) 로직
            bgGray = Mat()
            Imgproc.cvtColor(bgMat, bgGray, Imgproc.COLOR_RGB2GRAY)
            
            bgGrayFloat = Mat()
            bgGray.convertTo(bgGrayFloat, CvType.CV_32FC1, 1.0 / 255.0) // 0.0 ~ 1.0 광량 정규화
            
            bgGrayFloatColor = Mat()
            Imgproc.cvtColor(bgGrayFloat, bgGrayFloatColor, Imgproc.COLOR_GRAY2BGR) // 3채널 확장
            
            // 텍스처 색상과 원본 밝기를 곱하여 반사광, 그림자, 어둠을 그대로 가져옴
            Core.multiply(texFloat, bgGrayFloatColor, texFloat)

            // 알파 블렌딩 처리
            maskFloat3 = Mat()
            Imgproc.cvtColor(maskFloat, maskFloat3, Imgproc.COLOR_GRAY2BGR)

            invMaskFloat3 = Mat()
            Core.subtract(Mat(maskFloat3.size(), maskFloat3.type(), Scalar(1.0, 1.0, 1.0)), maskFloat3, invMaskFloat3)

            blendedTex = Mat(); blendedBg = Mat(); finalFloat = Mat()
            Core.multiply(texFloat, maskFloat3, blendedTex)
            Core.multiply(bgFloat, invMaskFloat3, blendedBg)
            Core.add(blendedTex, blendedBg, finalFloat)

            finalResultMat = Mat()
            finalFloat.convertTo(finalResultMat, CvType.CV_8UC3)
            val resultBitmap = Bitmap.createBitmap(finalResultMat.cols(), finalResultMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(finalResultMat, resultBitmap)

            return resultBitmap

        } finally {
            bgMat?.release(); texMat?.release(); warpedTex?.release(); mask?.release()
            bgFloat?.release(); texFloat?.release(); maskFloat?.release(); maskFloat3?.release()
            invMaskFloat3?.release(); blendedTex?.release(); blendedBg?.release(); erodeElement?.release()
            finalFloat?.release(); finalResultMat?.release()
            
            // 조명 전사 관련 메모리 해제
            bgGray?.release(); bgGrayFloat?.release(); bgGrayFloatColor?.release()
        }
    }
}
