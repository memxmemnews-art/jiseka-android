package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(stageName: String, bitmap: Bitmap)
    }

    private fun addDebugHUD(original: Bitmap, title: String, logs: List<String>): Bitmap {
        val targetWidth = 1080f
        val scaleFactor = targetWidth / original.width.toFloat()
        
        val scaledImgWidth = targetWidth.toInt()
        val scaledImgHeight = (original.height * scaleFactor).toInt()
        val scaledImg = Bitmap.createScaledBitmap(original, scaledImgWidth, max(1, scaledImgHeight), true)

        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 45f 
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }
        val bgPaint = Paint().apply { color = Color.parseColor("#E6000000") }

        val padding = 30f
        val lineHeight = 60f
        val hudHeight = (padding + lineHeight + (logs.size * lineHeight) + padding).toInt()

        val combinedBmp = Bitmap.createBitmap(scaledImgWidth, hudHeight + scaledImgHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combinedBmp)
        
        canvas.drawRect(0f, 0f, scaledImgWidth.toFloat(), hudHeight.toFloat(), bgPaint)

        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        canvas.drawText(title, padding, padding + 45f, paint)

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        var currentY = padding + lineHeight + 45f
        for (log in logs) {
            canvas.drawText(log, padding, currentY, paint)
            currentY += lineHeight
        }

        canvas.drawBitmap(scaledImg, 0f, hudHeight.toFloat(), null)
        scaledImg.recycle()

        return combinedBmp
    }

    fun rescuePlateFromLine(
        fullBitmap: Bitmap, 
        startX: Float, startY: Float, 
        endX: Float, endY: Float,
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat()
        val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        var p1x = startX.toDouble(); var p1y = startY.toDouble()
        var p2x = endX.toDouble(); var p2y = endY.toDouble()
        if (p1x > p2x) {
            val tx = p1x; val ty = p1y; p1x = p2x; p1y = p2y; p2x = tx; p2y = ty
        }

        val dx = p2x - p1x
        val dy = p2y - p1y
        val lineLen = hypot(dx, dy)
        val cx = (p1x + p2x) / 2.0
        val cy = (p1y + p2y) / 2.0

        // =====================================================================
        // 🚀 [1단계] 넉넉한 정사각형 임시 ROI 확보 (회전 전 코너 잘림 방지)
        // =====================================================================
        val looseSize = lineLen * 1.5 
        val looseLeft = (cx - looseSize / 2.0).toInt().coerceIn(0, fullMat.cols() - 1)
        val looseTop = (cy - looseSize / 2.0).toInt().coerceIn(0, fullMat.rows() - 1)
        val looseRight = (cx + looseSize / 2.0).toInt().coerceIn(1, fullMat.cols())
        val looseBottom = (cy + looseSize / 2.0).toInt().coerceIn(1, fullMat.rows())

        val looseRect = Rect(looseLeft, looseTop, looseRight - looseLeft, looseBottom - looseTop)
        
        val looseMat = Mat(); val looseGray = Mat()
        fullMat.submat(looseRect).copyTo(looseMat)
        fullGray.submat(looseRect).copyTo(looseGray)

        val roiCx = cx - looseLeft
        val roiCy = cy - looseTop

        // =====================================================================
        // 🚀 [2단계] 수평 정렬 후 완벽한 4.7:1 비율로 타이트하게 자르기
        // =====================================================================
        val angle = Math.toDegrees(Math.atan2(dy, dx))
        val rotMat = Imgproc.getRotationMatrix2D(Point(roiCx, roiCy), -angle, 1.0)

        val rotatedLooseMat = Mat(); val rotatedLooseGray = Mat()
        Imgproc.warpAffine(looseMat, rotatedLooseMat, rotMat, looseMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(looseGray, rotatedLooseGray, rotMat, looseGray.size(), Imgproc.INTER_LINEAR)

        // 수평으로 반듯해진 이미지 위에서 타이트한 비율을 적용!
        val expectedWidth = lineLen * 1.2 
        val expectedHeight = expectedWidth / 4.7 

        val tightLeft = (roiCx - expectedWidth / 2.0).toInt().coerceIn(0, rotatedLooseMat.cols() - 1)
        val tightTop = (roiCy - expectedHeight / 2.0).toInt().coerceIn(0, rotatedLooseMat.rows() - 1)
        val tightRight = (roiCx + expectedWidth / 2.0).toInt().coerceIn(1, rotatedLooseMat.cols())
        val tightBottom = (roiCy + expectedHeight / 2.0).toInt().coerceIn(1, rotatedLooseMat.rows())

        val tightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)

        val tightMat = Mat(); val tightGray = Mat()
        rotatedLooseMat.submat(tightRect).copyTo(tightMat)
        rotatedLooseGray.submat(tightRect).copyTo(tightGray)

        debugListener?.let {
            val debugMat = rotatedLooseMat.clone()
            // 돌아간 전체 이미지 위에 타이트하게 잘린 영역을 파란색으로 표시
            Imgproc.rectangle(debugMat, tightRect, Scalar(0.0, 150.0, 255.0, 255.0), 4)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)

            val hudBmp = addDebugHUD(debugBmp, "Step 1 & 2: Level & Tight Ratio Crop", listOf(
                "Rotation Angle: ${String.format("%.1f", -angle)} deg",
                "Tight Width (1.2x): ${String.format("%.1f", expectedWidth)} px",
                "Tight Height (/4.7): ${String.format("%.1f", expectedHeight)} px",
                "Status: Corners perfectly preserved"
            ))
            
            it.pauseAndShowStep("1~2단계: 수평 정렬 및 비율 ROI 크롭", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [3단계] OpenCV 내장 윤곽선(Contour) 탐지 알고리즘 (tightGray 기준)
        // =====================================================================
        val thresh = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 7.0))
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            Imgproc.GaussianBlur(tightGray, thresh, Size(5.0, 5.0), 0.0)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-1: Gaussian Blur", listOf(
                    "Kernel Size: 5x5",
                    "Status: High-frequency noise reduced"
                ))
                it.pauseAndShowStep("3-1단계: 가우시안 블러", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 10.0)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-2: Adaptive Threshold", listOf(
                    "Block Size: 15, C: 10.0",
                    "Status: Foreground (Text/Borders) extracted"
                ))
                it.pauseAndShowStep("3-2단계: 적응형 이진화", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-3: Morphology Close", listOf(
                    "Kernel Size: 25x7 (Horizontal emphasis)",
                    "Status: Text and boundaries merged into blocks"
                ))
                it.pauseAndShowStep("3-3단계: 모폴로지 닫기", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var maxArea = -1.0
            var bestContour: MatOfPoint? = null
            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area > maxArea) {
                    maxArea = area
                    bestContour = contour
                }
            }

            val rectPoints = arrayOf(Point(), Point(), Point(), Point())
            if (bestContour != null && maxArea > 1000.0) { 
                val contour2f = MatOfPoint2f(*bestContour.toArray())
                val minRect = Imgproc.minAreaRect(contour2f)
                minRect.points(rectPoints)
                contour2f.release()
            } else {
                rectPoints[0] = Point(0.0, tightGray.rows().toDouble())
                rectPoints[1] = Point(0.0, 0.0)
                rectPoints[2] = Point(tightGray.cols().toDouble(), 0.0)
                rectPoints[3] = Point(tightGray.cols().toDouble(), tightGray.rows().toDouble())
            }

            debugListener?.let {
                val debugMat = tightMat.clone()
                val color = Scalar(0.0, 255.0, 0.0, 255.0)
                for (i in 0..3) {
                    Imgproc.line(debugMat, rectPoints[i], rectPoints[(i + 1) % 4], color, 4)
                }
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val hudBmp = addDebugHUD(debugBmp, "Step 3-4: Final Rectangle Extraction", listOf(
                    "Detected Plate Area: ${String.format("%.1f", maxArea)} px",
                    "Method: MinAreaRect on largest contour",
                    "Status: Ready to un-warp and mask"
                ))
                it.pauseAndShowStep("3-4단계: 최종 번호판 사각형 추출", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🚀 [4단계] 좌표 원복 (타이트 오프셋 -> 역회전 -> 임시 영역 오프셋)
            // =====================================================================
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)

            // 1. tightRect 안의 좌표를 rotatedLooseMat 좌표계로 변환
            val pointsInRotatedLoose = rectPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val srcMat = MatOfPoint2f(*pointsInRotatedLoose)
            val dstMat = MatOfPoint2f()

            // 2. 회전을 역으로 풀어 looseMat 좌표계로 복귀
            Core.transform(srcMat, dstMat, invRotMat)

            // 3. looseRect 오프셋을 더해 원본 사진(fullMat) 좌표계로 최종 복귀
            resultPoints = dstMat.toArray().map { 
                ImmutablePoint((it.x + looseRect.x).toFloat(), (it.y + looseRect.y).toFloat()) 
            }

            invRotMat.release(); srcMat.release(); dstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release(); kernel.release()
            contours.forEach { it.release() }; hierarchy.release()
            
            rotMat.release()
            looseMat.release(); looseGray.release()
            rotatedLooseMat.release(); rotatedLooseGray.release()
            tightMat.release(); tightGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
