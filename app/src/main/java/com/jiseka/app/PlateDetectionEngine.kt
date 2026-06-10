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
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 50f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }
        val bgPaint = Paint().apply { color = Color.parseColor("#CC000000") }

        val padding = 30f
        val lineHeight = 60f
        val hudHeight = padding + lineHeight + (logs.size * lineHeight) + padding

        var maxTextWidth = paint.measureText(title)
        for (log in logs) {
            val logWidth = paint.measureText(log)
            if (logWidth > maxTextWidth) maxTextWidth = logWidth
        }

        val requiredWidth = max(original.width.toFloat(), maxTextWidth + padding * 2).toInt()
        val requiredHeight = (original.height + hudHeight).toInt()

        val combinedBmp = Bitmap.createBitmap(requiredWidth, requiredHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combinedBmp)
        
        canvas.drawRect(0f, 0f, requiredWidth.toFloat(), hudHeight, bgPaint)

        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        canvas.drawText(title, padding, padding + 40f, paint)

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        var currentY = padding + lineHeight + 40f
        for (log in logs) {
            canvas.drawText(log, padding, currentY, paint)
            currentY += lineHeight
        }

        val imageOffsetX = if (original.width < requiredWidth) (requiredWidth - original.width) / 2f else 0f
        canvas.drawBitmap(original, imageOffsetX, hudHeight, null)

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
        // 🚀 [1단계] 선 길이 기반 최신 번호판 비율(4.7:1) 동적 ROI 할당
        // =====================================================================
        val expectedWidth = lineLen * 1.2 
        val expectedHeight = expectedWidth / 4.7 

        val paddedLeft = (cx - expectedWidth / 2.0).toInt().coerceIn(0, fullMat.cols() - 1)
        val paddedTop = (cy - expectedHeight / 2.0).toInt().coerceIn(0, fullMat.rows() - 1)
        val paddedRight = (cx + expectedWidth / 2.0).toInt().coerceIn(1, fullMat.cols())
        val paddedBottom = (cy + expectedHeight / 2.0).toInt().coerceIn(1, fullMat.rows())

        val paddedRect = Rect(paddedLeft, paddedTop, paddedRight - paddedLeft, paddedBottom - paddedTop)
        
        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.line(debugMat, Point(p1x, p1y), Point(p2x, p2y), Scalar(255.0, 255.0, 0.0, 255.0), 8)
            Imgproc.rectangle(debugMat, paddedRect, Scalar(255.0, 0.0, 0.0, 255.0), 12)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val hudBmp = addDebugHUD(debugBmp, "Step 1: Ratio-Optimized ROI", listOf(
                "Line Length: ${String.format("%.1f", lineLen)} px",
                "Expected Width (1.2x): ${String.format("%.1f", expectedWidth)} px",
                "Expected Height (/4.7): ${String.format("%.1f", expectedHeight)} px",
                "Result: Tightly wrapped to modern plate ratio"
            ))
            
            it.pauseAndShowStep("1단계: 번호판 비율 적용 Padded ROI", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        val paddedMat = Mat(); val paddedGray = Mat()
        fullMat.submat(paddedRect).copyTo(paddedMat)
        fullGray.submat(paddedRect).copyTo(paddedGray)

        val roiCx = cx - paddedLeft
        val roiCy = cy - paddedTop

        // =====================================================================
        // [2단계] 수평 정렬
        // =====================================================================
        val angle = Math.toDegrees(Math.atan2(dy, dx))
        val rotMat = Imgproc.getRotationMatrix2D(Point(roiCx, roiCy), -angle, 1.0)

        val rotatedPaddedMat = Mat(); val rotatedPaddedGray = Mat()
        Imgproc.warpAffine(paddedMat, rotatedPaddedMat, rotMat, paddedMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(paddedGray, rotatedPaddedGray, rotMat, paddedGray.size(), Imgproc.INTER_LINEAR)

        // =====================================================================
        // 🚀 [3단계] OpenCV 내장 윤곽선(Contour) 탐지 알고리즘 적용
        // =====================================================================
        val thresh = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 7.0))
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            // ---------------------------------------------------------
            // 3-1. 가우시안 블러 (노이즈 제거)
            // ---------------------------------------------------------
            Imgproc.GaussianBlur(rotatedPaddedGray, thresh, Size(5.0, 5.0), 0.0)
            
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

            // ---------------------------------------------------------
            // 3-2. 적응형 이진화 (흑백 대비 극대화)
            // ---------------------------------------------------------
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

            // ---------------------------------------------------------
            // 3-3. 모폴로지 닫기 (글자와 테두리 뭉치기)
            // ---------------------------------------------------------
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

            // ---------------------------------------------------------
            // 3-4. 윤곽선 탐지 및 최종 사각형 도출
            // ---------------------------------------------------------
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

            val rectPoints = arrayOfNulls<Point>(4)
            if (bestContour != null && maxArea > 1000.0) { 
                val contour2f = MatOfPoint2f(*bestContour.toArray())
                val minRect = Imgproc.minAreaRect(contour2f)
                minRect.points(rectPoints)
                contour2f.release()
            } else {
                rectPoints[0] = Point(0.0, rotatedPaddedGray.rows().toDouble())
                rectPoints[1] = Point(0.0, 0.0)
                rectPoints[2] = Point(rotatedPaddedGray.cols().toDouble(), 0.0)
                rectPoints[3] = Point(rotatedPaddedGray.cols().toDouble(), rotatedPaddedGray.rows().toDouble())
            }

            debugListener?.let {
                val debugMat = rotatedPaddedMat.clone()
                val color = Scalar(0.0, 255.0, 0.0, 255.0)
                for (i in 0..3) {
                    Imgproc.line(debugMat, rectPoints[i]!!, rectPoints[(i + 1) % 4]!!, color, 4)
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
            // [4단계] 좌표 원복 (역회전 및 오프셋 더하기)
            // =====================================================================
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)

            val srcMat = MatOfPoint2f(*rectPoints.filterNotNull().toTypedArray())
            val dstMat = MatOfPoint2f()

            Core.transform(srcMat, dstMat, invRotMat)

            resultPoints = dstMat.toArray().map { 
                ImmutablePoint((it.x + paddedRect.x).toFloat(), (it.y + paddedRect.y).toFloat()) 
            }

            invRotMat.release(); srcMat.release(); dstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release(); kernel.release()
            contours.forEach { it.release() }; hierarchy.release()
            
            rotMat.release()
            paddedMat.release(); paddedGray.release()
            rotatedPaddedMat.release(); rotatedPaddedGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
