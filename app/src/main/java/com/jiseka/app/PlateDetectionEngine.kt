package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(stageName: String, bitmap: Bitmap)
    }

    private fun drawTextWithWrap(canvas: Canvas, text: String, x: Float, y: Float, paint: Paint, maxWidth: Float, lineHeight: Float): Float {
        var currentY = y
        val originalTextSize = paint.textSize
        
        var textWidth = paint.measureText(text)
        if (textWidth > maxWidth) {
            paint.textSize = originalTextSize * 0.85f
            textWidth = paint.measureText(text)
        }

        if (textWidth > maxWidth) {
            val words = text.split(" ")
            var currentLine = ""

            for (word in words) {
                val testLine = if (currentLine.isEmpty()) word else "$currentLine $word"
                if (paint.measureText(testLine) > maxWidth && currentLine.isNotEmpty()) {
                    canvas.drawText(currentLine, x, currentY, paint)
                    currentLine = word
                    currentY += lineHeight
                } else {
                    currentLine = testLine
                }
            }
            if (currentLine.isNotEmpty()) {
                canvas.drawText(currentLine, x, currentY, paint)
                currentY += lineHeight
            }
        } else {
            canvas.drawText(text, x, currentY, paint)
            currentY += lineHeight
        }

        paint.textSize = originalTextSize
        return currentY
    }

    private fun addDebugHUD(original: Bitmap, title: String, logs: List<String>, screenRatio: Float): Bitmap {
        val canvasWidth = 1080
        val canvasHeight = (canvasWidth * screenRatio).toInt()
        val combinedBmp = Bitmap.createBitmap(canvasWidth, canvasHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combinedBmp)
        
        val bgPaint = Paint().apply { color = Color.parseColor("#E6000000") }
        canvas.drawRect(0f, 0f, canvasWidth.toFloat(), canvasHeight.toFloat(), bgPaint)

        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 45f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val paddingX = 120f
        val lineHeight = 65f
        val maxTextWidth = canvasWidth - (paddingX * 2)
        var currentY = 120f 
        
        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        currentY = drawTextWithWrap(canvas, title, paddingX, currentY, paint, maxTextWidth, lineHeight)

        currentY += 15f 

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        for (log in logs) {
            currentY = drawTextWithWrap(canvas, log, paddingX, currentY, paint, maxTextWidth, lineHeight)
        }

        val textBottom = currentY + 30f 
        val margin = 50f
        
        val maxImgWidth = canvasWidth - (margin * 2)
        val maxImgHeight = canvasHeight - textBottom - margin 

        if (maxImgHeight > 0) {
            val scaleX = maxImgWidth / original.width.toFloat()
            val scaleY = maxImgHeight / original.height.toFloat()
            val safeScaleFactor = min(scaleX, scaleY)

            val scaledWidth = (original.width * safeScaleFactor).toInt()
            val scaledHeight = (original.height * safeScaleFactor).toInt()
            
            val scaledImg = Bitmap.createScaledBitmap(original, max(1, scaledWidth), max(1, scaledHeight), true)
            
            val imgX = (canvasWidth - scaledWidth) / 2f
            val imgY = textBottom + (maxImgHeight - scaledHeight) / 2f

            val borderPaint = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 5f }
            canvas.drawRect(imgX - 2f, imgY - 2f, imgX + scaledWidth + 2f, imgY + scaledHeight + 2f, borderPaint)
            
            canvas.drawBitmap(scaledImg, imgX, imgY, null)
            scaledImg.recycle()
        }

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

        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

        val p1x = startX.toDouble()
        val p1y = startY.toDouble()
        val p2x = endX.toDouble()
        val p2y = endY.toDouble()

        val dx = p2x - p1x
        val dy = p2y - p1y
        val lineLen = hypot(dx, dy)
        val cx = (p1x + p2x) / 2.0
        val cy = (p1y + p2y) / 2.0

        // =====================================================================
        // [1단계] 넉넉한 정사각형 임시 ROI 확보 
        // =====================================================================
        val looseSize = lineLen * 2.0 
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
        // [2단계] 수평 정렬 후 실제 선분 좌표 기반 타이트 박스 생성
        // =====================================================================
        val angle = Math.toDegrees(Math.atan2(dy, dx))
        val rotMat = Imgproc.getRotationMatrix2D(Point(roiCx, roiCy), angle, 1.0)

        val rotatedLooseMat = Mat(); val rotatedLooseGray = Mat()
        Imgproc.warpAffine(looseMat, rotatedLooseMat, rotMat, looseMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(looseGray, rotatedLooseGray, rotMat, looseGray.size(), Imgproc.INTER_LINEAR)

        val srcLinePts = MatOfPoint2f(
            Point(p1x - looseLeft, p1y - looseTop),
            Point(p2x - looseLeft, p2y - looseTop)
        )
        val dstLinePts = MatOfPoint2f()
        Core.transform(srcLinePts, dstLinePts, rotMat)

        val rotLinePts = dstLinePts.toArray()
        val rotP1 = rotLinePts[0]
        val rotP2 = rotLinePts[1]

        val minX = min(rotP1.x, rotP2.x)
        val maxX = max(rotP1.x, rotP2.x)
        val rotLineLen = maxX - minX

        val marginX = rotLineLen * 0.10
        var tightLeft = (minX - marginX).toInt()
        var tightRight = (maxX + marginX).toInt()
        val tightWidth = tightRight - tightLeft

        val midY = (rotP1.y + rotP2.y) / 2.0
        val expectedHeight = max(tightWidth / 3.0, 80.0) 
        var tightTop = (midY - expectedHeight / 2.0).toInt()
        var tightBottom = (midY + expectedHeight / 2.0).toInt()

        tightLeft = tightLeft.coerceIn(0, rotatedLooseMat.cols() - 1)
        tightRight = tightRight.coerceIn(1, rotatedLooseMat.cols())
        tightTop = tightTop.coerceIn(0, rotatedLooseMat.rows() - 1)
        tightBottom = tightBottom.coerceIn(1, rotatedLooseMat.rows())

        val tightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)

        val tightMat = Mat(); val tightGray = Mat()
        rotatedLooseMat.submat(tightRect).copyTo(tightMat)
        rotatedLooseGray.submat(tightRect).copyTo(tightGray)

        srcLinePts.release(); dstLinePts.release()

        // 디버그 리스너 (1~2단계 생략가능하지만 유지)
        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.line(debugMat, Point(p1x, p1y), Point(p2x, p2y), Scalar(255.0, 255.0, 0.0, 255.0), 6)
            
            val tightRectPts = arrayOf(
                Point(tightLeft.toDouble(), tightTop.toDouble()),
                Point(tightRight.toDouble(), tightTop.toDouble()),
                Point(tightRight.toDouble(), tightBottom.toDouble()),
                Point(tightLeft.toDouble(), tightBottom.toDouble())
            )
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)
            val srcPts = MatOfPoint2f(*tightRectPts)
            val dstPts = MatOfPoint2f()
            Core.transform(srcPts, dstPts, invRotMat)
            val fullPts = dstPts.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }
            
            for (i in 0..3) {
                Imgproc.line(debugMat, fullPts[i], fullPts[(i + 1) % 4], Scalar(0.0, 200.0, 255.0, 255.0), 8)
            }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)

            val hudBmp = addDebugHUD(debugBmp, "Step 1 & 2: True Endpoint ROI Mapping", listOf(
                "dx: ${dx.toInt()} px, dy: ${dy.toInt()} px",
                "Rotation Angle: ${String.format("%.1f", angle)} deg",
                "Status: Angle sign & swap bugs fixed!"
            ), screenRatio)
            
            it.pauseAndShowStep("1~2단계: 실제 선분 기반 ROI 확정", hudBmp)
            debugMat.release(); debugBmp.recycle(); invRotMat.release(); srcPts.release(); dstPts.release()
        }

        // =====================================================================
        // [3단계] OpenCV 윤곽선 탐지
        // =====================================================================
        val thresh = Mat()
        
        val openKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 3.0))
        val closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(18.0, 5.0))
        
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            // 3-1: 가우시안 블러
            Imgproc.GaussianBlur(tightGray, thresh, Size(5.0, 5.0), 0.0)
            
            // 3-2: 적응형 이진화
            Imgproc.adaptiveThreshold(
                thresh, thresh, 255.0, 
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, 
                Imgproc.THRESH_BINARY_INV, 31, 7.0
            )
            
            // 3-3: 모폴로지 OPEN -> CLOSE
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, openKernel)
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, closeKernel)
            
            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            // =====================================================================
            // 🚀 [4단계] 모든 후보의 채점표를 투명하게 공개하는 디버그 강화판
            // =====================================================================
            var bestScore = -1.0
            var bestContour: MatOfPoint? = null
            var bestRatio = 0.0
            var bestRectangularity = 0.0
            
            val rejectedRects = mutableListOf<Array<Point>>()
            // 🌟 디버그용: 점수 미달로 탈락한 모든 박스 저장
            val lowScoreRects = mutableListOf<Array<Point>>() 
            val borderTouchedRects = mutableListOf<Array<Point>>()
            
            var evaluatedCount = 0
            
            val centerX = tightGray.cols() / 2.0
            val centerY = tightGray.rows() / 2.0

            Log.d("PLATE_DEBUG", "========== STEP 4: CONTOUR EVALUATION START ==========")

            for ((i, contour) in contours.withIndex()) {
                val area = Imgproc.contourArea(contour)
                val boundingRect = Imgproc.boundingRect(contour)

                val contour2f = MatOfPoint2f(*contour.toArray())
                val rect = Imgproc.minAreaRect(contour2f)

                val w = rect.size.width
                val h = rect.size.height
                val ratio = max(w, h) / max(min(w, h), 1.0)
                val rectArea = w * h
                val rectangularity = area / max(rectArea, 1.0)
                
                val pts = arrayOf(Point(), Point(), Point(), Point())
                rect.points(pts)

                val touchesBorder = 
                    boundingRect.x <= 2 || 
                    boundingRect.y <= 2 || 
                    boundingRect.x + boundingRect.width >= thresh.cols() - 2 || 
                    boundingRect.y + boundingRect.height >= thresh.rows() - 2

                // 🌟 [상세 로그] 모든 후보의 생얼과 탈락 사유를 기록
                Log.d("PLATE_DEBUG", "Contour[$i]: area=${String.format("%.1f", area)}, rectArea=${String.format("%.1f", rectArea)}, ratio=${String.format("%.2f", ratio)}, rectang=${String.format("%.2f", rectangularity)}, touchesBorder=$touchesBorder")

                if (touchesBorder) {
                    Log.d("PLATE_DEBUG", " -> REJECTED: Touches Border")
                    borderTouchedRects.add(pts)
                    contour2f.release()
                    continue
                }

                if (area < 500) {
                    Log.d("PLATE_DEBUG", " -> REJECTED: Area < 500")
                    contour2f.release()
                    continue 
                }

                evaluatedCount++

                if (ratio !in 1.8..7.0 || rectangularity < 0.60) {
                    Log.d("PLATE_DEBUG", " -> REJECTED: Ratio/Rectangularity Invalid")
                    rejectedRects.add(pts)
                    contour2f.release()
                    continue
                }

                val dist = hypot(rect.center.x - centerX, rect.center.y - centerY)
                val centerScore = 1000.0 - dist
                val score = (area * 0.3) + (rectangularity * 7000.0) + (centerScore * 3.0)

                Log.d("PLATE_DEBUG", " -> PASSED FILTER: Score Calculated = ${String.format("%.1f", score)}")

                if (score > bestScore) {
                    if (bestContour != null) {
                        val prevContour2f = MatOfPoint2f(*bestContour.toArray())
                        val prevRect = Imgproc.minAreaRect(prevContour2f)
                        val prevPts = arrayOf(Point(), Point(), Point(), Point())
                        prevRect.points(prevPts)
                        lowScoreRects.add(prevPts) // 1등에게 밀린 루저들 보관
                        prevContour2f.release()
                    }
                    bestScore = score
                    bestContour = contour
                    bestRatio = ratio
                    bestRectangularity = rectangularity
                    Log.d("PLATE_DEBUG", " -> ✨ NEW BEST WINNER ✨")
                } else {
                    Log.d("PLATE_DEBUG", " -> REJECTED: Score too low (Score: $score vs Best: $bestScore)")
                    lowScoreRects.add(pts)
                }
                contour2f.release()
            }

            Log.d("PLATE_DEBUG", "========== STEP 4: EVALUATION END ==========")

            if (bestContour == null) {
                debugListener?.let {
                    val debugMat = tightMat.clone()
                    val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(debugMat, debugBmp)
                    val hudBmp = addDebugHUD(debugBmp, "Step 4: Smart Scoring FAILED", listOf(
                        "Evaluated Contours: $evaluatedCount",
                        "Status: No valid plate detected (Null Returned)"
                    ), screenRatio)
                    it.pauseAndShowStep("4단계: 유효 윤곽선 없음 (탐지 실패)", hudBmp)
                    debugMat.release(); debugBmp.recycle()
                }
                return null
            }

            val rawPoints = arrayOf(Point(), Point(), Point(), Point())
            val finalContour2f = MatOfPoint2f(*bestContour.toArray())
            val minRect = Imgproc.minAreaRect(finalContour2f)
            minRect.points(rawPoints)
            finalContour2f.release()

            debugListener?.let {
                val debugMat = tightMat.clone()
                val colorRed = Scalar(255.0, 0.0, 0.0, 255.0)       // 규격 미달
                val colorOrange = Scalar(255.0, 165.0, 0.0, 255.0) // 규격은 통과했으나 점수 미달
                val colorBlue = Scalar(0.0, 0.0, 255.0, 255.0)     // 경계선 터치 탈락
                val colorGreen = Scalar(0.0, 255.0, 0.0, 255.0)    // 1등 합격

                // 🌟 화면에 탈락한 모든 후보들을 각기 다른 색상으로 그림
                for (pts in rejectedRects) {
                    for (i in 0..3) Imgproc.line(debugMat, pts[i], pts[(i + 1) % 4], colorRed, 1)
                }
                for (pts in lowScoreRects) {
                    for (i in 0..3) Imgproc.line(debugMat, pts[i], pts[(i + 1) % 4], colorOrange, 2)
                }
                for (pts in borderTouchedRects) {
                    for (i in 0..3) Imgproc.line(debugMat, pts[i], pts[(i + 1) % 4], colorBlue, 2)
                }
                for (i in 0..3) {
                    Imgproc.line(debugMat, rawPoints[i], rawPoints[(i + 1) % 4], colorGreen, 4)
                }

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 4: FULL COMPETITION MAP", listOf(
                    "Blue: Touched Border, Red: Invalid Ratio/Rect",
                    "Orange: Low Score, Green: WINNER!",
                    "Winner Score: ${String.format("%.1f", bestScore)}"
                ), screenRatio)
                it.pauseAndShowStep("4단계: 모든 후보 평가 결과 (색상별 분석)", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // [5단계] 정렬 및 미세 확장, 안전장치(Fallback) 적용
            // =====================================================================
            val sum = rawPoints.map { it.x + it.y }
            val diff = rawPoints.map { it.x - it.y }
            val tl = rawPoints[sum.indexOf(sum.minOrNull()!!)]
            val br = rawPoints[sum.indexOf(sum.maxOrNull()!!)]
            val tr = rawPoints[diff.indexOf(diff.maxOrNull()!!)]
            val bl = rawPoints[diff.indexOf(diff.minOrNull()!!)]
            val orderedPoints = arrayOf(tl, tr, br, bl)

            val rectCx = orderedPoints.map { it.x }.average()
            val rectCy = orderedPoints.map { it.y }.average()
            val scaleX = 1.03
            val scaleY = 1.05 
            val expandedPoints = Array(4) { Point() }
            for (i in 0..3) {
                val pt = orderedPoints[i]
                expandedPoints[i] = Point(
                    rectCx + (pt.x - rectCx) * scaleX,
                    rectCy + (pt.y - rectCy) * scaleY
                )
            }

            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)

            val pointsInRotatedLoose = expandedPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val srcMat = MatOfPoint2f(*pointsInRotatedLoose)
            val dstMat = MatOfPoint2f()

            Core.transform(srcMat, dstMat, invRotMat)

            val finalPts = dstMat.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }

            val finalWidth = hypot(finalPts[1].x - finalPts[0].x, finalPts[1].y - finalPts[0].y)
            val finalHeight = max(hypot(finalPts[3].x - finalPts[0].x, finalPts[3].y - finalPts[0].y), 1.0)
            val finalRatio = finalWidth / finalHeight

            var safeResultPts = finalPts
            if (finalRatio < 1.8 || finalRatio > 7.0) {
                val fallbackPts = arrayOf(
                    Point(tightLeft.toDouble(), tightTop.toDouble()),
                    Point(tightRight.toDouble(), tightTop.toDouble()),
                    Point(tightRight.toDouble(), tightBottom.toDouble()),
                    Point(tightLeft.toDouble(), tightBottom.toDouble())
                )
                val fallbackSrc = MatOfPoint2f(*fallbackPts)
                val fallbackDst = MatOfPoint2f()
                Core.transform(fallbackSrc, fallbackDst, invRotMat)
                safeResultPts = fallbackDst.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }
                fallbackSrc.release(); fallbackDst.release()
            }

            debugListener?.let {
                val debugMat = fullMat.clone()
                
                val colors = arrayOf(
                    Scalar(255.0, 0.0, 0.0, 255.0),   // TL
                    Scalar(0.0, 255.0, 0.0, 255.0),   // TR
                    Scalar(0.0, 0.0, 255.0, 255.0),   // BR
                    Scalar(255.0, 255.0, 0.0, 255.0)  // BL
                )
                val labels = arrayOf("TL", "TR", "BR", "BL")

                for (i in 0..3) {
                    Imgproc.line(debugMat, safeResultPts[i], safeResultPts[(i + 1) % 4], Scalar(255.0, 255.0, 255.0, 255.0), 4)
                    Imgproc.circle(debugMat, safeResultPts[i], 15, colors[i], -1)
                    Imgproc.putText(debugMat, labels[i], Point(safeResultPts[i].x - 20, safeResultPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, colors[i], 4)
                }

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val statusLog = if (finalRatio < 1.8 || finalRatio > 7.0) "FALLBACK: Invalid ratio ($finalRatio)" else "SUCCESS: Valid ratio secured"

                val hudBmp = addDebugHUD(debugBmp, "Step 5: Final Evaluation & Output", listOf(
                    "Order: TL -> TR -> BR -> BL",
                    "Final Aspect Ratio: ${String.format("%.1f", finalRatio)}",
                    statusLog
                ), screenRatio)
                
                it.pauseAndShowStep("5단계: 최종 비율 평가 및 마스킹 준비", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = safeResultPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }

            invRotMat.release(); srcMat.release(); dstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release(); openKernel.release(); closeKernel.release()
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
