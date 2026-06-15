package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
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
                "Rotation Angle: ${String.format("%.1f", angle)} deg"
            ), screenRatio)
            
            it.pauseAndShowStep("1~2단계: 실제 선분 기반 ROI 확정", hudBmp)
            debugMat.release(); debugBmp.recycle(); invRotMat.release(); srcPts.release(); dstPts.release()
        }

        val thresh = Mat()
        val openKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 3.0))
        
        // 💡 [수정됨] 4% 비율 + 하드 클램프 (그릴 과융합 방지용 보수적 커널)
        val dynamicKernelWidth = (tightGray.cols() * 0.04).coerceIn(20.0, 36.0)
        val closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(dynamicKernelWidth, 5.0))
        
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            Imgproc.medianBlur(tightGray, tightGray, 3)
            Imgproc.GaussianBlur(tightGray, thresh, Size(5.0, 5.0), 0.0)
            
            Imgproc.adaptiveThreshold(
                thresh, thresh, 255.0, 
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, 
                Imgproc.THRESH_BINARY_INV, 31, 7.0
            )
            
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, openKernel)
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, closeKernel)
            
            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(thresh, debugMat, Imgproc.COLOR_GRAY2RGBA)
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3: Morphology (Open + Close)", listOf(
                    "Adaptive Thresh -> Open (5x3) -> Close (${dynamicKernelWidth.toInt()}x5)"
                ), screenRatio)
                it.pauseAndShowStep("3단계: 모폴로지 (Threshold+Morph)", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            // =====================================================================
            // 🚀 [4단계] 앙상블 스코어링 (Plate Body 45% + Text Density 35% + Center 20%)
            // =====================================================================
            var bestScore = -Double.MAX_VALUE
            var bestContour: MatOfPoint? = null
            var bestRatio = 0.0
            var bestRectangularity = 0.0
            
            val rejectedRects = mutableListOf<Pair<Int, Array<Point>>>()
            val lowScoreRects = mutableListOf<Pair<Int, Array<Point>>>() 
            val borderTouchedRects = mutableListOf<Pair<Int, Array<Point>>>()
            val areaTooSmallRects = mutableListOf<Pair<Int, Array<Point>>>()
            
            val roiWidth = thresh.cols()
            val roiHeight = thresh.rows()
            val allRects = contours.map { Imgproc.boundingRect(it) }

            Log.d("PLATE_DEBUG", "========== STEP 4: CONTOUR EVALUATION ==========")

            for (i in contours.indices) {
                val contour = contours[i]
                val boundingRect = allRects[i]
                val area = Imgproc.contourArea(contour)

                val contour2f = MatOfPoint2f(*contour.toArray())
                val rect = Imgproc.minAreaRect(contour2f)

                val w = rect.size.width
                val h = rect.size.height
                val ratio = max(w, h) / max(min(w, h), 1.0)
                val rectangularity = area / max(w * h, 1.0)
                
                val pts = arrayOf(Point(), Point(), Point(), Point())
                rect.points(pts)

                if (area < 500) {
                    areaTooSmallRects.add(Pair(i, pts))
                    contour2f.release()
                    continue 
                }

                if (ratio !in 1.5..7.5 || rectangularity < 0.30) {
                    rejectedRects.add(Pair(i, pts))
                    contour2f.release()
                    continue
                }

                // 1. Text Density (자식 윤곽선 탐색)
                var validChildCount = 0
                var minChildX = Int.MAX_VALUE
                var maxChildX = Int.MIN_VALUE

                for (j in allRects.indices) {
                    if (i == j) continue
                    val child = allRects[j]
                    if (child.x >= boundingRect.x - 2 && child.y >= boundingRect.y - 2 && 
                        child.x + child.width <= boundingRect.x + boundingRect.width + 2 && 
                        child.y + child.height <= boundingRect.y + boundingRect.height + 2) {
                        
                        val isTallEnough = child.height >= boundingRect.height * 0.25
                        val isNotTooWide = child.width <= child.height * 2.5
                        val isNotTooLarge = child.area() <= boundingRect.area() * 0.3
                        
                        if (isTallEnough && isNotTooWide && isNotTooLarge) {
                            validChildCount++
                            minChildX = min(minChildX, child.x)
                            maxChildX = max(maxChildX, child.x + child.width)
                        }
                    }
                }

                var coverageRatio = 0.0
                if (validChildCount >= 2) {
                    val coverageWidth = maxChildX - minChildX
                    coverageRatio = coverageWidth.toDouble() / boundingRect.width
                }

                // 🌟 가중치 기반 스코어링 (총점 10000점 만점 기준)
                
                // [1] Plate Body (45%) -> 최대 4500점
                val plateBodyScore = (rectangularity * 4500.0) - (abs(ratio - 3.5) * 500.0)

                // [2] Text Density & Coverage (35%) -> 최대 3500점
                var textDensityScore = 0.0
                when (validChildCount) {
                    in 5..12 -> textDensityScore += 2000.0
                    in 3..4 -> textDensityScore += 800.0
                    in 13..25 -> textDensityScore += 400.0
                }
                if (validChildCount >= 2 && coverageRatio > 0.4) {
                    textDensityScore += (coverageRatio * 1500.0)
                }

                // [3] Center Bias (20%) -> 최대 2000점
                val centerX = roiWidth / 2.0
                val centerY = roiHeight / 2.0
                val dist = hypot(rect.center.x - centerX, rect.center.y - centerY)
                val maxDist = hypot(roiWidth / 2.0, roiHeight / 2.0)
                val centerBiasScore = max(0.0, 1.0 - (dist / maxDist)) * 2000.0

                // 오버플로우 패널티 및 Halo 방어
                val overflowRatioX = max(0.0, (roiWidth * 0.03 - boundingRect.x)) + max(0.0, (boundingRect.x + boundingRect.width) - (roiWidth - roiWidth * 0.03)) / boundingRect.width
                val overflowRatioY = max(0.0, (roiHeight * 0.05 - boundingRect.y)) + max(0.0, (boundingRect.y + boundingRect.height) - (roiHeight - roiHeight * 0.05)) / boundingRect.height
                val overflowPenalty = (overflowRatioX + overflowRatioY) * 1500.0

                val isBoundaryHalo = (boundingRect.width.toDouble() / roiWidth > 0.90 && boundingRect.height.toDouble() / roiHeight > 0.80)

                var finalScore = plateBodyScore + textDensityScore + centerBiasScore - overflowPenalty

                if (isBoundaryHalo) {
                    finalScore -= 5000.0 
                    borderTouchedRects.add(Pair(i, pts))
                }

                Log.d("PLATE_DEBUG", "ID[$i] total:${String.format("%.0f", finalScore)} | Body:${String.format("%.0f", plateBodyScore)} | Text:${String.format("%.0f", textDensityScore)} | Center:${String.format("%.0f", centerBiasScore)}")

                if (finalScore > bestScore) {
                    if (bestContour != null) {
                        val prevContour2f = MatOfPoint2f(*bestContour!!.toArray())
                        val prevRect = Imgproc.minAreaRect(prevContour2f)
                        val prevPts = arrayOf(Point(), Point(), Point(), Point())
                        prevRect.points(prevPts)
                        lowScoreRects.add(Pair(-1, prevPts)) 
                        prevContour2f.release()
                    }
                    bestScore = finalScore
                    bestContour = contour
                    bestRatio = ratio
                    bestRectangularity = rectangularity
                } else {
                    if (!isBoundaryHalo) lowScoreRects.add(Pair(i, pts))
                }
                contour2f.release()
            }

            val drawFullMap = { bmp: Bitmap -> 
                val canvasMat = Mat()
                Utils.bitmapToMat(bmp, canvasMat)
                
                val colorRed = Scalar(255.0, 0.0, 0.0, 255.0)       
                val colorOrange = Scalar(255.0, 165.0, 0.0, 255.0) 
                val colorBlue = Scalar(0.0, 0.0, 255.0, 255.0)     
                val colorYellow = Scalar(255.0, 255.0, 0.0, 255.0)   
                val colorGreen = Scalar(0.0, 255.0, 0.0, 255.0)    

                val drawRectWithId = { list: List<Pair<Int, Array<Point>>>, color: Scalar ->
                    for (item in list) {
                        val id = item.first
                        val pts = item.second
                        for (i in 0..3) Imgproc.line(canvasMat, pts[i], pts[(i + 1) % 4], color, 2)
                        if (id >= 0) { 
                            Imgproc.putText(canvasMat, "$id", Point(pts[1].x, pts[1].y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        }
                    }
                }

                drawRectWithId(rejectedRects, colorRed)
                drawRectWithId(lowScoreRects, colorOrange)
                drawRectWithId(borderTouchedRects, colorBlue)
                drawRectWithId(areaTooSmallRects, colorYellow)
                
                val rawPts = arrayOf(Point(), Point(), Point(), Point())
                if (bestContour != null) {
                    val finalContour2f = MatOfPoint2f(*bestContour!!.toArray())
                    val minRect = Imgproc.minAreaRect(finalContour2f)
                    minRect.points(rawPts)
                    finalContour2f.release()
                    for (i in 0..3) Imgproc.line(canvasMat, rawPts[i], rawPts[(i + 1) % 4], colorGreen, 5)
                    Imgproc.putText(canvasMat, "WINNER", Point(rawPts[1].x, rawPts[1].y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, colorGreen, 4)
                }
                
                Utils.matToBitmap(canvasMat, bmp)
                canvasMat.release()
            }

            if (bestContour == null) {
                debugListener?.let {
                    val debugMat = tightMat.clone()
                    val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(debugMat, debugBmp)
                    
                    drawFullMap(debugBmp)

                    val hudBmp = addDebugHUD(debugBmp, "Step 4: Scoring FAILED", listOf(
                        "No valid candidates survived.",
                        "Check Logcat (PLATE_DEBUG) for scores."
                    ), screenRatio)
                    it.pauseAndShowStep("4단계: 유효 윤곽선 없음", hudBmp)
                    debugMat.release(); debugBmp.recycle()
                }
                return null
            }

            debugListener?.let {
                val debugMat = tightMat.clone()
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                
                drawFullMap(debugBmp)

                val hudBmp = addDebugHUD(debugBmp, "Step 4: Scoring Applied", listOf(
                    "Masterkey Score Activated!",
                    "Ratio: ${String.format("%.2f", bestRatio)} | Rect: ${String.format("%.2f", bestRectangularity)}",
                    "Winner Secured."
                ), screenRatio)
                it.pauseAndShowStep("4단계: 최종 스코어 맵", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🚀 [5단계] 기하학 정렬: Convex Hull + approxPolyDP + Fallback
            // =====================================================================
            val contour2f = MatOfPoint2f(*bestContour!!.toArray())
            
            // 1. Convex Hull: 팽팽한 고무줄처럼 감싸 노이즈 억제
            val hull = MatOfInt()
            Imgproc.convexHull(bestContour, hull)
            
            val contourArray = bestContour!!.toArray()
            val hullPoints = hull.toArray().map { contourArray[it] }.toTypedArray()
            val hull2f = MatOfPoint2f(*hullPoints)

            // 2. approxPolyDP: 다각형 근사화로 원근감이 살아있는 4개의 꼭짓점 추론 시도
            val approx = MatOfPoint2f()
            val arcLength = Imgproc.arcLength(hull2f, true)
            Imgproc.approxPolyDP(hull2f, approx, arcLength * 0.03, true)

            val approxPts = approx.toArray()
            var rawPoints: Array<Point>

            // 3. 검증 및 Fallback 로직
            var geometryMethod = "Fallback (minAreaRect)"
            if (approxPts.size == 4) {
                rawPoints = approxPts
                geometryMethod = "approxPolyDP (Perspective)"
                Log.d("PLATE_DEBUG", "Step 5: approxPolyDP SUCCESS (4 points)")
            } else {
                val minRect = Imgproc.minAreaRect(contour2f)
                rawPoints = arrayOf(Point(), Point(), Point(), Point())
                minRect.points(rawPoints)
                Log.d("PLATE_DEBUG", "Step 5: approxPolyDP FAILED (${approxPts.size} pts) -> Fallback to minAreaRect")
            }

            // 메모리 해제
            hull.release()
            hull2f.release()
            approx.release()
            contour2f.release()

            // 4. 추출된 4개의 점 정렬 (TL, TR, BR, BL 순서 보장)
            val sum = rawPoints.map { it.x + it.y }
            val diff = rawPoints.map { it.x - it.y }
            val tl = rawPoints[sum.indexOf(sum.minOrNull()!!)]
            val br = rawPoints[sum.indexOf(sum.maxOrNull()!!)]
            val tr = rawPoints[diff.indexOf(diff.maxOrNull()!!)]
            val bl = rawPoints[diff.indexOf(diff.minOrNull()!!)]
            val orderedPoints = arrayOf(tl, tr, br, bl)

            // 5. 미세 확장 (Scale Up) - 번호판 틈새 노출 방지
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

            // 6. 회전(Rotation) 원복 변환 (Inverse Affine)
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)

            val pointsInRotatedLoose = expandedPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val srcMat = MatOfPoint2f(*pointsInRotatedLoose)
            val dstMat = MatOfPoint2f()

            Core.transform(srcMat, dstMat, invRotMat)

            val finalPts = dstMat.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }
            
            // 7. Width / Height 계산 및 기하학 무결성 검증 (Geometry Validation)
            val tlPt = finalPts[0]
            val trPt = finalPts[1]
            val brPt = finalPts[2]
            val blPt = finalPts[3]

            val widthTop = hypot(trPt.x - tlPt.x, trPt.y - tlPt.y)
            val widthBottom = hypot(brPt.x - blPt.x, brPt.y - blPt.y)
            val heightLeft = hypot(tlPt.x - blPt.x, tlPt.y - blPt.y)
            val heightRight = hypot(trPt.x - brPt.x, trPt.y - brPt.y)

            val maxWidth = max(widthTop, widthBottom)
            val maxHeight = max(heightLeft, heightRight).coerceAtLeast(1.0)
            val finalRatio = maxWidth / maxHeight

            Log.d("PLATE_DEBUG", "--- Step 5 Geometry Validation ---")
            Log.d("PLATE_DEBUG", "Method: $geometryMethod")
            Log.d("PLATE_DEBUG", "TL=$tlPt, TR=$trPt, BR=$brPt, BL=$blPt")
            Log.d("PLATE_DEBUG", "widthTop=${String.format("%.1f", widthTop)}, widthBottom=${String.format("%.1f", widthBottom)}")
            Log.d("PLATE_DEBUG", "heightLeft=${String.format("%.1f", heightLeft)}, heightRight=${String.format("%.1f", heightRight)}")
            Log.d("PLATE_DEBUG", "finalAspectRatio=${String.format("%.2f", finalRatio)}")

            var safeResultPts = finalPts
            var statusLog = "SUCCESS: Valid ratio secured ($geometryMethod)"

            // 최종 비율이 한국 번호판 규격을 심하게 벗어나면 최후의 안전장치 발동
            if (finalRatio < 1.5 || finalRatio > 7.5) {
                statusLog = "FINAL FALLBACK: Invalid ratio ($finalRatio)"
                Log.d("PLATE_DEBUG", "Step 5: FINAL FALLBACK TRIGGERED (Invalid Ratio: $finalRatio)")
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
                    Scalar(255.0, 0.0, 0.0, 255.0),   
                    Scalar(0.0, 255.0, 0.0, 255.0),   
                    Scalar(0.0, 0.0, 255.0, 255.0),   
                    Scalar(255.0, 255.0, 0.0, 255.0)  
                )
                val labels = arrayOf("TL", "TR", "BR", "BL")

                for (i in 0..3) {
                    Imgproc.line(debugMat, safeResultPts[i], safeResultPts[(i + 1) % 4], Scalar(255.0, 255.0, 255.0, 255.0), 4)
                    Imgproc.circle(debugMat, safeResultPts[i], 15, colors[i], -1)
                    Imgproc.putText(debugMat, labels[i], Point(safeResultPts[i].x - 20, safeResultPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, colors[i], 4)
                }

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val hudBmp = addDebugHUD(debugBmp, "Step 5: Geometry Validation", listOf(
                    "Order: TL -> TR -> BR -> BL",
                    "Final Aspect Ratio: ${String.format("%.2f", finalRatio)}",
                    statusLog
                ), screenRatio)
                
                it.pauseAndShowStep("5단계: 기하학 검증 및 마스킹 준비", hudBmp)
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
