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

        // 🌟 Step 1 & 2 영구 복구
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

        // =====================================================================
        // [3단계] OpenCV 윤곽선 탐지 
        // =====================================================================
        val thresh = Mat()
        val openKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 3.0))
        val closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(24.0, 5.0))
        
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
                    "Adaptive Thresh -> Open (5x3) -> Close (24x5)"
                ), screenRatio)
                it.pauseAndShowStep("3단계: 모폴로지 (Threshold+Morph)", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            // =====================================================================
            // 🚀 [4단계] Visual ID Tagging & Advanced Scoring System
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
            val roiArea = roiWidth * roiHeight.toDouble()

            // 💡 퍼포먼스 향상을 위해 모든 윤곽선의 BoundingRect 미리 캐싱
            val allRects = contours.map { Imgproc.boundingRect(it) }

            Log.d("PLATE_DEBUG", "========== STEP 4: CONTOUR EVALUATION START ==========")

            for (i in contours.indices) {
                val contour = contours[i]
                val boundingRect = allRects[i]
                val area = Imgproc.contourArea(contour)

                val contour2f = MatOfPoint2f(*contour.toArray())
                val rect = Imgproc.minAreaRect(contour2f)

                val w = rect.size.width
                val h = rect.size.height
                val ratio = max(w, h) / max(min(w, h), 1.0)
                val rectArea = w * h
                val rectangularity = area / max(rectArea, 1.0)
                
                val pts = arrayOf(Point(), Point(), Point(), Point())
                rect.points(pts)

                // 1. 너무 작은 노이즈 1차 필터링
                if (area < 500) {
                    areaTooSmallRects.add(Pair(i, pts))
                    contour2f.release()
                    continue 
                }

                // 2. 기본 형태(비율, 직사각형성) 미달 필터링 (측면 왜곡 대비 0.30으로 완화)
                if (ratio !in 1.5..7.5 || rectangularity < 0.30) {
                    rejectedRects.add(Pair(i, pts))
                    contour2f.release()
                    continue
                }

                // =====================================================================
                // 🚀 핵심 아이디어 1 & 2: 내부 글자 밀도(Density) 및 커버리지(Coverage) 계산
                // =====================================================================
                var validChildCount = 0
                var minChildX = Int.MAX_VALUE
                var maxChildX = Int.MIN_VALUE

                for (j in allRects.indices) {
                    if (i == j) continue
                    val child = allRects[j]
                    
                    // 후보 윤곽선 내부에 포함되는지 확인 (약간의 마진 허용)
                    if (child.x >= boundingRect.x - 2 && 
                        child.y >= boundingRect.y - 2 && 
                        child.x + child.width <= boundingRect.x + boundingRect.width + 2 && 
                        child.y + child.height <= boundingRect.y + boundingRect.height + 2) {
                        
                        // 글자의 형태학적 특징: 후보 상자 높이의 최소 25%, 너무 넓지 않고, 전체 면적을 다 덮지 않음
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

                // 밀도 점수 부여 (한국 번호판 7~8자 기준)
                val densityScore = when (validChildCount) {
                    in 5..12 -> 1500.0   // 번호판 확률 매우 높음
                    in 3..4 -> 600.0     // 일부분이 가려지거나 잘린 경우
                    in 13..25 -> 300.0   // 살짝 노이즈가 끼었지만 가능성 있음
                    else -> 0.0          // 텅 비어있거나(0) 구멍이 너무 많음(그릴)
                }

                // 커버리지 점수 부여 (내부 글자가 넓게 퍼져있을수록 진짜 번호판)
                var coverageScore = 0.0
                var coverageRatio = 0.0
                if (validChildCount >= 2) {
                    val coverageWidth = maxChildX - minChildX
                    coverageRatio = coverageWidth.toDouble() / boundingRect.width
                    if (coverageRatio > 0.4) {
                        coverageScore = coverageRatio * 1500.0
                    }
                }

                // =====================================================================
                // 🚀 핵심 아이디어 3: 오버플로우 기반 유연한 패널티 (Soft Penalty)
                // =====================================================================
                val borderMarginX = roiWidth * 0.03
                val borderMarginY = roiHeight * 0.05
                
                val overflowLeft = max(0.0, borderMarginX - boundingRect.x)
                val overflowRight = max(0.0, (boundingRect.x + boundingRect.width) - (roiWidth - borderMarginX))
                val overflowTop = max(0.0, borderMarginY - boundingRect.y)
                val overflowBottom = max(0.0, (boundingRect.y + boundingRect.height) - (roiHeight - borderMarginY))
                
                val overflowRatioX = (overflowLeft + overflowRight) / boundingRect.width
                val overflowRatioY = (overflowTop + overflowBottom) / boundingRect.height
                
                // 경계 침범 정도에 따른 감점 (살짝 닿으면 조금, 많이 닿으면 크게 감점)
                val overflowPenalty = -((overflowRatioX + overflowRatioY) * 1500.0)

                // 최악의 경우: ROI 전체를 덮어버리는 가짜 껍데기(Halo)는 즉시 치명타 부여
                val widthRatio = boundingRect.width.toDouble() / roiWidth
                val heightRatio = boundingRect.height.toDouble() / roiHeight
                val isBoundaryHalo = (widthRatio > 0.90 && heightRatio > 0.80)

                // =====================================================================
                // 최종 스코어링 합산
                // =====================================================================
                val areaRatio = area / roiArea
                val areaScore = when {
                    areaRatio < 0.05 -> -500.0   
                    areaRatio > 0.80 -> -500.0   
                    else -> 1000.0 * areaRatio   
                }

                val centerX = roiWidth / 2.0
                val centerY = roiHeight / 2.0
                val dist = hypot(rect.center.x - centerX, rect.center.y - centerY)
                val centerScore = 1000.0 - dist
                
                val rectScore = rectangularity * 5000.0
                val aspectScore = -abs(ratio - 3.5) * 400.0

                var score = areaScore + rectScore + aspectScore + (centerScore * 3.0) + 
                            densityScore + coverageScore + overflowPenalty

                if (isBoundaryHalo) {
                    score -= 4000.0 // 절대 우승 불가
                    borderTouchedRects.add(Pair(i, pts))
                }

                Log.d("PLATE_DEBUG", "ID[$i] score:${String.format("%.0f", score)} | dens:$validChildCount(+${String.format("%.0f", densityScore)}) | cov:${String.format("%.2f", coverageRatio)}(+${String.format("%.0f", coverageScore)}) | ovfPen:${String.format("%.0f", overflowPenalty)}")

                if (score > bestScore) {
                    if (bestContour != null) {
                        val prevContour2f = MatOfPoint2f(*bestContour!!.toArray())
                        val prevRect = Imgproc.minAreaRect(prevContour2f)
                        val prevPts = arrayOf(Point(), Point(), Point(), Point())
                        prevRect.points(prevPts)
                        lowScoreRects.add(Pair(-1, prevPts)) 
                        prevContour2f.release()
                    }
                    bestScore = score
                    bestContour = contour
                    bestRatio = ratio
                    bestRectangularity = rectangularity
                    Log.d("PLATE_DEBUG", " -> ✨ NEW BEST WINNER ✨ (ID: $i)")
                } else {
                    if (!isBoundaryHalo) lowScoreRects.add(Pair(i, pts))
                }
                contour2f.release()
            }

            Log.d("PLATE_DEBUG", "========== STEP 4: EVALUATION END ==========")

            // 🌟 맵 렌더링 함수
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

            val rawPoints = arrayOf(Point(), Point(), Point(), Point())
            val finalContour2f = MatOfPoint2f(*bestContour!!.toArray())
            val minRect = Imgproc.minAreaRect(finalContour2f)
            minRect.points(rawPoints)
            finalContour2f.release()

            debugListener?.let {
                val debugMat = tightMat.clone()
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                
                drawFullMap(debugBmp)

                val hudBmp = addDebugHUD(debugBmp, "Step 4: Density & Coverage Applied", listOf(
                    "Masterkey Score Activated!",
                    "Ratio: ${String.format("%.2f", bestRatio)} | Rect: ${String.format("%.2f", bestRectangularity)}",
                    "Winner Secured."
                ), screenRatio)
                it.pauseAndShowStep("4단계: 최종 스코어 맵", hudBmp)
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
            if (finalRatio < 1.5 || finalRatio > 7.5) {
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

                val statusLog = if (finalRatio < 1.5 || finalRatio > 7.5) "FALLBACK: Invalid ratio ($finalRatio)" else "SUCCESS: Valid ratio secured"

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

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
