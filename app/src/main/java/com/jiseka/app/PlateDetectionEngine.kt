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
        // [3단계] OpenCV 윤곽선 탐지 (가우시안, 이진화, 모폴로지)
        // =====================================================================
        val thresh = Mat()
        val kernelX = max(12.0, tightRect.width * 0.06) 
        val kernelY = max(3.0, tightRect.height * 0.05)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelX, kernelY))
        
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
                ), screenRatio)
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
                    "Status: Foreground extracted"
                ), screenRatio)
                it.pauseAndShowStep("3-2단계: 적응형 이진화", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-3: Morphology Close (Reduced)", listOf(
                    "Kernel Size: ${kernelX.toInt()} x ${kernelY.toInt()}",
                    "Status: Bleeding into bumper prevented"
                ), screenRatio)
                it.pauseAndShowStep("3-3단계: 커널 축소 모폴로지 닫기", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            // 🚀 [추가된 부분] 가장자리 테두리를 검은색으로 칠해 거대한 바운딩 박스 생성 방지 (Border Clearing)
            Imgproc.rectangle(
                thresh, 
                Point(0.0, 0.0), 
                Point(thresh.cols().toDouble() - 1, thresh.rows().toDouble() - 1), 
                Scalar(0.0), 
                10 // 테두리 두께 10px 마스킹
            )

            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            // =====================================================================
            // 🚀 [4단계] 기하학적 스코어링 (시각적 필터링 디버깅 추가)
            // =====================================================================
            var bestScore = -1.0
            var bestContour: MatOfPoint? = null
            var bestRatio = 0.0
            var bestRectangularity = 0.0
            
            val rejectedRects = mutableListOf<Array<Point>>()
            var evaluatedCount = 0

            // 🚀 [추가된 부분] ROI 전체 캔버스의 면적 계산 (이 면적의 85%를 넘으면 테두리 노이즈로 간주)
            val imageTotalArea = tightGray.cols() * tightGray.rows().toDouble()

            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area < 500) continue 

                // 🚀 [추가된 부분] 거대한 윤곽선 탈락 로직 (Area Limit Filtering)
                // 만약 윤곽선 면적이 이미지 전체 면적의 85%를 넘어간다면, 이는 화면 테두리를 둘러싼 노이즈이므로 즉시 거부(빨간 박스) 처리합니다.
                if (area > imageTotalArea * 0.85) {
                    val contour2f = MatOfPoint2f(*contour.toArray())
                    val rect = Imgproc.minAreaRect(contour2f)
                    val pts = arrayOf(Point(), Point(), Point(), Point(), Point())
                    rect.points(pts)
                    rejectedRects.add(pts)
                    contour2f.release()
                    continue
                }

                evaluatedCount++

                val contour2f = MatOfPoint2f(*contour.toArray())
                val rect = Imgproc.minAreaRect(contour2f)

                val w = rect.size.width
                val h = rect.size.height
                val ratio = max(w, h) / max(min(w, h), 1.0)
                val rectArea = w * h
                val rectangularity = area / max(rectArea, 1.0)

                contour2f.release()

                val pts = arrayOf(Point(), Point(), Point(), Point())
                rect.points(pts)

                // 조건 미달: 빨간 박스행
                if (ratio !in 2.5..6.5 || rectangularity < 0.60) {
                    rejectedRects.add(pts)
                    continue
                }

                val score = area * 0.5 + rectangularity * 10000

                if (score > bestScore) {
                    // 기존 1등도 밀려났으므로 빨간 박스행
                    if (bestContour != null) {
                        val prevContour2f = MatOfPoint2f(*bestContour.toArray())
                        val prevRect = Imgproc.minAreaRect(prevContour2f)
                        val prevPts = arrayOf(Point(), Point(), Point(), Point())
                        prevRect.points(prevPts)
                        rejectedRects.add(prevPts)
                        prevContour2f.release()
                    }
                    bestScore = score
                    bestContour = contour
                    bestRatio = ratio
                    bestRectangularity = rectangularity
                } else {
                    // 조건은 통과했으나 점수가 낮아 탈락
                    rejectedRects.add(pts)
                }
            }

            val rawPoints = arrayOf(Point(), Point(), Point(), Point())
            if (bestContour != null) { 
                val contour2f = MatOfPoint2f(*bestContour.toArray())
                val minRect = Imgproc.minAreaRect(contour2f)
                minRect.points(rawPoints)
                contour2f.release()
            } else {
                rawPoints[0] = Point(0.0, tightGray.rows().toDouble())
                rawPoints[1] = Point(0.0, 0.0)
                rawPoints[2] = Point(tightGray.cols().toDouble(), 0.0)
                rawPoints[3] = Point(tightGray.cols().toDouble(), tightGray.rows().toDouble())
            }

            // 🚀 [디버깅] 빨간색(탈락) vs 초록색(합격) 시각화
            debugListener?.let {
                val debugMat = tightMat.clone()
                val colorRed = Scalar(255.0, 0.0, 0.0, 255.0)
                val colorGreen = Scalar(0.0, 255.0, 0.0, 255.0)
                
                // 탈락한 덩어리들을 빨간색으로 그림
                for (pts in rejectedRects) {
                    for (i in 0..3) {
                        Imgproc.line(debugMat, pts[i], pts[(i + 1) % 4], colorRed, 2)
                    }
                }
                
                // 최종 합격한 덩어리만 굵은 초록색으로 그림
                for (i in 0..3) {
                    Imgproc.line(debugMat, rawPoints[i], rawPoints[(i + 1) % 4], colorGreen, 6)
                }

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 4: Smart Scoring & Filtering", listOf(
                    "Evaluated Contours: $evaluatedCount",
                    "Winner Ratio: ${String.format("%.2f", bestRatio)} (Valid: 2.5~6.5)",
                    "Winner Rectangularity: ${String.format("%.2f", bestRectangularity)} (Valid: >0.60)",
                    "Status: Red=Rejected, Green=Selected!"
                ), screenRatio)
                it.pauseAndShowStep("4단계: 필터링 시각화 (Red vs Green)", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🚀 [5단계] 정렬 및 미세 확장, 안전장치(Fallback) 적용
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
            if (finalRatio < 2.0 || finalRatio > 7.0) {
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

            // 🚀 최종 마스킹 준비 상태 디버그 확인
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

                val statusLog = if (finalRatio < 2.0 || finalRatio > 7.0) "FALLBACK: Invalid ratio ($finalRatio)" else "SUCCESS: Valid ratio secured"

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
