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
            textSize = 38f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val paddingX = 80f
        val lineHeight = 55f
        val maxTextWidth = canvasWidth - (paddingX * 2)
        var currentY = 100f 
        
        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        paint.textSize = 45f
        currentY = drawTextWithWrap(canvas, title, paddingX, currentY, paint, maxTextWidth, lineHeight)

        currentY += 20f 

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        paint.textSize = 35f
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

            val borderPaint = Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 6f }
            canvas.drawRect(imgX - 3f, imgY - 3f, imgX + scaledWidth + 3f, imgY + scaledHeight + 3f, borderPaint)
            
            canvas.drawBitmap(scaledImg, imgX, imgY, null)
            scaledImg.recycle()
        }

        return combinedBmp
    }

    fun rescuePlateFromPoint(
        fullBitmap: Bitmap, 
        touchX: Float, touchY: Float, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat(); val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

        val cx = touchX.toDouble()
        val cy = touchY.toDouble()

        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.circle(debugMat, Point(cx, cy), 20, Scalar(255.0, 0.0, 0.0, 255.0), -1)
            Imgproc.circle(debugMat, Point(cx, cy), 25, Scalar(255.0, 255.0, 255.0, 255.0), 4)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 1: User Touch Point", listOf(
                "Action: 터치 좌표 수신 완료",
                "Touch Point X: ${cx.toInt()} px", 
                "Touch Point Y: ${cy.toInt()} px",
                "Resolution: ${fullMat.cols()} x ${fullMat.rows()}"
            ), screenRatio)
            it.pauseAndShowStep("1단계: 터치 좌표 매핑", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 2] 초기 ROI 영역 설정 (🌟 2:1에서 3:1 비율로 수정됨)
        // =====================================================================
        val roiWidth = (fullMat.cols() * 0.28).toInt() // 가로폭 고해상도 대응 (28%)
        val roiHeight = (roiWidth / 3.0).toInt()       // 🌟 Fix: 가로세로 3:1 비율 최적화 크기로 조정

        val looseLeft = (cx - roiWidth / 2.0).toInt().coerceIn(0, fullMat.cols() - 1)
        val looseTop = (cy - roiHeight / 2.0).toInt().coerceIn(0, fullMat.rows() - 1)
        val looseRight = (cx + roiWidth / 2.0).toInt().coerceIn(1, fullMat.cols())
        val looseBottom = (cy + roiHeight / 2.0).toInt().coerceIn(1, fullMat.rows())

        val looseRect = Rect(looseLeft, looseTop, looseRight - looseLeft, looseBottom - looseTop)
        
        val looseMat = Mat(); val looseGray = Mat()
        fullMat.submat(looseRect).copyTo(looseMat)
        fullGray.submat(looseRect).copyTo(looseGray)

        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.rectangle(debugMat, looseRect, Scalar(0.0, 255.0, 0.0, 255.0), 8)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 2: Optimized ROI (Ratio 3:1)", listOf(
                "Fix: 밸런스를 고려하여 수직 높이를 3:1 구조로 조정 완료",
                "ROI Width: $roiWidth px (화면 가로폭의 28%)",
                "ROI Height: $roiHeight px (3:1 비율 반영)",
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 8% 탐색 구역(Loose ROI) 설정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 3 & 4] 글자 탐색을 위한 1차 이진화 및 모폴로지
        // =====================================================================
        val thresh = Mat()
        Imgproc.medianBlur(looseGray, looseGray, 3)
        Imgproc.GaussianBlur(looseGray, thresh, Size(5.0, 5.0), 0.0)
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, 7.0)

        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        
        // =====================================================================
        // 🚀 [Step 5] 문자 중심점 및 기울기 도출
        // =====================================================================
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect)
        val charList = mutableListOf<CharData>()

        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)

            if (ratio in 1.2..4.2 && area > 120 && rect.height >= 40 && area < looseRect.area() * 0.08) {
                val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
                charList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect))
            }
        }
        tempContours.forEach { it.release() }; tempHierarchy.release(); tempOpen.release(); tempClose.release()

        var angle = 0.0
        var tightLeft = 0; var tightRight = looseRect.width
        var tightTop = 0; var tightBottom = looseRect.height

        if (charList.size >= 2) {
            val pointsMat = MatOfPoint2f(*charList.map { it.center }.toTypedArray())
            val line = Mat()
            Imgproc.fitLine(pointsMat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
            val vx = line.get(0, 0)[0]; val vy = line.get(1, 0)[0]
            angle = Math.toDegrees(Math.atan2(vy, vx))
            
            val tempRotMat = Imgproc.getRotationMatrix2D(Point(looseRect.width / 2.0, looseRect.height / 2.0), angle, 1.0)
            val dstCenterPts = MatOfPoint2f()
            Core.transform(pointsMat, dstCenterPts, tempRotMat)
            
            val rotCenters = dstCenterPts.toArray()
            val minX = rotCenters.minOf { it.x }
            val maxX = rotCenters.maxOf { it.x }
            val avgY = rotCenters.map { it.y }.average()
            val avgH = charList.map { it.height }.average()
            val textSpreadWidth = maxX - minX
            
            val expectedHeight = max(avgH * 4.5, 280.0).coerceAtMost(looseRect.height.toDouble()) 
            val marginX = max(textSpreadWidth * 0.25, avgH * 3.0)

            tightLeft = (minX - marginX).toInt()
            tightRight = (maxX + marginX).toInt()
            tightTop = (avgY - expectedHeight / 2.0).toInt()
            tightBottom = (avgY + expectedHeight / 2.0).toInt()
            
            debugListener?.let {
                val debugMat = looseMat.clone()
                val x0 = line.get(2, 0)[0]; val y0 = line.get(3, 0)[0]
                val pt1 = Point(x0 - vx * 1000, y0 - vy * 1000)
                val pt2 = Point(x0 + vx * 1000, y0 + vy * 1000)
                Imgproc.line(debugMat, pt1, pt2, Scalar(0.0, 255.0, 0.0, 255.0), 3)
                for (charData in charList) Imgproc.circle(debugMat, charData.center, 5, Scalar(255.0, 0.0, 0.0, 255.0), -1)
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3~5: Cleaned Text Fitting", listOf(
                    "Valid Characters Found: ${charList.size} (그릴 무늬 대폭 제거)",
                    "Recalculated Avg Height: ${String.format("%.1f", avgH)} px",
                    "Fitted Line Angle: ${String.format("%.2f", angle)} deg",
                    "Status: 세부 탐색 박스 최소 높이를 280px로 잠금(Clamp)"
                ), screenRatio)
                it.pauseAndShowStep("3~5단계: 문자 탐색 및 기울기 계산", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }
            pointsMat.release(); line.release(); tempRotMat.release(); dstCenterPts.release()
        }

        // =====================================================================
        // 🚀 [Step 6 & 7] 회전 및 넉넉한 Tight ROI 추출
        // =====================================================================
        val rotMat = Imgproc.getRotationMatrix2D(Point(looseRect.width / 2.0, looseRect.height / 2.0), angle, 1.0)
        val rotatedLooseMat = Mat(); val rotatedLooseGray = Mat()
        Imgproc.warpAffine(looseMat, rotatedLooseMat, rotMat, looseMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(looseGray, rotatedLooseGray, rotMat, looseGray.size(), Imgproc.INTER_LINEAR)

        tightLeft = tightLeft.coerceIn(0, rotatedLooseMat.cols() - 1)
        tightRight = tightRight.coerceIn(1, rotatedLooseMat.cols())
        tightTop = tightTop.coerceIn(0, rotatedLooseMat.rows() - 1)
        tightBottom = tightBottom.coerceIn(1, rotatedLooseMat.rows())

        val tightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)
        val tightMat = Mat(); val tightGray = Mat()
        rotatedLooseMat.submat(tightRect).copyTo(tightMat)
        rotatedLooseGray.submat(tightRect).copyTo(tightGray)

        debugListener?.let {
            val debugMat = rotatedLooseMat.clone()
            Imgproc.rectangle(debugMat, tightRect, Scalar(255.0, 165.0, 0.0, 255.0), 6)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 6~7: True Plate Bound ROI", listOf(
                "Image leveled to 0 degrees.",
                "Tight ROI size: ${tightRect.width} x ${tightRect.height} (정상 비율 복구 완료)",
                "Ready for Outer Edge Tracking."
            ), screenRatio)
            it.pauseAndShowStep("6~7단계: 수평 회전 및 넉넉한 탐색 영역 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 8 & 9] 진짜 테두리 찾기 (Canny Edge + Dilate)
        // =====================================================================
        val edges = Mat()
        val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, Size(5.0, 5.0))
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            Imgproc.GaussianBlur(tightGray, tightGray, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(tightGray, edges, 50.0, 160.0) 
            Imgproc.dilate(edges, edges, dilateKernel)

            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(edges, debugMat, Imgproc.COLOR_GRAY2RGBA)
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 8~9: Canny Edge Map", listOf(
                    "Method: Canny(50, 160) + Dilate",
                    "Notice: 번호판 외곽 테두리 선이 상하 컷팅 없이 완벽히 노출됨."
                ), screenRatio)
                it.pauseAndShowStep("8~9단계: 번호판 물리적 테두리(Edge) 추출", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🚀 [Step 10] 윤곽선 계층(Hierarchy) 구조 탐색 및 채점
            // =====================================================================
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestScore = -Double.MAX_VALUE
            var bestContour: MatOfPoint? = null
            var bestRatio = 0.0
            var bestChildCount = 0
            
            val rejectedRects = mutableListOf<Pair<Int, Array<Point>>>()
            val lowScoreRects = mutableListOf<Pair<Int, Array<Point>>>() 

            for (i in contours.indices) {
                val contour = contours[i]
                val contour2f = MatOfPoint2f(*contour.toArray())
                val rect = Imgproc.minAreaRect(contour2f)

                val w = rect.size.width
                val h = rect.size.height
                val ratio = max(w, h) / max(min(w, h), 1.0)
                val boxArea = w * h
                val perimeter = Imgproc.arcLength(contour2f, true)
                val expectedPerimeter = 2 * (w + h)
                val perimeterRatio = perimeter / max(expectedPerimeter, 1.0) 

                val pts = arrayOf(Point(), Point(), Point(), Point())
                rect.points(pts)

                if (boxArea < 1500 || ratio !in 1.5..7.5 || perimeterRatio < 0.5 || perimeterRatio > 1.8) {
                    rejectedRects.add(Pair(i, pts))
                    contour2f.release()
                    continue
                }

                var childCount = 0
                val node = hierarchy.get(0, i)
                if (node != null) {
                    var childIdx = node[2].toInt()
                    while (childIdx != -1) {
                        childCount++
                        val childNode = hierarchy.get(0, childIdx)
                        if (childNode != null) {
                            childIdx = childNode[0].toInt()
                        } else break
                    }
                }

                val shapeScore = max(0.0, 1.0 - Math.abs(1.0 - perimeterRatio)) * 3000.0 
                val dist = Math.hypot(rect.center.x - tightGray.cols() / 2.0, rect.center.y - tightGray.rows() / 2.0)
                val centerBiasScore = max(0.0, 1.0 - (dist / Math.hypot(tightGray.cols() / 2.0, tightGray.rows() / 2.0))) * 2000.0
                
                var hierarchyBonus = 0.0
                if (childCount in 4..22) {
                    hierarchyBonus = childCount * 3500.0 
                } else if (childCount == 0) {
                    hierarchyBonus = -6000.0
                }

                val finalScore = shapeScore + centerBiasScore + hierarchyBonus

                if (finalScore > bestScore && childCount > 0) {
                    if (bestContour != null) {
                        val prevContour2f = MatOfPoint2f(*bestContour!!.toArray())
                        val prevPts = arrayOf(Point(), Point(), Point(), Point())
                        Imgproc.minAreaRect(prevContour2f).points(prevPts)
                        lowScoreRects.add(Pair(-1, prevPts)) 
                        prevContour2f.release()
                    }
                    bestScore = finalScore
                    bestContour = contour
                    bestRatio = ratio
                    bestChildCount = childCount
                } else {
                    lowScoreRects.add(Pair(i, pts))
                }
                contour2f.release()
            }

            debugListener?.let {
                val debugMat = tightMat.clone()
                for (item in rejectedRects) { for(i in 0..3) Imgproc.line(debugMat, item.second[i], item.second[(i+1)%4], Scalar(255.0, 0.0, 0.0, 255.0), 2) }
                for (item in lowScoreRects) { for(i in 0..3) Imgproc.line(debugMat, item.second[i], item.second[(i+1)%4], Scalar(255.0, 165.0, 0.0, 255.0), 3) }
                
                var statusText = "FAILED: No valid plate boundary found."
                if (bestContour != null) {
                    val rawPts = arrayOf(Point(), Point(), Point(), Point())
                    val minRect = Imgproc.minAreaRect(MatOfPoint2f(*bestContour!!.toArray()))
                    minRect.points(rawPts)
                    for (i in 0..3) Imgproc.line(debugMat, rawPts[i], rawPts[(i + 1) % 4], Scalar(0.0, 255.0, 0.0, 255.0), 6)
                    statusText = "WINNER SECURED! (Outer Boundary)"
                }

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 10: Hierarchy Scoring (Tree)", listOf(
                    statusText,
                    "Total Contours Inside Window: ${contours.size}",
                    "Winner Core Children Count: $bestChildCount",
                    "Winner Score: ${String.format("%.0f", bestScore)} pts",
                    "Winner Aspect Ratio: ${String.format("%.2f", bestRatio)}"
                ), screenRatio)
                it.pauseAndShowStep("10단계: 계층 구조(Hierarchy) 채점 및 우승 테두리 선정", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            if (bestContour == null) return null

            // =====================================================================
            // 🚀 [Step 11] 기하학 정렬 및 최종 4점 추출
            // =====================================================================
            val contour2f = MatOfPoint2f(*bestContour!!.toArray())
            val minRect = Imgproc.minAreaRect(contour2f)
            val rawPoints = arrayOf(Point(), Point(), Point(), Point())
            minRect.points(rawPoints)
            contour2f.release()

            val sum = rawPoints.map { it.x + it.y }; val diff = rawPoints.map { it.x - it.y }
            val orderedPoints = arrayOf(
                rawPoints[sum.indexOf(sum.minOrNull()!!)], 
                rawPoints[diff.indexOf(diff.maxOrNull()!!)],
                rawPoints[sum.indexOf(sum.maxOrNull()!!)], 
                rawPoints[diff.indexOf(diff.minOrNull()!!)]
            )

            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)
            val pointsInRotatedLoose = orderedPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val srcMat = MatOfPoint2f(*pointsInRotatedLoose)
            val dstMat = MatOfPoint2f()
            Core.transform(srcMat, dstMat, invRotMat)
            
            val finalPts = dstMat.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }

            debugListener?.let {
                val debugMat = fullMat.clone()
                val colors = arrayOf(Scalar(255.0,0.0,0.0,255.0), Scalar(0.0,255.0,0.0,255.0), Scalar(0.0,0.0,255.0,255.0), Scalar(255.0,255.0,0.0,255.0))
                val labels = arrayOf("TL", "TR", "BR", "BL")
                for (i in 0..3) {
                    Imgproc.line(debugMat, finalPts[i], finalPts[(i + 1) % 4], Scalar(255.0, 255.0, 255.0, 255.0), 4)
                    Imgproc.circle(debugMat, finalPts[i], 15, colors[i], -1)
                    Imgproc.putText(debugMat, labels[i], Point(finalPts[i].x - 20, finalPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, colors[i], 4)
                }
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 11: Final Geometry Export", listOf(
                    "Inverse Affine Transform: SUCCESS",
                    "Perfected 4 Outer Boundary Points mapped to original Image.",
                    "Ready to warp perspective mask!"
                ), screenRatio)
                it.pauseAndShowStep("11단계: 최종 외부 테두리 4점 원본 이미지 보정", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
            invRotMat.release(); srcMat.release(); dstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            edges.release()
            dilateKernel.release()
            contours.forEach { it.release() }
            hierarchy.release()
            
            thresh.release()
            rotMat.release(); looseMat.release(); looseGray.release()
            rotatedLooseMat.release(); rotatedLooseGray.release()
            tightMat.release(); tightGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
