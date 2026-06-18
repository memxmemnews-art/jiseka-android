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
        // 🚀 [Step 2] 초기 ROI 설정
        // =====================================================================
        val roiWidth = (fullMat.cols() * 0.22).toInt() 
        val roiHeight = (roiWidth / 2.0).toInt()       

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
            val hudBmp = addDebugHUD(debugBmp, "Step 2: Downsized ROI (Ratio 2:1)", listOf(
                "ROI Width: $roiWidth px (화면 가로폭의 22%)",
                "ROI Height: $roiHeight px (2:1 비율 완벽 적용)",
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 컴팩트 2:1 탐색 구역 설정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 3 & 4] 1차 이진화 및 모폴로지
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
            
            val expectedHeight = (avgH * 2.8).coerceAtMost(looseRect.height.toDouble()) 
            val marginX = max(textSpreadWidth * 0.15, avgH * 1.5)

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
                    "Valid Characters Found: ${charList.size}",
                    "Fitted Line Angle: ${String.format("%.2f", angle)} deg",
                    "Status: 텍스트 클러스터 정상 감지"
                ), screenRatio)
                it.pauseAndShowStep("3~5단계: 문자 탐색 및 기울기 계산", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }
            pointsMat.release(); line.release(); tempRotMat.release(); dstCenterPts.release()
            
        } else {
            tightTop = (looseRect.height * 0.3).toInt()
            tightBottom = (looseRect.height * 0.7).toInt()
        }

        // =====================================================================
        // 🚀 [Step 6 & 7] 회전 및 타이트 ROI 추출
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
                "Tight ROI size: ${tightRect.width} x ${tightRect.height}",
                "Ready for Outer Edge Tracking."
            ), screenRatio)
            it.pauseAndShowStep("6~7단계: 수평 회전 및 타이트 탐색 영역 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 8 & 9] 진짜 테두리 찾기 (수직/수평 이중 구조적 모폴로지 최적화)
        // =====================================================================
        val edges = Mat()
        // 💡 수직 단절을 완벽하게 교량하기 위한 수직 전용 슬림 커널 (1x7)
        val verticalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, 7.0))
        // 💡 그릴 번짐 부작용을 통제하면서 가로선을 매끄럽게 닫아주는 수평 전용 커널 (3x1)
        val horizontalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 1.0))
        
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            // 💡 모서리를 둥글게 깎아내던 가우시안 블러 대신 직선 보존력이 우수한 medianBlur 적용
            Imgproc.medianBlur(tightGray, tightGray, 3)
            
            // 💡 어두운 좌측 반사 필름 영역의 경계 유실을 복구하기 위해 Canny 임계값 최적화
            Imgproc.Canny(tightGray, edges, 35.0, 100.0) 
            
            // 💡 세로 결합을 선행하여 단절을 깁고, 가로 결합으로 최종 루프 폐합
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, verticalKernel)
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, horizontalKernel)

            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(edges, debugMat, Imgproc.COLOR_GRAY2RGBA)
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 8~9: Fixed Canny Map", listOf(
                    "Method: medianBlur(3) + Canny(35,100)",
                    "Morphology: Vertical Close(1x7) -> Horizontal Close(3x1)",
                    "Insight: 세로 단절 우선 복구 및 과연결 부작용 제어"
                ), screenRatio)
                it.pauseAndShowStep("8~9단계: 번호판 테두리 이중 봉합 완료", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🚀 [Step 10] 윤곽선 채점 및 검증 (perimeterRatio 임계값 완화)
            // =====================================================================
            val tightCharCenters = mutableListOf<Point>()
            if (charList.isNotEmpty()) {
                val srcPts = MatOfPoint2f(*charList.map { it.center }.toTypedArray())
                val dstPts = MatOfPoint2f()
                Core.transform(srcPts, dstPts, rotMat)
                dstPts.toArray().forEach { pt ->
                    tightCharCenters.add(Point(pt.x - tightRect.x, pt.y - tightRect.y))
                }
                srcPts.release()
                dstPts.release()
            }

            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestScore = -Double.MAX_VALUE
            var bestContour: MatOfPoint? = null
            var bestChildCount = 0
            
            val rejectedRects = mutableListOf<Pair<Int, Array<Point>>>()
            val lowScoreRects = mutableListOf<Pair<Int, Array<Point>>>() 
            val minDynamicArea = tightRect.width * tightRect.height * 0.12

            var isFallback = false
            
            for (attempt in 0..1) {
                bestScore = -Double.MAX_VALUE
                bestContour = null
                rejectedRects.clear()
                lowScoreRects.clear()
                
                for (i in contours.indices) {
                    val contour = contours[i]
                    val contour2f = MatOfPoint2f(*contour.toArray())
                    val rect = Imgproc.minAreaRect(contour2f)

                    val w = rect.size.width
                    val h = rect.size.height
                    val longSide = max(w, h)
                    val shortSide = min(w, h)
                    
                    val ratio = longSide / max(shortSide, 1.0)
                    val boxArea = longSide * shortSide
                    
                    val perimeter = Imgproc.arcLength(contour2f, true)
                    val expectedPerimeter = 2 * (longSide + shortSide)
                    val perimeterRatio = perimeter / max(expectedPerimeter, 1.0) 

                    val pts = arrayOf(Point(), Point(), Point(), Point())
                    rect.points(pts)

                    // 💡 왜곡이나 일부 에지 돌출로 인한 탈락을 막기 위해 상한 비율을 1.9에서 2.2로 완화
                    val currentMaxPerimeter = if (isFallback) 3.0 else 2.2

                    if (boxArea < minDynamicArea || longSide < 160 || shortSide < 35 || ratio !in 1.5..7.5 || perimeterRatio < 0.4 || perimeterRatio > currentMaxPerimeter) {
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

                    var containedCharCount = 0
                    for (pt in tightCharCenters) {
                        if (Imgproc.pointPolygonTest(contour2f, pt, false) >= 0.0) {
                            containedCharCount++
                        }
                    }
                    val effectiveChildCount = max(childCount, containedCharCount)

                    var hierarchyBonus = 0.0
                    if (!isFallback) {
                        if (effectiveChildCount in 4..25) hierarchyBonus = effectiveChildCount * 3500.0 
                        else if (effectiveChildCount == 0) hierarchyBonus = -6000.0
                        else if (effectiveChildCount > 35) hierarchyBonus = -15000.0 
                    } else {
                        if (effectiveChildCount in 1..35) hierarchyBonus = 3000.0 
                        else if (effectiveChildCount == 0) hierarchyBonus = 0.0
                        else hierarchyBonus = -10000.0 
                    }

                    val noisePenaltyThreshold = if (isFallback) 1.8 else 1.2
                    val noisePenaltyAmount = if (isFallback) -2000.0 else -5000.0
                    val noisePenalty = if (perimeterRatio > noisePenaltyThreshold) noisePenaltyAmount else 0.0

                    val shapeScore = max(0.0, 1.0 - Math.abs(1.0 - perimeterRatio)) * 3000.0 
                    val dist = Math.hypot(rect.center.x - tightGray.cols() / 2.0, rect.center.y - tightGray.rows() / 2.0)
                    val centerBiasScore = max(0.0, 1.0 - (dist / Math.hypot(tightGray.cols() / 2.0, tightGray.rows() / 2.0))) * 2000.0
                    
                    val areaScore = if (isFallback) (boxArea / (tightRect.width * tightRect.height.toDouble())) * 5000.0 else 0.0
                    
                    val finalScore = shapeScore + centerBiasScore + hierarchyBonus + noisePenalty + areaScore
                    val isValid = if (isFallback) true else (effectiveChildCount > 0)

                    if (finalScore > bestScore && isValid) {
                        if (bestContour != null) {
                            val prevContour2f = MatOfPoint2f(*bestContour!!.toArray())
                            val prevPts = arrayOf(Point(), Point(), Point(), Point())
                            Imgproc.minAreaRect(prevContour2f).points(prevPts)
                            lowScoreRects.add(Pair(-1, prevPts)) 
                            prevContour2f.release()
                        }
                        bestScore = finalScore
                        bestContour = contour
                        bestChildCount = effectiveChildCount
                    } else {
                        lowScoreRects.add(Pair(i, pts))
                    }
                    contour2f.release()
                }
                
                if (bestContour != null) break
                isFallback = true
            }

            debugListener?.let {
                val debugMat = tightMat.clone()
                for (item in rejectedRects) { for(i in 0..3) Imgproc.line(debugMat, item.second[i], item.second[(i+1)%4], Scalar(255.0, 0.0, 0.0, 255.0), 2) }
                for (item in lowScoreRects) { for(i in 0..3) Imgproc.line(debugMat, item.second[i], item.second[(i+1)%4], Scalar(255.0, 165.0, 0.0, 255.0), 3) }
                
                var statusText = "FAILED: No valid plate boundary found."
                val modeText = if (isFallback) "Mode: 2차 Fallback (룰 완화 + Area Score)" else "Mode: 1차 Strict (고정밀 탐색)"
                
                if (bestContour != null) {
                    val rawPts = arrayOf(Point(), Point(), Point(), Point())
                    val minRect = Imgproc.minAreaRect(MatOfPoint2f(*bestContour!!.toArray()))
                    minRect.points(rawPts)
                    for (i in 0..3) Imgproc.line(debugMat, rawPts[i], rawPts[(i + 1) % 4], Scalar(0.0, 255.0, 0.0, 255.0), 6)
                    statusText = "WINNER SECURED! (Outer Boundary)"
                }

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 10: Size Filtered Hierarchy Scoring", listOf(
                    statusText,
                    modeText,
                    "Winner Core Children Count: $bestChildCount",
                    "Winner Score: ${String.format("%.0f", bestScore)} pts"
                ), screenRatio)
                it.pauseAndShowStep("10단계: 계층 구조 채점 및 완화 필터링 검증", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            if (bestContour == null) return null

            // =====================================================================
            // 🚀 [Step 11] 기하학 정렬: 원근 왜곡을 보존하는 순수 극단점 1:1 매핑
            // =====================================================================
            val contour2f = MatOfPoint2f(*bestContour!!.toArray())
            val approxCurve = MatOfPoint2f()
            val perimeter = Imgproc.arcLength(contour2f, true)
            Imgproc.approxPolyDP(contour2f, approxCurve, perimeter * 0.02, true) 
            val cleanPts = approxCurve.toArray()

            // 전처리 단에서 완벽하게 폐합되었으므로 강제 변형 없는 기하학적 극단점 공식 복원
            val rawTopLeft = cleanPts.minByOrNull { it.x + it.y }!!
            val rawBottomRight = cleanPts.maxByOrNull { it.x + it.y }!!
            val rawTopRight = cleanPts.maxByOrNull { it.x - it.y }!!
            val rawBottomLeft = cleanPts.minByOrNull { it.x - it.y }!!
            val orderedPoints = arrayOf(rawTopLeft, rawTopRight, rawBottomRight, rawBottomLeft)

            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)
            
            val rectRotated = orderedPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val rectSrcMat = MatOfPoint2f(*rectRotated)
            val rectDstMat = MatOfPoint2f()
            Core.transform(rectSrcMat, rectDstMat, invRotMat)
            val finalPts = rectDstMat.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }

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
                    "Vertices Count: ${cleanPts.size} (Perspective Intact)",
                    "Fixed bleeding at source preprocessing.",
                    "Ready to warp perspective mask!"
                ), screenRatio)
                it.pauseAndShowStep("11단계: 최종 4점 원본 이미지 보정 완료", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
            
            contour2f.release()
            approxCurve.release()
            invRotMat.release()
            rectSrcMat.release()
            rectDstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            edges.release()
            horizontalKernel.release() 
            verticalKernel.release()
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
