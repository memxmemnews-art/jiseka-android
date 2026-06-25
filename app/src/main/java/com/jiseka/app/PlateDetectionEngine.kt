package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
        val cx = touchX.toDouble(); val cy = touchY.toDouble()

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
            val hudBmp = addDebugHUD(debugBmp, "Step 2: Downsized ROI", listOf(
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 컴팩트 탐색 구역 설정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        val thresh = Mat()
        Imgproc.medianBlur(looseGray, looseGray, 3)
        Imgproc.GaussianBlur(looseGray, thresh, Size(5.0, 5.0), 0.0)
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, 7.0)

        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect)
        val charList = mutableListOf<CharData>()
        val rejectedList = mutableListOf<CharData>() 

        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
            val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
            
            if (area > 100 && area < looseRect.area() * 0.08) {
                if (ratio in 0.9..5.5 && rect.height >= 20) {
                    charList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect)) 
                } else if (ratio in 0.25..1.5 && rect.height >= 15) {
                    rejectedList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect)) 
                }
            }
        }
        tempContours.forEach { it.release() }; tempHierarchy.release(); tempOpen.release(); tempClose.release()

        if (charList.isEmpty()) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        var sortedChars = charList.sortedBy { it.center.x }.toMutableList()
        val rescueCandidates = mutableListOf<CharData>()

        for (i in 0 until sortedChars.size - 1) {
            val leftChar = sortedChars[i]
            val rightChar = sortedChars[i + 1]
            val gapX = rightChar.center.x - leftChar.center.x
            val avgW = (leftChar.width + rightChar.width) / 2.0

            if (gapX > avgW * 1.2) { 
                val avgH = (leftChar.height + rightChar.height) / 2.0
                val avgY = (leftChar.center.y + rightChar.center.y) / 2.0
                
                val rescuers = rejectedList.filter { r ->
                    r.center.x > leftChar.center.x + leftChar.width * 0.4 && 
                    r.center.x < rightChar.center.x - rightChar.width * 0.4 && 
                    abs(r.center.y - avgY) < avgH * 0.5 
                }
                rescueCandidates.addAll(rescuers) 
            }
        }
        
        sortedChars.addAll(rescueCandidates)
        sortedChars = sortedChars.sortedBy { it.center.x }.toMutableList()

        val gaps = (1 until sortedChars.size).map { sortedChars[it].center.x - sortedChars[it - 1].center.x }
        val medianGap = if (gaps.isNotEmpty()) gaps.sorted()[gaps.size / 2] else 0.0
        val avgWidth = sortedChars.map { it.width }.average()
        
        val maxGap = max(medianGap * 1.7, avgWidth * 1.8) 

        val clusters = mutableListOf<MutableList<CharData>>()
        var currentCluster = mutableListOf(sortedChars.first())

        for (i in 1 until sortedChars.size) {
            val prev = sortedChars[i - 1]
            val curr = sortedChars[i]
            val gap = curr.center.x - prev.center.x
            val yDiff = abs(curr.center.y - prev.center.y)
            val avgH = (prev.height + curr.height) / 2.0

            val localMaxGap = if (gap > maxGap && gap < avgWidth * 3.5 && yDiff < avgH * 0.35) {
                avgWidth * 3.5 
            } else {
                maxGap
            }

            if (gap > localMaxGap || yDiff > avgH * 0.35) {
                clusters.add(currentCluster) 
                currentCluster = mutableListOf(curr) 
            } else {
                currentCluster.add(curr)
            }
        }
        clusters.add(currentCluster)

        val validChars = (clusters.maxByOrNull { it.size } ?: sortedChars).toMutableList()

        if (validChars.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        // =====================================================================
        // 문자 중심선(Line Consistency) 검증 (20% 강화)
        // =====================================================================
        val pointsMatTemp = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
        val lineTemp = Mat()
        Imgproc.fitLine(pointsMatTemp, lineTemp, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        
        val vxTemp = lineTemp.get(0, 0)[0]
        val vyTemp = lineTemp.get(1, 0)[0]
        val x0Temp = lineTemp.get(2, 0)[0]
        val y0Temp = lineTemp.get(3, 0)[0]
        
        pointsMatTemp.release(); lineTemp.release()

        val A = vyTemp
        val B = -vxTemp
        val C = vxTemp * y0Temp - vyTemp * x0Temp
        val denominator = hypot(A, B)

        val localAvgHeight = validChars.map { it.height }.average()

        val iterator = validChars.iterator()
        while (iterator.hasNext()) {
            val charData = iterator.next()
            val dist = abs(A * charData.center.x + B * charData.center.y + C) / denominator
            
            if (dist > localAvgHeight * 0.20) {
                iterator.remove()
            }
        }

        if (validChars.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        // =====================================================================
        // KOR 마크 검출 및 뼈대에서 제거 
        // =====================================================================
        var hasKorMark = false
        val firstChar = validChars.first() 
        val roi = looseMat.submat(firstChar.rect)
        val meanColor = Core.mean(roi)
        roi.release() 
        
        if (meanColor.`val`[2] > meanColor.`val`[0] + 25 && meanColor.`val`[2] > meanColor.`val`[1] + 15) {
            hasKorMark = true 
            validChars.removeAt(0) 
        } else {
            val checkW = firstChar.rect.width.toInt() * 2
            val leftX = max(0, firstChar.rect.x - checkW)
            val scanW = firstChar.rect.x - leftX
            
            if (scanW > 10) {
                val leftRoi = looseMat.submat(Rect(leftX, firstChar.rect.y, scanW, firstChar.rect.height))
                val leftMean = Core.mean(leftRoi)
                leftRoi.release()
                
                if (leftMean.`val`[2] > leftMean.`val`[0] + 10 && leftMean.`val`[2] > leftMean.`val`[1] + 5) {
                    hasKorMark = true
                }
            }
        }

        if (validChars.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        val pointsMat = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(pointsMat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        
        var vx = line.get(0, 0)[0]; var vy = line.get(1, 0)[0]
        if (vx < 0) { vx = -vx; vy = -vy } 
        
        val topPtsArray = validChars.map { Point(it.center.x, it.rect.y.toDouble()) }.toTypedArray()
        val bottomPtsArray = validChars.map { Point(it.center.x, it.rect.y.toDouble() + it.rect.height) }.toTypedArray()
        
        val topPts = MatOfPoint2f(*topPtsArray); val bottomPts = MatOfPoint2f(*bottomPtsArray)
        val topLineMat = Mat(); val bottomLineMat = Mat()
        
        Imgproc.fitLine(topPts, topLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        Imgproc.fitLine(bottomPts, bottomLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)

        val tvx = topLineMat.get(0, 0)[0]; val tvy = topLineMat.get(1, 0)[0]
        val tx0 = topLineMat.get(2, 0)[0]; val ty0 = topLineMat.get(3, 0)[0]
        
        val bvx = bottomLineMat.get(0, 0)[0]; val bvy = bottomLineMat.get(1, 0)[0]
        val bx0 = bottomLineMat.get(2, 0)[0]; val by0 = bottomLineMat.get(3, 0)[0]

        // 💡 쏠림 현상 진단을 위해 '겉 테두리 경계선' 기준이 아닌 '글자 중심점' 기준으로 선을 그어 교차점을 구함
        val leftTopMid = Point(validChars.first().rect.x + validChars.first().rect.width / 2.0, validChars.first().rect.y.toDouble())
        val leftCenter = validChars.first().center
        val leftBottomMid = Point(validChars.first().rect.x + validChars.first().rect.width / 2.0, validChars.first().rect.y + validChars.first().rect.height.toDouble())

        val rightTopMid = Point(validChars.last().rect.x + validChars.last().rect.width / 2.0, validChars.last().rect.y.toDouble())
        val rightCenter = validChars.last().center
        val rightBottomMid = Point(validChars.last().rect.x + validChars.last().rect.width / 2.0, validChars.last().rect.y + validChars.last().rect.height.toDouble())

        val leftPts = MatOfPoint2f(leftTopMid, leftCenter, leftBottomMid)
        val leftLine = Mat()
        Imgproc.fitLine(leftPts, leftLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var lvx = leftLine.get(0, 0)[0]; var lvy = leftLine.get(1, 0)[0]
        if (lvy < 0) { lvx = -lvx; lvy = -lvy }
        val lx0 = validChars.first().center.x; val ly0 = validChars.first().center.y

        val rightPts = MatOfPoint2f(rightTopMid, rightCenter, rightBottomMid)
        val rightLine = Mat()
        Imgproc.fitLine(rightPts, rightLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var rvx = rightLine.get(0, 0)[0]; var rvy = rightLine.get(1, 0)[0]
        if (rvy < 0) { rvx = -rvx; rvy = -rvy }
        val rx0 = validChars.last().center.x; val ry0 = validChars.last().center.y

        fun getIntersect(x1: Double, y1: Double, vx1: Double, vy1: Double, x2: Double, y2: Double, vx2: Double, vy2: Double): Point {
            val dx = x2 - x1; val dy = y2 - y1
            val det = vx2 * vy1 - vy2 * vx1
            if (abs(det) < 1e-6) return Point(x1, y1)
            val u = (dy * vx1 - dx * vy1) / det
            return Point(x2 + u * vx2, y2 + u * vy2)
        }

        val initTL = getIntersect(tx0, ty0, tvx, tvy, lx0, ly0, lvx, lvy)
        val initTR = getIntersect(tx0, ty0, tvx, tvy, rx0, ry0, rvx, rvy)
        val initBR = getIntersect(bx0, by0, bvx, bvy, rx0, ry0, rvx, rvy)
        val initBL = getIntersect(bx0, by0, bvx, bvy, lx0, ly0, lvx, lvy)
        
        debugListener?.let {
            val debugMat = looseMat.clone()
            for (charData in charList) {
                val color = if (validChars.contains(charData)) Scalar(0.0, 255.0, 0.0, 255.0) else Scalar(255.0, 0.0, 0.0, 255.0)
                Imgproc.rectangle(debugMat, charData.rect, color, 2)
            }
            for (charData in rescueCandidates) {
                if (validChars.contains(charData)) Imgproc.rectangle(debugMat, charData.rect, Scalar(0.0, 255.0, 255.0, 255.0), 3) 
            }
            for (i in 0..3) {
                val pts = arrayOf(initTL, initTR, initBR, initBL)
                Imgproc.line(debugMat, pts[i], pts[(i+1)%4], Scalar(255.0, 0.0, 255.0, 255.0), 3)
            }
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 3~5: Base Wireframe", listOf(
                "Method: 2-Track KOR 마크 검증 (조명 착시 오인 차단)",
                "Status: KOR Mark Detected = $hasKorMark",
                "결과: 하얀 숫자가 온전히 뼈대로 보존됨"
            ), screenRatio)
            it.pauseAndShowStep("3~5단계: 노이즈 필터링 및 기초 뼈대 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🔍 [신규 디버그] 모서리 교차점(Intersection) 생성 원리 확인
        // =====================================================================
        debugListener?.let {
            val debugMat = looseMat.clone()
            
            // 상단, 하단, 좌측, 우측 무한 연장선(Line) 그리기
            val scale = 500.0
            
            // Top Line (Red)
            val topPt1 = Point(tx0 - tvx * scale, ty0 - tvy * scale)
            val topPt2 = Point(tx0 + tvx * scale, ty0 + tvy * scale)
            Imgproc.line(debugMat, topPt1, topPt2, Scalar(255.0, 0.0, 0.0, 255.0), 2)
            
            // Bottom Line (Blue)
            val botPt1 = Point(bx0 - bvx * scale, by0 - bvy * scale)
            val botPt2 = Point(bx0 + bvx * scale, by0 + bvy * scale)
            Imgproc.line(debugMat, botPt1, botPt2, Scalar(0.0, 0.0, 255.0, 255.0), 2)

            // Left Line (Green)
            val leftPt1 = Point(lx0 - lvx * scale, ly0 - lvy * scale)
            val leftPt2 = Point(lx0 + lvx * scale, ly0 + lvy * scale)
            Imgproc.line(debugMat, leftPt1, leftPt2, Scalar(0.0, 255.0, 0.0, 255.0), 2)

            // Right Line (Yellow)
            val rightPt1 = Point(rx0 - rvx * scale, ry0 - rvy * scale)
            val rightPt2 = Point(rx0 + rvx * scale, ry0 + rvy * scale)
            Imgproc.line(debugMat, rightPt1, rightPt2, Scalar(255.0, 255.0, 0.0, 255.0), 2)

            // 교차점(Corners) 원으로 표시
            val pts = arrayOf(initTL, initTR, initBR, initBL)
            for (pt in pts) {
                Imgproc.circle(debugMat, pt, 5, Scalar(255.0, 255.0, 255.0, 255.0), -1)
            }

            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            // 첫 글자(2)와 마지막 글자(6)의 폭(Width) 비교 출력
            val firstW = validChars.first().width.toInt()
            val lastW = validChars.last().width.toInt()
            
            val hudBmp = addDebugHUD(debugBmp, "Step 5.5: Intersection Diagnosis", listOf(
                "분석: 가림막 모서리 생성을 위한 가상 교차선 검증",
                "첫 글자 폭: ${firstW}px | 마지막 글자 폭: ${lastW}px",
                "진단: 두 글자의 폭이 다르면 중심점 기반 교차 시 쏠림 발생"
            ), screenRatio)
            it.pauseAndShowStep("5.5단계: 교차선 쏠림 원인 진단", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        pointsMat.release(); line.release(); topPts.release(); bottomPts.release()
        topLineMat.release(); bottomLineMat.release()
        leftPts.release(); rightPts.release(); leftLine.release(); rightLine.release()

        // =====================================================================
        // 🚀 [Step 6] 중심점 대칭 스케일링
        // =====================================================================
        var resultPoints: List<ImmutablePoint>? = null

        try {
            val midX = (initTL.x + initTR.x + initBR.x + initBL.x) / 4.0
            val midY = (initTL.y + initTR.y + initBR.y + initBL.y) / 4.0

            val scaleY = 1.35 
            val scaleX = 1.35 

            val nX = -vy; val nY = vx

            val finalPts = listOf(initTL, initTR, initBR, initBL).map { pt ->
                val dx = pt.x - midX
                val dy = pt.y - midY
                
                val localX = dx * vx + dy * vy
                val localY = dx * nX + dy * nY
                
                val scaledX = localX * scaleX
                val scaledY = localY * scaleY
                
                Point(
                    midX + scaledX * vx + scaledY * nX + looseRect.x,
                    midY + scaledX * vy + scaledY * nY + looseRect.y
                )
            }

            debugListener?.let {
                val debugMat = fullMat.clone()
                val colors = arrayOf(Scalar(255.0, 0.0, 0.0, 255.0), Scalar(0.0, 255.0, 0.0, 255.0), 
                                     Scalar(0.0, 0.0, 255.0, 255.0), Scalar(255.0, 255.0, 0.0, 255.0))
                val labels = arrayOf("TL", "TR", "BR", "BL")

                for (i in 0..3) {
                    Imgproc.line(debugMat, finalPts[i], finalPts[(i + 1) % 4], Scalar(0.0, 255.0, 0.0, 255.0), 5)
                    Imgproc.circle(debugMat, finalPts[i], 15, colors[i], -1)
                    Imgproc.putText(debugMat, labels[i], Point(finalPts[i].x - 20, finalPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 1.8, colors[i], 4)
                }

                Imgproc.circle(debugMat, Point(midX + looseRect.x, midY + looseRect.y), 8, Scalar(0.0, 255.0, 255.0, 255.0), -1)

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 6: Symmetric Scaling", listOf(
                    "Mode: 대칭 팽창",
                    "Status: Shift 0.0 (Perfectly Centered)",
                    "Result: 우측 쏠림 발생"
                ), screenRatio)
                it.pauseAndShowStep("최종 단계: 대칭 팽창 가림막 좌표 확정", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release()
            looseMat.release(); looseGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
