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
            textSize = 35f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val paddingX = 40f
        val lineHeight = 35f 
        val maxTextWidth = canvasWidth - (paddingX * 2)
        var currentY = 60f 
        
        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        paint.textSize = 38f
        currentY = drawTextWithWrap(canvas, title, paddingX, currentY, paint, maxTextWidth, lineHeight)

        currentY += 10f 

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        paint.textSize = 26f 
        for (log in logs) {
            if (log.startsWith("->") || log.startsWith("[경고]")) {
                paint.color = Color.parseColor("#FF8888") 
            } else if (log.startsWith("[진단")) {
                paint.color = Color.parseColor("#55FF55")
            } else if (log.startsWith("[정보]") || log.startsWith("[기준]")) {
                paint.color = Color.parseColor("#55FFFF") 
            } else if (log.contains("FAIL") || log.contains("삭제")) {
                paint.color = Color.parseColor("#FF5555") 
            } else {
                paint.color = Color.WHITE
            }
            currentY = drawTextWithWrap(canvas, log, paddingX, currentY, paint, maxTextWidth, lineHeight)
        }

        val textBottom = currentY + 15f 
        val margin = 20f
        
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

        val thresh = Mat()
        Imgproc.medianBlur(looseGray, looseGray, 3)
        Imgproc.GaussianBlur(looseGray, thresh, Size(5.0, 5.0), 0.0)
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, 7.0)

        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect, var rejectReason: String = "")
        val charList = mutableListOf<CharData>()
        val rejectedList = mutableListOf<CharData>() 
        val totalRejectedRects = mutableListOf<CharData>() 
        
        var failReason = ""

        // ==========================================================
        // [디버그 화면 1] 1차 검증 및 이진화 뷰
        // ==========================================================
        val step1Logs = mutableListOf<String>()
        step1Logs.add("[진단 1] 1차 통과 분석 (${tempContours.size}개 발견)")

        val maxAreaBound = looseRect.area() * 0.08
        val onScreenTextsStep1 = mutableListOf<Pair<Rect, List<String>>>()

        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
            val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
            
            var rReason = ""
            if (area <= 100) rReason = "A(Min)"
            else if (area >= maxAreaBound) rReason = "A(Max)"
            else {
                if (ratio in 0.9..5.5 && rect.height >= 20) {
                    // Pass
                } else if (ratio in 0.25..1.5 && rect.height >= 15) {
                    // Rescue
                } else {
                    if (ratio !in 0.25..5.5) rReason = "R(${String.format("%.1f", ratio)})"
                    else rReason = "H(${rect.height})"
                }
            }

            if (rReason.isEmpty()) {
                if (ratio in 0.9..5.5 && rect.height >= 20) {
                    charList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect)) 
                    onScreenTextsStep1.add(Pair(rect, listOf("C(${center.x.toInt()},${center.y.toInt()})", "W:${rect.width} H:${rect.height}")))
                } else {
                    rejectedList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect, "Rescue")) 
                }
            } else {
                totalRejectedRects.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect, rReason))
            }
        }
        
        step1Logs.add(" -> 1차 생존: ${charList.size}개 / 완전 탈락: ${totalRejectedRects.size}개")
        if (charList.isEmpty()) failReason = "원인 1: 1차 검증 글자 전멸"

        debugListener?.let {
            val debugMat1 = Mat()
            Imgproc.cvtColor(thresh, debugMat1, Imgproc.COLOR_GRAY2RGBA)
            
            for (charData in totalRejectedRects) {
                Imgproc.rectangle(debugMat1, charData.rect, Scalar(0.0, 0.0, 255.0, 255.0), 1)
                
                // 💡 크래시 픽스: submat 대신 안전한 Imgproc.rectangle (두께 -1) 채우기 방식 사용
                Imgproc.rectangle(debugMat1, Point(charData.rect.x.toDouble(), charData.rect.y.toDouble() + 2), Point(charData.rect.x.toDouble() + 45, charData.rect.y.toDouble() + 14), Scalar(0.0, 0.0, 0.0, 255.0), -1)
                Imgproc.putText(debugMat1, charData.rejectReason, Point(charData.rect.x.toDouble() + 2, charData.rect.y.toDouble() + 11), Imgproc.FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0.0, 255.0, 255.0, 255.0), 1)
            }
            
            for (charData in rejectedList) {
                Imgproc.rectangle(debugMat1, charData.rect, Scalar(80.0, 80.0, 80.0, 255.0), 1)
            }
            for (charData in charList) {
                Imgproc.rectangle(debugMat1, charData.rect, Scalar(255.0, 0.0, 0.0, 255.0), 2)
            }
            
            for ((rect, lines) in onScreenTextsStep1) {
                for (idx in lines.indices) {
                    val textY = rect.y.toDouble() - 4 - (12 * (lines.size - 1 - idx))
                    
                    // 💡 크래시 픽스: submat 교체
                    Imgproc.rectangle(debugMat1, Point(rect.x.toDouble(), textY - 9), Point(rect.x.toDouble() + 85, textY + 2), Scalar(0.0, 0.0, 0.0, 255.0), -1)
                    Imgproc.putText(debugMat1, lines[idx], Point(rect.x.toDouble() + 2, textY), Imgproc.FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255.0, 255.0, 0.0, 255.0), 1)
                }
            }
            
            val debugBmp1 = Bitmap.createBitmap(debugMat1.cols(), debugMat1.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat1, debugBmp1)
            val title = if (failReason.isNotEmpty()) "디버그 1/3 중단 ($failReason)" else "디버그 1/3: 1차 검증 (이진화 뷰)"
            val hudBmp1 = addDebugHUD(debugBmp1, title, step1Logs, screenRatio)
            it.pauseAndShowStep("디버그 1/3: 이진화 뷰", hudBmp1)
            debugMat1.release(); debugBmp1.recycle()
        }

        if (failReason.isNotEmpty()) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            tempOpen.release(); tempClose.release(); tempContours.forEach { it.release() }; tempHierarchy.release()
            return null
        }

        // ==========================================================
        // [디버그 화면 2] 군집화 의사결정 추적
        // ==========================================================
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
        
        val step2Logs = mutableListOf<String>()
        step2Logs.add("[기준] AvgW:${String.format("%.1f", avgWidth)} / MedGap:${String.format("%.1f", medianGap)} / BaseMaxGap:${String.format("%.1f", maxGap)}")
        
        var clusters = mutableListOf<MutableList<CharData>>()
        val allClustersForDebug = mutableListOf<List<CharData>>()
        var currentCluster = mutableListOf(sortedChars.first())
        
        for (i in 1 until sortedChars.size) {
            val curr = sortedChars[i]
            val prev = sortedChars[i - 1]
            val gap = curr.center.x - prev.center.x
            val yDiff = abs(curr.center.y - prev.center.y)
            val avgH = (prev.height + curr.height) / 2.0

            val localMaxGap = if (gap > maxGap && gap < avgWidth * 3.5 && yDiff < avgH * 0.45) avgWidth * 3.5 else maxGap
            val yLimit = avgH * 0.45
            
            val gapFail = gap > localMaxGap
            val yFail = yDiff > yLimit
            var isSplit = gapFail || yFail
            var splitReason = if (gapFail && yFail) "Gap+Y" else if (gapFail) "Gap" else "Y축"

            if (isSplit && gap < avgWidth * 1.3 && yDiff <= avgH * 0.85 && currentCluster.size >= 2) {
                val prev1 = currentCluster.last()
                val prev2 = currentCluster[currentCluster.size - 2]
                val slope1 = (prev1.center.y - prev2.center.y) / max(prev1.center.x - prev2.center.x, 1.0)
                val slope2 = (curr.center.y - prev1.center.y) / max(curr.center.x - prev1.center.x, 1.0)
                
                if (abs(slope1 - slope2) < 0.2) {
                    isSplit = false 
                    splitReason = "예외(사선)"
                }
            }

            val statusMsg = if (isSplit) "FAIL($splitReason)" else "PASS"
            step2Logs.add(" -> [${i-1}→$i] G:${String.format("%.1f", gap)}(${String.format("%.1f", localMaxGap)}) Y:${String.format("%.1f", yDiff)}(${String.format("%.1f", yLimit)}) => $statusMsg")

            if (isSplit) {
                clusters.add(currentCluster) 
                allClustersForDebug.add(currentCluster.toList())
                currentCluster = mutableListOf(curr) 
            } else {
                currentCluster.add(curr)
            }
        }
        clusters.add(currentCluster)
        allClustersForDebug.add(currentCluster.toList())

        var validChars = (clusters.maxByOrNull { it.size } ?: sortedChars).toMutableList()
        if (validChars.size < 2) failReason = "원인 2: 군집 토막남 (최대 1개)"

        debugListener?.let {
            val debugMat2 = looseMat.clone()
            
            for (i in 0 until sortedChars.size) {
                val charData = sortedChars[i]
                Imgproc.rectangle(debugMat2, charData.rect, Scalar(255.0, 255.0, 255.0, 255.0), 2)
                
                // 💡 크래시 픽스: submat 교체
                Imgproc.rectangle(debugMat2, Point(charData.rect.x.toDouble(), charData.rect.y.toDouble() - 20), Point(charData.rect.x.toDouble() + 25, charData.rect.y.toDouble()), Scalar(0.0, 0.0, 0.0, 255.0), -1)
                Imgproc.putText(debugMat2, "$i", Point(charData.rect.x.toDouble() + 4, charData.rect.y.toDouble() - 4), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 0.0, 255.0), 2)
            }
            
            val debugBmp2 = Bitmap.createBitmap(debugMat2.cols(), debugMat2.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat2, debugBmp2)
            val title = if (failReason.isNotEmpty()) "디버그 2/3 중단 ($failReason)" else "디버그 2/3: 군집화 의사결정 추적"
            val hudBmp2 = addDebugHUD(debugBmp2, title, step2Logs, screenRatio)
            it.pauseAndShowStep("디버그 2/3: 군집화 판별", hudBmp2)
            debugMat2.release(); debugBmp2.recycle()
        }

        if (failReason.isNotEmpty()) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            tempOpen.release(); tempClose.release(); tempContours.forEach { it.release() }; tempHierarchy.release()
            return null
        }

        // ==========================================================
        // [디버그 화면 3] 최종 군집 및 fitLine 검증
        // ==========================================================
        val step3Logs = mutableListOf<String>()
        var clusterSummary = "[정보] 군집 결과: "
        for (idx in 0 until allClustersForDebug.size) clusterSummary += "G$idx(${allClustersForDebug[idx].size}개) "
        step3Logs.add(clusterSummary)

        val pointsMatTemp = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
        val lineTemp = Mat()
        Imgproc.fitLine(pointsMatTemp, lineTemp, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        
        val vxTemp = lineTemp.get(0, 0)[0]; val vyTemp = lineTemp.get(1, 0)[0]
        val x0Temp = lineTemp.get(2, 0)[0]; val y0Temp = lineTemp.get(3, 0)[0]
        pointsMatTemp.release(); lineTemp.release()

        val A = vyTemp; val B = -vxTemp; val C = vxTemp * y0Temp - vyTemp * x0Temp
        val denominator = hypot(A, B)
        val localAvgHeight = validChars.map { it.height }.average()
        val fitLineLimit = localAvgHeight * 0.20

        val iterator = validChars.iterator()
        val fitLineRemovedChars = mutableListOf<CharData>()
        while (iterator.hasNext()) {
            val charData = iterator.next()
            val dist = abs(A * charData.center.x + B * charData.center.y + C) / denominator
            
            if (dist > fitLineLimit) {
                step3Logs.add(" -> [삭제] X:${charData.center.x.toInt()} (Dist:${String.format("%.1f", dist)} > Lim:${String.format("%.1f", fitLineLimit)})")
                fitLineRemovedChars.add(charData)
                iterator.remove()
            }
        }
        
        step3Logs.add("[진단 3] fitLine ${fitLineRemovedChars.size}개 삭제, 최종 ${validChars.size}개 생존")

        if (validChars.size < 2) {
            failReason = "직선 검증 중 삭제되어 2개 미만 됨"
        } else {
            val firstChar = validChars.first() 
            val roi = looseMat.submat(firstChar.rect)
            val meanColor = Core.mean(roi)
            roi.release() 
            
            val r = meanColor.`val`[0].toInt(); val g = meanColor.`val`[1].toInt(); val b = meanColor.`val`[2].toInt()
            if (b > r + 25 && b > g + 15) {
                validChars.removeAt(0) 
                step3Logs.add(" -> [정보] KOR 파랑 인식 삭제 완료")
            }
        }

        debugListener?.let {
            val debugMat3 = looseMat.clone()
            
            val clusterColors = arrayOf(Scalar(255.0, 255.0, 0.0, 255.0), Scalar(255.0, 0.0, 255.0, 255.0), Scalar(0.0, 255.0, 255.0, 255.0), Scalar(255.0, 150.0, 0.0, 255.0))
            for (clusterIndex in 0 until allClustersForDebug.size) {
                val cluster = allClustersForDebug[clusterIndex]
                val color = if (clusterIndex < clusterColors.size) clusterColors[clusterIndex] else Scalar(200.0, 200.0, 200.0, 255.0)
                for (charData in cluster) {
                    Imgproc.rectangle(debugMat3, charData.rect, color, 1)
                }
            }

            for (charData in fitLineRemovedChars) {
                Imgproc.line(debugMat3, Point(charData.rect.x.toDouble(), charData.rect.y.toDouble()), Point((charData.rect.x + charData.rect.width).toDouble(), (charData.rect.y + charData.rect.height).toDouble()), Scalar(0.0, 0.0, 255.0, 255.0), 2)
                Imgproc.line(debugMat3, Point((charData.rect.x + charData.rect.width).toDouble(), charData.rect.y.toDouble()), Point(charData.rect.x.toDouble(), (charData.rect.y + charData.rect.height).toDouble()), Scalar(0.0, 0.0, 255.0, 255.0), 2)
            }

            if (validChars.isNotEmpty()) {
                for (i in 0 until validChars.size) {
                    Imgproc.rectangle(debugMat3, validChars[i].rect, Scalar(0.0, 255.0, 0.0, 255.0), 3)
                    if (i > 0) Imgproc.line(debugMat3, validChars[i-1].center, validChars[i].center, Scalar(0.0, 255.0, 0.0, 255.0), 2)
                }
            }
            
            val debugBmp3 = Bitmap.createBitmap(debugMat3.cols(), debugMat3.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat3, debugBmp3)
            val title = if (failReason.isNotEmpty()) "디버그 3/3 중단 ($failReason)" else "디버그 3/3: 최종 뼈대 확정"
            val hudBmp3 = addDebugHUD(debugBmp3, title, step3Logs, screenRatio)
            it.pauseAndShowStep("디버그 3/3: 최종 뼈대", hudBmp3)
            debugMat3.release(); debugBmp3.recycle()
        }

        tempContours.forEach { it.release() }; tempHierarchy.release(); tempOpen.release(); tempClose.release()

        if (failReason.isNotEmpty() || validChars.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        // ==========================================================
        // [정상 루틴] 모서리선 생성 및 최종 스케일링
        // ==========================================================
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

        val leftTopEdge = Point(validChars.first().rect.x.toDouble(), validChars.first().rect.y.toDouble())
        val leftMidEdge = Point(validChars.first().rect.x.toDouble(), validChars.first().center.y)
        val leftBottomEdge = Point(validChars.first().rect.x.toDouble(), validChars.first().rect.y + validChars.first().rect.height.toDouble())

        val rightX = validChars.last().rect.x + validChars.last().rect.width.toDouble()
        val rightTopEdge = Point(rightX, validChars.last().rect.y.toDouble())
        val rightMidEdge = Point(rightX, validChars.last().center.y)
        val rightBottomEdge = Point(rightX, validChars.last().rect.y + validChars.last().rect.height.toDouble())

        val leftPts = MatOfPoint2f(leftTopEdge, leftMidEdge, leftBottomEdge)
        val leftLine = Mat()
        Imgproc.fitLine(leftPts, leftLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var lvx = leftLine.get(0, 0)[0]; var lvy = leftLine.get(1, 0)[0]
        if (lvy < 0) { lvx = -lvx; lvy = -lvy }
        val lx0 = validChars.first().rect.x.toDouble()
        val ly0 = validChars.first().center.y

        val rightPts = MatOfPoint2f(rightTopEdge, rightMidEdge, rightBottomEdge)
        val rightLine = Mat()
        Imgproc.fitLine(rightPts, rightLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var rvx = rightLine.get(0, 0)[0]; var rvy = rightLine.get(1, 0)[0]
        if (rvy < 0) { rvx = -rvx; rvy = -rvy }
        val rx0 = rightX
        val ry0 = validChars.last().center.y

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

        pointsMat.release(); line.release(); topPts.release(); bottomPts.release()
        topLineMat.release(); bottomLineMat.release()
        leftPts.release(); rightPts.release(); leftLine.release(); rightLine.release()

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
                    "Mode: 겉 테두리 기준 대칭 팽창 완료",
                    "Status: Shift 0.0"
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
