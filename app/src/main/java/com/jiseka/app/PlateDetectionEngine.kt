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

    private data class CoreResult(
        val points: List<ImmutablePoint>?,
        val hasMergedBlob: Boolean
    )

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
            } else if (log.contains("FAIL") || log.contains("삭제") || log.contains("Yes") || log.contains("탈락") || log.contains("불량")) {
                paint.color = Color.parseColor("#FF5555") 
            } else if (log.contains("No") || log.contains("조립") || log.contains("부활")) {
                paint.color = Color.parseColor("#55FF55")
            } else if (log.contains("[Fallback]")) {
                paint.color = Color.parseColor("#FFA500") 
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
        
        val firstTry = executeDetectionCore(fullBitmap, touchX, touchY, blockSize = 31, C = 7.0, attempt = 1, debugListener)
        
        if (firstTry.points != null) {
            return firstTry.points
        }
        
        if (firstTry.hasMergedBlob || firstTry.points == null) {
            android.util.Log.d("JISEKA", "1차 실패 (병합 발생 또는 유효군집 없음) -> 극단적 분리 모드(2차 Fallback) 가동")
            val secondTry = executeDetectionCore(fullBitmap, touchX, touchY, blockSize = 19, C = 15.0, attempt = 2, debugListener)
            return secondTry.points
        }
        
        return null
    }

    private fun executeDetectionCore(
        fullBitmap: Bitmap,
        touchX: Float, touchY: Float,
        blockSize: Int, C: Double,
        attempt: Int,
        debugListener: DetectionDebugListener?
    ): CoreResult {
        
        var hasMergedBlob = false 

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
        
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, blockSize, C)

        debugListener?.let {
            val debugMat0 = Mat()
            Imgproc.cvtColor(thresh, debugMat0, Imgproc.COLOR_GRAY2RGBA)
            val debugBmp0 = Bitmap.createBitmap(debugMat0.cols(), debugMat0.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat0, debugBmp0)
            
            val logs0 = listOf(
                "[Fallback] 현재 시도 상태: ${attempt}차 시도 (BlockSize: $blockSize, C: $C)",
                "[진단 0] Morph 연산 전 순수 Threshold 결과",
                " -> 확인: 여기서 글자와 테두리가 이미 붙었는가?",
                " -> (Yes) Threshold 단계에서 이미 Merge 발생",
                " -> (No) 다음 단계인 Morph Close가 Merge 원인"
            )
            
            val hudBmp0 = addDebugHUD(debugBmp0, "디버그 1/4: Threshold 직후 ($attempt 차 시도)", logs0, screenRatio)
            it.pauseAndShowStep("디버그 1/4: Threshold 상태 확인", hudBmp0)
            
            debugMat0.release()
            debugBmp0.recycle()
        }

        if (attempt == 1) {
            val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
            tempClose.release()
        } else {
            val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
            tempClose.release()
        }
        
        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        tempOpen.release()
        
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect, var rejectReason: String = "", var contrast: Double = 0.0, var density: Double = 0.0)
        val charList = mutableListOf<CharData>()
        val rejectedList = mutableListOf<CharData>() 
        val totalRejectedRects = mutableListOf<CharData>() 
        val onScreenDebugTexts = mutableListOf<Pair<Rect, List<String>>>()

        val step1Logs = mutableListOf<String>()
        var failReason = ""
        var resultPoints: List<ImmutablePoint>? = null

        step1Logs.add("[진단 1] 1차 기하학적 형태 검증 (${tempContours.size}개 발견)")

        val maxAreaBound = looseRect.area() * 0.08
        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
            val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
            
            var rReason = ""
            if (area <= 100) {
                rReason = "A(Min)"
            } else if (area >= maxAreaBound) {
                rReason = "A(Max)"
                hasMergedBlob = true 
            } else {
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
                } else {
                    rejectedList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect, "Rescue")) 
                }
            } else {
                totalRejectedRects.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect, rReason))
            }
        }

        if (charList.isNotEmpty() || rejectedList.isNotEmpty()) {
            var totalContrast = 0.0
            val allCandidates = charList + rejectedList
            for (c in allCandidates) {
                val roiGray = looseGray.submat(c.rect)
                val minMax = Core.minMaxLoc(roiGray)
                c.contrast = minMax.maxVal - minMax.minVal
                roiGray.release()

                val roiThresh = thresh.submat(c.rect)
                val whitePixelCount = Core.countNonZero(roiThresh)
                c.density = whitePixelCount.toDouble() / c.rect.area() 
                roiThresh.release()
                
                if (charList.contains(c)) {
                    totalContrast += c.contrast
                }
            }

            val avgContrast = if (charList.isNotEmpty()) totalContrast / charList.size else 0.0
            step1Logs.add("[기준] 그룹 평균 명암차(AvgC): ${avgContrast.toInt()}")
            
            val contrastLimit = max(avgContrast * 0.40, 40.0) 

            val charIter = charList.iterator()
            while (charIter.hasNext()) {
                val c = charIter.next()
                if (c.density < 0.12 || c.density > 0.65) {
                    c.rejectReason = "D(${(c.density * 100).toInt()}%)"
                    totalRejectedRects.add(c)
                    charIter.remove()
                    continue
                }
                if (abs(c.contrast - avgContrast) > contrastLimit) {
                    c.rejectReason = "C(${c.contrast.toInt()})"
                    totalRejectedRects.add(c)
                    charIter.remove()
                    continue
                }
                onScreenDebugTexts.add(Pair(c.rect, listOf("D:${(c.density * 100).toInt()}% C:${c.contrast.toInt()}", "W:${c.rect.width} H:${c.rect.height}")))
            }

            val rejIter = rejectedList.iterator()
            while (rejIter.hasNext()) {
                val c = rejIter.next()
                if (c.density < 0.12 || c.density > 0.65) {
                    c.rejectReason = "D(${(c.density * 100).toInt()}%)"
                    totalRejectedRects.add(c)
                    rejIter.remove()
                } else if (abs(c.contrast - avgContrast) > contrastLimit) {
                    c.rejectReason = "C(${c.contrast.toInt()})"
                    totalRejectedRects.add(c)
                    rejIter.remove()
                }
            }
        }
        
        step1Logs.add(" -> 1.5단계 후 생존: ${charList.size}개 / 완전 탈락: ${totalRejectedRects.size}개")
        if (charList.isEmpty()) failReason = "원인 1: 1차 검증 후 글자 전멸"

        debugListener?.let {
            val debugMat1 = Mat()
            Imgproc.cvtColor(thresh, debugMat1, Imgproc.COLOR_GRAY2RGBA)
            
            for (charData in totalRejectedRects) {
                Imgproc.rectangle(debugMat1, charData.rect, Scalar(0.0, 0.0, 255.0, 255.0), 1)
                Imgproc.rectangle(debugMat1, Point(charData.rect.x.toDouble(), charData.rect.y.toDouble() + 2), Point(charData.rect.x.toDouble() + 50, charData.rect.y.toDouble() + 14), Scalar(0.0, 0.0, 0.0, 255.0), -1)
                Imgproc.putText(debugMat1, charData.rejectReason, Point(charData.rect.x.toDouble() + 2, charData.rect.y.toDouble() + 11), Imgproc.FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0.0, 255.0, 255.0, 255.0), 1)
            }
            
            for (charData in rejectedList) {
                Imgproc.rectangle(debugMat1, charData.rect, Scalar(80.0, 80.0, 80.0, 255.0), 1)
            }
            for (charData in charList) {
                Imgproc.rectangle(debugMat1, charData.rect, Scalar(255.0, 0.0, 0.0, 255.0), 2)
            }
            
            for ((rect, lines) in onScreenDebugTexts) {
                for (idx in lines.indices) {
                    val textY = rect.y.toDouble() - 4 - (12 * (lines.size - 1 - idx))
                    Imgproc.rectangle(debugMat1, Point(rect.x.toDouble(), textY - 9), Point(rect.x.toDouble() + 85, textY + 2), Scalar(0.0, 0.0, 0.0, 255.0), -1)
                    Imgproc.putText(debugMat1, lines[idx], Point(rect.x.toDouble() + 2, textY), Imgproc.FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255.0, 255.0, 0.0, 255.0), 1)
                }
            }
            
            val debugBmp1 = Bitmap.createBitmap(debugMat1.cols(), debugMat1.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat1, debugBmp1)
            val title = if (failReason.isNotEmpty()) "디버그 2/4 중단 ($failReason)" else "디버그 2/4: 1.5단계 검증 ($attempt 차 시도)"
            val hudBmp1 = addDebugHUD(debugBmp1, title, step1Logs, screenRatio)
            it.pauseAndShowStep("디버그 2/4: 1.5단계 픽셀 검증", hudBmp1)
            debugMat1.release(); debugBmp1.recycle()
        }

        if (failReason.isNotEmpty()) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            tempContours.forEach { it.release() }; tempHierarchy.release()
            return CoreResult(null, hasMergedBlob) 
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

        // --- [추가된 로직: 번호판 내부 '안전 지대(Safe Zone)' 기반 극단적 파편 사전 구출] ---
        if (sortedChars.size >= 2) {
            val safeMinX = sortedChars.first().center.x
            val safeMaxX = sortedChars.last().center.x

            val extremeRescuers = totalRejectedRects.filter { reject ->
                val isInsideSafeZone = reject.center.x > safeMinX && reject.center.x < safeMaxX
                
                if (!isInsideSafeZone) return@filter false

                sortedChars.any { validChar ->
                    val xOverlap = min(reject.rect.x + reject.rect.width, validChar.rect.x + validChar.rect.width) - max(reject.rect.x, validChar.rect.x)
                    val xMatch = xOverlap > min(reject.rect.width, validChar.rect.width) * 0.4
                    
                    val yGap = max(0, max(reject.rect.y, validChar.rect.y) - min(reject.rect.y + reject.rect.height, validChar.rect.y + validChar.rect.height))
                    
                    xMatch && yGap < 20 
                }
            }

            if (extremeRescuers.isNotEmpty()) {
                sortedChars.addAll(extremeRescuers)
                sortedChars = sortedChars.sortedBy { it.center.x }.toMutableList()
                step1Logs.add(" -> [사전 구출] 안전 지대 내 극단적 파편 ${extremeRescuers.size}개 부활 성공")
            }
        }
        // -------------------------------------------------------------------------

        // --- [수정된 로직: 위치 무관, '비율'과 '긴 여백' 기반 완벽한 한글 파편 조립] ---
        var j = 0
        while (j < sortedChars.size - 1) {
            val curr = sortedChars[j]
            val next = sortedChars[j + 1]

            val currRight = curr.rect.x + curr.rect.width
            val currBottom = curr.rect.y + curr.rect.height
            val nextRight = next.rect.x + next.rect.width
            val nextBottom = next.rect.y + next.rect.height

            val xOverlap = min(currRight, nextRight) - max(curr.rect.x, next.rect.x)
            val yOverlap = min(currBottom, nextBottom) - max(curr.rect.y, next.rect.y)
            
            val xGap = max(0, max(curr.rect.x, next.rect.x) - min(currRight, nextRight))
            val yGap = max(0, max(curr.rect.y, next.rect.y) - min(currBottom, nextBottom))

            val currRatio = curr.height / max(curr.width, 1.0)
            val nextRatio = next.height / max(next.width, 1.0)

            val gapAfterNext = if (j + 2 < sortedChars.size) {
                max(0.0, sortedChars[j + 2].rect.x.toDouble() - nextRight)
            } else {
                999.0 
            }
            
            val hasLongSpaceAfter = gapAfterNext > max(xGap * 2.5, 15.0)

            val isVerticalSplit = xOverlap > 0 && yGap < 20 && abs(curr.center.x - next.center.x) < 20.0 &&
                                  (currRatio < 1.0 || nextRatio < 1.0)
            
            val isHorizontalSplit = yOverlap > 0 && xGap < 15 && abs(curr.center.y - next.center.y) < 15.0 &&
                                    (currRatio > 3.0 || nextRatio > 3.0) && hasLongSpaceAfter

            if (isVerticalSplit || isHorizontalSplit) {
                val unionLeft = min(curr.rect.x, next.rect.x)
                val unionTop = min(curr.rect.y, next.rect.y)
                val unionRight = max(currRight, nextRight)
                val unionBottom = max(currBottom, nextBottom)
                
                val unionRect = Rect(unionLeft, unionTop, unionRight - unionLeft, unionBottom - unionTop)
                val unionCenter = Point(unionRect.x + unionRect.width / 2.0, unionRect.y + unionRect.height / 2.0)
                
                val mergedChar = CharData(unionCenter, unionRect.width.toDouble(), unionRect.height.toDouble(), unionRect)
                
                sortedChars.removeAt(j + 1)
                sortedChars.removeAt(j)
                sortedChars.add(j, mergedChar)
                
                step1Logs.add(" -> [파편 조립] X:${curr.center.x.toInt()} & ${next.center.x.toInt()} 합병 완료")
            } else {
                j++
            }
        }
        // -------------------------------------------------------------------------

        // --- [추가된 로직: 양 끝단 볼트/노이즈 제거 (Peeling)] ---
        if (sortedChars.size > 2) {
            val sortedH = sortedChars.map { it.height }.sorted()
            val medianH = sortedH[sortedChars.size / 2]
            val boltHeightLimit = medianH * 0.55
            
            while (sortedChars.isNotEmpty() && sortedChars.first().height < boltHeightLimit) {
                val removed = sortedChars.removeAt(0)
                step1Logs.add(" -> [볼트 제거] 좌측 끝 노이즈 삭제 (X:${removed.center.x.toInt()}, H:${removed.height.toInt()})")
            }
            
            while (sortedChars.isNotEmpty() && sortedChars.last().height < boltHeightLimit) {
                val removed = sortedChars.removeAt(sortedChars.size - 1)
                step1Logs.add(" -> [볼트 제거] 우측 끝 노이즈 삭제 (X:${removed.center.x.toInt()}, H:${removed.height.toInt()})")
            }
        }
        // -------------------------------------------------------------------------

        val gaps = (1 until sortedChars.size).map { sortedChars[it].center.x - sortedChars[it - 1].center.x }
        val medianGap = if (gaps.isNotEmpty()) gaps.sorted()[gaps.size / 2] else 0.0
        val avgWidth = sortedChars.map { it.width }.average()
        val maxGap = max(medianGap * 1.7, avgWidth * 1.8) 
        
        val step2Logs = mutableListOf<String>()
        step2Logs.add("[기준] AvgW:${String.format("%.1f", avgWidth)} / MedGap:${String.format("%.1f", medianGap)} / BaseMaxGap:${String.format("%.1f", maxGap)}")
        
        val clusters = mutableListOf<MutableList<CharData>>()
        val allClustersForDebug = mutableListOf<List<CharData>>()
        if (sortedChars.isNotEmpty()) {
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
        }

        val roiCenterX = looseRect.width / 2.0
        val roiCenterY = looseRect.height / 2.0
        val maxDistX = looseRect.width * 0.4
        val maxDistY = looseRect.height * 0.4 

        val centerFilteredClusters = mutableListOf<MutableList<CharData>>()

        for (i in 0 until clusters.size) {
            val cluster = clusters[i]
            if (cluster.isEmpty()) continue

            val avgX = cluster.map { it.center.x }.average()
            val avgY = cluster.map { it.center.y }.average()
            val distX = abs(avgX - roiCenterX)
            val distY = abs(avgY - roiCenterY)

            if (distX > maxDistX || distY > maxDistY) {
                step2Logs.add(" -> [군집 탈락 G$i] 중심 이탈 (dX:${distX.toInt()}, dY:${distY.toInt()})")
            } else {
                centerFilteredClusters.add(cluster)
            }
        }

        var validChars = mutableListOf<CharData>()

        if (centerFilteredClusters.isNotEmpty()) {
            val finalClusters = mutableListOf<MutableList<CharData>>()
            
            for (i in 0 until centerFilteredClusters.size) {
                val cluster = centerFilteredClusters[i]
                
                if (cluster.isEmpty()) continue
                
                val sortedY = cluster.map { it.center.y }.sorted()
                val medianY = sortedY[cluster.size / 2]
                
                val sortedH = cluster.map { it.height }.sorted()
                val medianH = sortedH[cluster.size / 2]
                
                val medianTop = medianY - medianH / 2.0
                val medianBottom = medianY + medianH / 2.0

                val alignedCluster = mutableListOf<CharData>()
                for (charData in cluster) {
                    val charTop = charData.center.y - charData.height / 2.0
                    val charBottom = charData.center.y + charData.height / 2.0
                    
                    val limit = medianH * 0.35
                    
                    val centerDiff = abs(charData.center.y - medianY)
                    val topDiff = abs(charTop - medianTop)
                    val bottomDiff = abs(charBottom - medianBottom)
                    
                    if (centerDiff <= limit && topDiff <= limit && bottomDiff <= limit) {
                        alignedCluster.add(charData)
                    } else {
                        step2Logs.add(" -> [개별 탈락] X:${charData.center.x.toInt()} 일직선 이탈 (상/중/하 불량)")
                    }
                }
                
                if (alignedCluster.size < 7) {
                    step2Logs.add(" -> [군집 탈락 G$i] 솎아낸 후 개수 미달 (${alignedCluster.size}개 < 7개)")
                    continue
                }

                finalClusters.add(alignedCluster)
            }

            validChars = (finalClusters.maxByOrNull { it.size } ?: mutableListOf()).toMutableList()
        }

        if (validChars.size < 7) {
            failReason = "원인 2: 유효 군집 없음 (7개 이상 & 일직선 정렬 실패)"
        }

        debugListener?.let {
            val debugMat2 = looseMat.clone()
            
            for (i in 0 until sortedChars.size) {
                val charData = sortedChars[i]
                Imgproc.rectangle(debugMat2, charData.rect, Scalar(255.0, 255.0, 255.0, 255.0), 2)
                Imgproc.rectangle(debugMat2, Point(charData.rect.x.toDouble(), charData.rect.y.toDouble() - 20), Point(charData.rect.x.toDouble() + 25, charData.rect.y.toDouble()), Scalar(0.0, 0.0, 0.0, 255.0), -1)
                Imgproc.putText(debugMat2, "$i", Point(charData.rect.x.toDouble() + 4, charData.rect.y.toDouble() - 4), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 0.0, 255.0), 2)
            }
            
            val debugBmp2 = Bitmap.createBitmap(debugMat2.cols(), debugMat2.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat2, debugBmp2)
            val title = if (failReason.isNotEmpty()) "디버그 3/4 중단 ($failReason)" else "디버그 3/4: 군집화 의사결정 추적 ($attempt 차 시도)"
            val hudBmp2 = addDebugHUD(debugBmp2, title, step2Logs, screenRatio)
            it.pauseAndShowStep("디버그 3/4: 군집화 판별", hudBmp2)
            debugMat2.release(); debugBmp2.recycle()
        }

        if (failReason.isNotEmpty()) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            tempContours.forEach { it.release() }; tempHierarchy.release()
            return CoreResult(null, hasMergedBlob) 
        }

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

        if (validChars.size < 7) {
            failReason = "직선 2차 검증 중 삭제되어 7개 미만 됨"
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
            val title = if (failReason.isNotEmpty()) "디버그 4/4 중단 ($failReason)" else "디버그 4/4: 최종 뼈대 확정 ($attempt 차 시도)"
            val hudBmp3 = addDebugHUD(debugBmp3, title, step3Logs, screenRatio)
            it.pauseAndShowStep("디버그 4/4: 최종 뼈대", hudBmp3)
            debugMat3.release(); debugBmp3.recycle()
        }

        tempContours.forEach { it.release() }; tempHierarchy.release()

        if (failReason.isNotEmpty() || validChars.size < 6) { 
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return CoreResult(null, hasMergedBlob) 
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

        return CoreResult(resultPoints, hasMergedBlob) 
    }
}
