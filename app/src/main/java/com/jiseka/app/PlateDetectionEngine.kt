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

    class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect, var contrast: Double = 0.0, var density: Double = 0.0)

    // ==========================================================================================
    // [공통 유틸리티] 디버그 UI 및 안전한 Rect 생성
    // ==========================================================================================
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
            if (log.startsWith("->") || log.startsWith("[경고]")) paint.color = Color.parseColor("#FF8888") 
            else if (log.startsWith("[진단")) paint.color = Color.parseColor("#55FF55")
            else if (log.startsWith("[정보]") || log.startsWith("[기준]")) paint.color = Color.parseColor("#55FFFF") 
            else if (log.contains("FAIL") || log.contains("중단") || log.contains("삭제") || log.contains("손절")) paint.color = Color.parseColor("#FF5555") 
            else if (log.contains("통과") || log.contains("성공") || log.contains("조립")) paint.color = Color.parseColor("#55FF55")
            else paint.color = Color.WHITE
            
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

    private fun getSafeRect(x: Int, y: Int, w: Int, h: Int, maxW: Int, maxH: Int): Rect {
        val safeX = x.coerceIn(0, maxW - 1)
        val safeY = y.coerceIn(0, maxH - 1)
        val safeW = w.coerceAtMost(maxW - safeX)
        val safeH = h.coerceAtMost(maxH - safeY)
        return Rect(safeX, safeY, safeW, safeH)
    }

    // ==========================================================================================
    // 메인 엔트리: 터치 지점부터 번호판 추적 시작
    // ==========================================================================================
    fun rescuePlateFromPoint(
        fullBitmap: Bitmap, 
        touchX: Float, touchY: Float, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat(); val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)
        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

        // 1. 스마트 Probe 기반 초기 씨앗(Seed) 추출
        val seedChars = extractSeedChars(fullMat, fullGray, touchX.toInt(), touchY.toInt(), debugListener)
        if (seedChars.isEmpty()) {
            fullMat.release(); fullGray.release()
            return null
        }

        // 2. 진정한 문자 추적(Tracking) 기반 좌우 확장
        val collectedChars = expandAndCollect(seedChars, fullMat, fullGray, debugListener, screenRatio)
        
        // 3. 글로벌 대청소, 최종 검증 및 와이어프레임 생성
        var resultPoints: List<ImmutablePoint>? = null
        if (collectedChars.size >= 4) { 
            resultPoints = buildWireframe(collectedChars, fullMat, debugListener, screenRatio)
        }

        fullMat.release(); fullGray.release()
        return resultPoints
    }

    // ==========================================================================================
    // [엔진 1] ExpansionEngine: 초기 씨앗 추출 (실시간 Probe 로직)
    // ==========================================================================================
    private fun extractSeedChars(
        fullMat: Mat, fullGray: Mat, cx: Int, cy: Int, debugListener: DetectionDebugListener?
    ): List<CharData> {
        
        var roiWidth = (fullMat.cols() * 0.05).toInt() 
        var roiHeight = (roiWidth * 1.5).toInt()       
        var currentX = cx - roiWidth / 2
        var currentY = cy - roiHeight / 2

        val maxExpansions = 5 
        val pad = (fullMat.cols() * 0.015).toInt() 

        var finalRect = getSafeRect(currentX, currentY, roiWidth, roiHeight, fullMat.cols(), fullMat.rows())

        for (i in 0 until maxExpansions) {
            val roiGray = Mat()
            fullGray.submat(finalRect).copyTo(roiGray)
            val thresh = Mat()
            Imgproc.medianBlur(roiGray, roiGray, 3)
            Imgproc.adaptiveThreshold(roiGray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 19, 12.0)

            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var touchesLeft = false; var touchesRight = false
            var touchesTop = false; var touchesBottom = false

            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                if (rect.area() < 40) continue 

                val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
                // 그릴/범퍼 실시간 감지 시 손절
                val isBumperOrGrille = ratio < 0.2 || ratio > 6.0 || rect.width > (fullMat.cols() * 0.3)
                if (isBumperOrGrille) continue 

                if (rect.x <= 2) touchesLeft = true
                if (rect.x + rect.width >= finalRect.width - 2) touchesRight = true
                if (rect.y <= 2) touchesTop = true
                if (rect.y + rect.height >= finalRect.height - 2) touchesBottom = true
            }

            roiGray.release(); thresh.release(); hierarchy.release()
            contours.forEach { it.release() }

            if (!touchesLeft && !touchesRight && !touchesTop && !touchesBottom) break

            if (touchesLeft) { currentX -= pad; roiWidth += pad }
            if (touchesRight) { roiWidth += pad }
            if (touchesTop) { currentY -= pad; roiHeight += pad }
            if (touchesBottom) { roiHeight += pad }

            finalRect = getSafeRect(currentX, currentY, roiWidth, roiHeight, fullMat.cols(), fullMat.rows())
        }

        // 스마트 보정된 ROI에서 정식 스캔 (1차 7.0 -> 실패 시 2차 15.0)
        var seeds = scanRegion(fullMat, fullGray, finalRect, 31, 7.0)
        if (seeds.isEmpty()) {
            seeds = scanRegion(fullMat, fullGray, finalRect, 19, 15.0)
        }
        return seeds
    }

    // ==========================================================================================
    // [엔진 2] ExpansionEngine: 진정한 '문자 추적(Tracking)' 기반 좌우 확장
    // ==========================================================================================
    private fun expandAndCollect(
        seeds: List<CharData>, fullMat: Mat, fullGray: Mat, 
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<CharData> {
        
        val currentCluster = seeds.toMutableList()
        val stepLogs = mutableListOf<String>()
        stepLogs.add("[진단] 초기 Seed 확보 완료. 문자 꼬리물기(Tracking) 시작.")

        // --- 좌측 추적 ---
        var expandLeft = true
        var leftLoops = 0
        while (expandLeft && leftLoops < 8) {
            leftLoops++
            currentCluster.sortBy { it.rect.x }
            val leftMost = currentCluster.first()
            
            val medianH = currentCluster.map { it.height }.sorted()[currentCluster.size / 2]
            val medianW = currentCluster.map { it.width }.sorted()[currentCluster.size / 2]
            val medianCenterY = currentCluster.map { it.center.y }.sorted()[currentCluster.size / 2]
            
            // 왼쪽 문자에 완벽히 맞물리는 타이트한 탐색 상자 생성
            val searchW = (medianW * 2.5).toInt() 
            val searchH = (medianH * 1.3).toInt() 
            val searchX = leftMost.rect.x - searchW
            val searchY = (medianCenterY - searchH / 2.0).toInt()
            val searchRect = getSafeRect(searchX, searchY, searchW, searchH, fullMat.cols(), fullMat.rows())
            
            if (searchRect.width < 10) break 

            var newCandidates = scanRegion(fullMat, fullGray, searchRect, 31, 7.0)
            if (newCandidates.isEmpty()) {
                stepLogs.add(" -> [좌측 추적] 1차(C=7) 실패, 2차 Rescue(C=15) 가동")
                newCandidates = scanRegion(fullMat, fullGray, searchRect, 19, 15.0)
            }
            
            val sortedCandidates = newCandidates.sortedByDescending { it.rect.x }
            var foundValid = false

            for (candidate in sortedCandidates) {
                if (currentCluster.any { abs(it.center.x - candidate.center.x) < it.width * 0.5 }) continue
                if (isValidNextChar(candidate, currentCluster, stepLogs, "좌측")) {
                    currentCluster.add(candidate)
                    foundValid = true
                    stepLogs.add(" -> [좌측 연결] X:${candidate.center.x.toInt()} 꼬리물기 성공")
                } else {
                    expandLeft = false 
                    break
                }
            }
            if (!foundValid) expandLeft = false 
        }

        // --- 우측 추적 ---
        var expandRight = true
        var rightLoops = 0
        while (expandRight && rightLoops < 8) {
            rightLoops++
            currentCluster.sortBy { it.rect.x }
            val rightMost = currentCluster.last()
            
            val medianH = currentCluster.map { it.height }.sorted()[currentCluster.size / 2]
            val medianW = currentCluster.map { it.width }.sorted()[currentCluster.size / 2]
            val medianCenterY = currentCluster.map { it.center.y }.sorted()[currentCluster.size / 2]
            
            // 오른쪽 문자에 완벽히 맞물리는 상자 생성
            val searchW = (medianW * 2.5).toInt()
            val searchH = (medianH * 1.3).toInt()
            val searchX = rightMost.rect.x + rightMost.rect.width
            val searchY = (medianCenterY - searchH / 2.0).toInt()
            val searchRect = getSafeRect(searchX, searchY, searchW, searchH, fullMat.cols(), fullMat.rows())
            
            if (searchRect.width < 10) break 

            var newCandidates = scanRegion(fullMat, fullGray, searchRect, 31, 7.0)
            if (newCandidates.isEmpty()) {
                stepLogs.add(" -> [우측 추적] 1차(C=7) 실패, 2차 Rescue(C=15) 가동")
                newCandidates = scanRegion(fullMat, fullGray, searchRect, 19, 15.0)
            }
            
            val sortedCandidates = newCandidates.sortedBy { it.rect.x }
            var foundValid = false

            for (candidate in sortedCandidates) {
                if (currentCluster.any { abs(it.center.x - candidate.center.x) < it.width * 0.5 }) continue
                if (isValidNextChar(candidate, currentCluster, stepLogs, "우측")) {
                    currentCluster.add(candidate)
                    foundValid = true
                    stepLogs.add(" -> [우측 연결] X:${candidate.center.x.toInt()} 꼬리물기 성공")
                } else {
                    expandRight = false 
                    break
                }
            }
            if (!foundValid) expandRight = false 
        }

        currentCluster.sortBy { it.rect.x }
        
        // -------------------------------------------------------------------------
        // 💡 [핵심 복원] 글로벌 대청소 (Top 3 기반 볼트/먼지 완벽 일괄 멸망)
        // -------------------------------------------------------------------------
        if (currentCluster.size > 2) {
            val heightsDesc = currentCluster.map { it.height }.sortedDescending()
            val trueHeight = if (heightsDesc.size >= 3) heightsDesc[2] else heightsDesc.first()
            val purgeHeightLimit = trueHeight * 0.50
            
            val purgeIterator = currentCluster.iterator()
            var purgedCount = 0
            while (purgeIterator.hasNext()) {
                val c = purgeIterator.next()
                if (c.height < purgeHeightLimit) {
                    purgeIterator.remove()
                    purgedCount++
                }
            }
            if (purgedCount > 0) stepLogs.add(" -> [글로벌 청소] 기준 미달 볼트/먼지 ${purgedCount}개 일괄 삭제")
        }

        stepLogs.add("[진단] 최종 수집 완료: 총 ${currentCluster.size}개 문자 확보")

        debugListener?.let {
            val debugMat = fullMat.clone()
            for (c in currentCluster) { Imgproc.rectangle(debugMat, c.rect, Scalar(0.0, 255.0, 0.0, 255.0), 3) }
            for (i in 0 until currentCluster.size - 1) {
                Imgproc.line(debugMat, currentCluster[i].center, currentCluster[i+1].center, Scalar(0.0, 255.0, 255.0, 255.0), 2)
            }
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "디버그 1/2: 추적(Tracking) 및 글로벌 청소 결과", stepLogs, screenRatio)
            it.pauseAndShowStep("디버그 1/2: 추적 결과", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return currentCluster
    }

    // 4대 하드 스톱 방어선
    private fun isValidNextChar(newChar: CharData, currentCluster: List<CharData>, logs: MutableList<String>, dir: String): Boolean {
        if (currentCluster.isEmpty()) return true
        
        val sortedTop = currentCluster.map { it.rect.y.toDouble() }.sorted()
        val sortedBottom = currentCluster.map { (it.rect.y + it.rect.height).toDouble() }.sorted()
        val sortedArea = currentCluster.map { it.rect.area() }.sorted()
        val sortedHeight = currentCluster.map { it.height }.sorted()

        val medianTop = sortedTop[currentCluster.size / 2]
        val medianBottom = sortedBottom[currentCluster.size / 2]
        val medianArea = sortedArea[currentCluster.size / 2]
        val medianH = sortedHeight[currentCluster.size / 2]

        val topDiff = abs(newChar.rect.y - medianTop)
        val bottomDiff = abs((newChar.rect.y + newChar.rect.height) - medianBottom)
        val isBoundsDeviated = topDiff > (medianH * 0.25) || bottomDiff > (medianH * 0.25)

        if (newChar.rect.area() < medianArea * 0.35 && isBoundsDeviated) {
            logs.add(" -> [$dir 중단] 볼트/먼지 차단 (X:${newChar.center.x.toInt()})")
            return false 
        }
        if (newChar.rect.area() > medianArea * 2.5 && isBoundsDeviated) {
            logs.add(" -> [$dir 중단] 범퍼/프레임 융합 차단 (X:${newChar.center.x.toInt()})")
            return false
        }
        if (abs(newChar.center.y - (medianTop + medianBottom) / 2.0) > medianH * 0.5) {
            logs.add(" -> [$dir 중단] Y축 궤도 완전 이탈 (X:${newChar.center.x.toInt()})")
            return false
        }
        if (newChar.height < medianH * 0.65 || newChar.height > medianH * 1.35) {
            logs.add(" -> [$dir 중단] 높이 급변 차단 (X:${newChar.center.x.toInt()})")
            return false
        }
        return true
    }

    // ==========================================================================================
    // [엔진 3] CharacterScanner: 로컬 파편 조립(Merge) 기능이 내장된 순수 문자 검출기
    // ==========================================================================================
    private fun scanRegion(fullMat: Mat, fullGray: Mat, roi: Rect, blockSize: Int, C: Double): List<CharData> {
        val roiGray = Mat()
        fullGray.submat(roi).copyTo(roiGray)
        val thresh = Mat()
        
        Imgproc.medianBlur(roiGray, roiGray, 3)
        Imgproc.GaussianBlur(roiGray, thresh, Size(5.0, 5.0), 0.0)
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, blockSize, C)

        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        tempClose.release()
        
        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        tempOpen.release()
        
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val rawBlobs = mutableListOf<CharData>()
        for (contour in contours) {
            val localRect = Imgproc.boundingRect(contour)
            if (localRect.area() < 30) continue
            val ratio = localRect.height.toDouble() / max(localRect.width.toDouble(), 1.0)
            
            // 파편(Rescue)까지 모두 담기 위해 0.15~7.0의 광범위한 비율 허용
            if (ratio in 0.15..7.0) {
                val globalRect = Rect(localRect.x + roi.x, localRect.y + roi.y, localRect.width, localRect.height)
                val globalCenter = Point(globalRect.x + globalRect.width / 2.0, globalRect.y + globalRect.height / 2.0)
                rawBlobs.add(CharData(globalCenter, globalRect.width.toDouble(), globalRect.height.toDouble(), globalRect))
            }
        }
        
        // -------------------------------------------------------------------------
        // 💡 [핵심 복원] 로컬 파편 조립 ('로'의 상하/좌우 분리 완벽 복구)
        // -------------------------------------------------------------------------
        rawBlobs.sortBy { it.center.x }
        var j = 0
        while (j < rawBlobs.size - 1) {
            val curr = rawBlobs[j]
            val next = rawBlobs[j + 1]

            val currRight = curr.rect.x + curr.rect.width
            val currBottom = curr.rect.y + curr.rect.height
            val nextRight = next.rect.x + next.rect.width
            val nextBottom = next.rect.y + next.rect.height

            val xOverlap = min(currRight, nextRight) - max(curr.rect.x, next.rect.x)
            val yOverlap = min(currBottom, nextBottom) - max(curr.rect.y, next.rect.y)
            val xGap = max(0.0, max(curr.rect.x, next.rect.x) - min(currRight, nextRight).toDouble())
            val yGap = max(0.0, max(curr.rect.y, next.rect.y) - min(currBottom, nextBottom).toDouble())

            val currRatio = curr.height / max(curr.width, 1.0)
            val nextRatio = next.height / max(next.width, 1.0)

            // 수직 병합 (비율 기반)
            val isVerticalSplit = xOverlap > min(curr.width, next.width) * 0.3 && 
                                  yGap < (curr.height * 0.45) &&
                                  (currRatio < 1.0 || nextRatio < 1.0 || currRatio > 1.2)
            
            // 수평 병합
            val isHorizontalSplit = yOverlap > 0 && xGap < 15 && abs(curr.center.y - next.center.y) < 15.0 &&
                                    (currRatio > 3.0 || nextRatio > 3.0)

            if (isVerticalSplit || isHorizontalSplit) {
                val unionLeft = min(curr.rect.x, next.rect.x)
                val unionTop = min(curr.rect.y, next.rect.y)
                val unionRight = max(currRight, nextRight)
                val unionBottom = max(currBottom, nextBottom)
                val unionRect = Rect(unionLeft, unionTop, unionRight - unionLeft, unionBottom - unionTop)
                val unionCenter = Point(unionRect.x + unionRect.width / 2.0, unionRect.y + unionRect.height / 2.0)
                
                rawBlobs.removeAt(j + 1)
                rawBlobs.removeAt(j)
                rawBlobs.add(j, CharData(unionCenter, unionRect.width.toDouble(), unionRect.height.toDouble(), unionRect))
            } else {
                j++
            }
        }

        // 조립된 최종 문자들 중 밀도/정상비율을 만족하는 것만 선별
        val finalCandidates = mutableListOf<CharData>()
        for (blob in rawBlobs) {
            val ratio = blob.height / max(blob.width, 1.0)
            if (ratio in 0.25..5.5 && blob.height >= 12) {
                // 원본 좌표계 기준 Local Rect 생성
                val localRect = Rect(blob.rect.x - roi.x, blob.rect.y - roi.y, blob.width.toInt(), blob.height.toInt())
                val roiThresh = thresh.submat(localRect)
                val whitePixelCount = Core.countNonZero(roiThresh)
                val density = whitePixelCount.toDouble() / blob.rect.area()
                roiThresh.release()
                
                if (density in 0.12..0.85) {
                    blob.density = density
                    finalCandidates.add(blob)
                }
            }
        }
        
        roiGray.release(); thresh.release(); hierarchy.release()
        contours.forEach { it.release() }
        
        return finalCandidates
    }

    // ==========================================================================================
    // [엔진 4] GeometryBuilder: 최종 검증 및 와이어프레임 계산
    // ==========================================================================================
    private fun buildWireframe(
        collectedChars: List<CharData>, fullMat: Mat, 
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<ImmutablePoint>? {
        
        val stepLogs = mutableListOf<String>()
        stepLogs.add("[진단] 와이어프레임 기하학 조립 전 최종 검증 시작")

        var validChars = collectedChars.toMutableList()

        // -------------------------------------------------------------------------
        // 🚨 [복원] KOR (파란색 태극 마크) 인식 및 제거
        // -------------------------------------------------------------------------
        if (validChars.isNotEmpty()) {
            val firstChar = validChars.first() 
            val roiMat = fullMat.submat(firstChar.rect)
            val meanColor = Core.mean(roiMat)
            roiMat.release() 
            
            val r = meanColor.`val`[0].toInt(); val g = meanColor.`val`[1].toInt(); val b = meanColor.`val`[2].toInt()
            
            if (b > r + 25 && b > g + 15) {
                validChars.removeAt(0) 
                stepLogs.add(" -> [검증] 좌측 KOR 파랑 마크 식별 및 제거 완료")
            }
        }

        // -------------------------------------------------------------------------
        // 🚨 [복원] fitLine 기반 Y축/기울기 궤도 정밀 아웃라이어 제거
        // -------------------------------------------------------------------------
        if (validChars.size >= 4) {
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
            var removedCount = 0
            while (iterator.hasNext()) {
                val charData = iterator.next()
                val dist = abs(A * charData.center.x + B * charData.center.y + C) / denominator
                
                if (dist > fitLineLimit) {
                    stepLogs.add(" -> [검증] X:${charData.center.x.toInt()} 선형 궤도 이탈 삭제")
                    iterator.remove()
                    removedCount++
                }
            }
        }

        if (validChars.size < 4) {
            stepLogs.add(" -> [FAIL] 검증 후 유효 문자가 4개 미만입니다.")
            return null
        }

        stepLogs.add("[진단] 최종 ${validChars.size}개 문자로 와이어프레임 렌더링")

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

        val firstChar = validChars.first()
        val lastChar = validChars.last()

        val leftTopEdge = Point(firstChar.rect.x.toDouble(), firstChar.rect.y.toDouble())
        val leftMidEdge = Point(firstChar.rect.x.toDouble(), firstChar.center.y)
        val leftBottomEdge = Point(firstChar.rect.x.toDouble(), firstChar.rect.y + firstChar.rect.height.toDouble())

        val rightX = lastChar.rect.x + lastChar.rect.width.toDouble()
        val rightTopEdge = Point(rightX, lastChar.rect.y.toDouble())
        val rightMidEdge = Point(rightX, lastChar.center.y)
        val rightBottomEdge = Point(rightX, lastChar.rect.y + lastChar.rect.height.toDouble())

        val leftPts = MatOfPoint2f(leftTopEdge, leftMidEdge, leftBottomEdge)
        val leftLine = Mat()
        Imgproc.fitLine(leftPts, leftLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var lvx = leftLine.get(0, 0)[0]; var lvy = leftLine.get(1, 0)[0]
        if (lvy < 0) { lvx = -lvx; lvy = -lvy }
        val lx0 = firstChar.rect.x.toDouble()
        val ly0 = firstChar.center.y

        val rightPts = MatOfPoint2f(rightTopEdge, rightMidEdge, rightBottomEdge)
        val rightLine = Mat()
        Imgproc.fitLine(rightPts, rightLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var rvx = rightLine.get(0, 0)[0]; var rvy = rightLine.get(1, 0)[0]
        if (rvy < 0) { rvx = -rvx; rvy = -rvy }
        val rx0 = rightX
        val ry0 = lastChar.center.y

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
                midX + scaledX * vx + scaledY * nX,
                midY + scaledX * vy + scaledY * nY
            )
        }

        stepLogs.add("[성공] 대칭 팽창(Scale: 1.35) 적용 완료")

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
            Imgproc.circle(debugMat, Point(midX, midY), 8, Scalar(0.0, 255.0, 255.0, 255.0), -1)

            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "디버그 2/2: 최종 와이어프레임 확정", stepLogs, screenRatio)
            it.pauseAndShowStep("디버그 2/2: 최종 와이어프레임", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
    }
}
