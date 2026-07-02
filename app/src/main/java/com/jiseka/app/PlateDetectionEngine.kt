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

    // 💡 [추가됨] 캐싱용 통계 데이터 클래스
    private data class ClusterStats(
        val medianHeight: Double,
        val medianWidth: Double,
        val sortedChars: List<CharData>
    )

    // ==========================================================================================
    // [공통 유틸리티]
    // ==========================================================================================
    private fun calculateClusterStats(cluster: MutableList<CharData>): ClusterStats {
        cluster.sortBy { it.rect.x }
        val heights = cluster.map { it.height }.sorted()
        val widths = cluster.map { it.width }.sorted()
        return ClusterStats(
            medianHeight = heights[heights.size / 2],
            medianWidth = widths[widths.size / 2],
            sortedChars = cluster.toList()
        )
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

    private fun checkLineIntersection(mat: Mat, startRow: Int, endRow: Int, pixelThreshold: Double): Boolean {
        val sub = mat.submat(startRow, endRow + 1, 0, mat.cols())
        val whitePixels = Core.countNonZero(sub)
        sub.release()
        val thickness = endRow - startRow + 1
        return whitePixels > (pixelThreshold * thickness)
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

        val seedChars = extractSeedChars(fullMat, fullGray, touchX.toInt(), touchY.toInt(), debugListener, screenRatio)
        if (seedChars.isEmpty()) {
            fullMat.release(); fullGray.release()
            return null
        }

        val collectedChars = expandAndCollect(seedChars, fullMat, fullGray, debugListener, screenRatio)
        
        var resultPoints: List<ImmutablePoint>? = null
        if (collectedChars.size >= 4) { 
            resultPoints = buildWireframe(collectedChars, fullMat, debugListener, screenRatio)
        }

        fullMat.release(); fullGray.release()
        return resultPoints
    }

    // ==========================================================================================
    // [엔진 1] ExpansionEngine: 초기 씨앗 추출 (디버그 1/9, 2/9)
    // ==========================================================================================
    private fun extractSeedChars(
        fullMat: Mat, fullGray: Mat, cx: Int, cy: Int, 
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<CharData> {
        
        var roiWidth = (fullMat.cols() * 0.05).toInt() 
        var roiHeight = (roiWidth * 1.5).toInt()       
        var currentX = cx - roiWidth / 2
        var currentY = cy - roiHeight / 2

        val maxExpansions = 5 
        val pad = (fullMat.cols() * 0.015).toInt() 

        var finalRect = getSafeRect(currentX, currentY, roiWidth, roiHeight, fullMat.cols(), fullMat.rows())
        val initialRect = finalRect.clone()

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
                if (ratio < 0.2 || ratio > 6.0 || rect.width > (fullMat.cols() * 0.3)) continue 

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

        // 💡 [디버그 1/9] Seed 탐색 ROI 확인
        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.circle(debugMat, Point(cx.toDouble(), cy.toDouble()), 10, Scalar(0.0, 0.0, 255.0, 255.0), -1)
            Imgproc.rectangle(debugMat, initialRect, Scalar(255.0, 255.0, 0.0, 255.0), 2)
            Imgproc.rectangle(debugMat, finalRect, Scalar(0.0, 255.0, 0.0, 255.0), 5)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "[1/9] 터치 인식 및 Seed ROI 확장", listOf("-> (노랑) 최초 설정 ROI", "-> (초록) 윤곽선 팽창 알고리즘 적용 후 최종 ROI", "[진단] 타겟 문자가 초록 박스 안에 온전히 들어왔는지 확인하세요."), screenRatio)
            it.pauseAndShowStep("디버그 1/9: Seed ROI", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // 💡 단일 전처리 파이프라인으로 추출 
        val params = listOf(Pair(31, 7.0), Pair(19, 15.0))
        val seeds = scanRegion(fullMat, fullGray, finalRect, params)

        // 💡 [디버그 2/9] Seed 추출 결과 확인
        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.rectangle(debugMat, finalRect, Scalar(255.0, 255.0, 255.0, 100.0), 2)
            seeds.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(0.0, 255.0, 0.0, 255.0), 3) }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "[2/9] 초기 Seed 문자 추출 완료", listOf("[진단] ${seeds.size}개의 초기 파편 획득", "-> 파편들이 문자의 형태를 잘 잡았는지 확인하세요."), screenRatio)
            it.pauseAndShowStep("디버그 2/9: Seed 결과", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return seeds
    }

    // ==========================================================================================
    // [엔진 2] ExpansionEngine: 방향별 확장 제어기 (디버그 7/9)
    // ==========================================================================================
    private fun expandAndCollect(
        seeds: List<CharData>, fullMat: Mat, fullGray: Mat, 
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<CharData> {
        
        val currentCluster = seeds.toMutableList()
        val stepLogs = mutableListOf<String>()
        stepLogs.add("[진단] 초기 Seed 확보 완료. 방향별 통합 확장 루프 기동.")

        expandOneDirection(currentCluster, fullMat, fullGray, true, stepLogs, debugListener, screenRatio)
        expandOneDirection(currentCluster, fullMat, fullGray, false, stepLogs, debugListener, screenRatio)

        currentCluster.sortBy { it.rect.x }
        
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
            if (purgedCount > 0) stepLogs.add(" -> [글로벌 청소] 기준 미달 노이즈 ${purgedCount}개 일괄 삭제")
        }

        // 💡 [디버그 7/9] 글로벌 대청소 완료
        debugListener?.let {
            val debugMat = fullMat.clone()
            currentCluster.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(0.0, 255.0, 255.0, 255.0), 3) }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "[7/9] 글로벌 대청소 (노이즈/볼트 제거)", stepLogs, screenRatio)
            it.pauseAndShowStep("디버그 7/9: 클러스터 정돈", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return currentCluster
    }

    // ==========================================================================================
    // [단일 방향 탐색기] 캐싱 최적화 + 디버그 3, 4, 5, 6
    // ==========================================================================================
    private fun expandOneDirection(
        currentCluster: MutableList<CharData>, fullMat: Mat, fullGray: Mat, 
        isLeft: Boolean, stepLogs: MutableList<String>,
        debugListener: DetectionDebugListener?, screenRatio: Float
    ) {
        var expand = true
        var loops = 0
        val dirStr = if (isLeft) "좌측" else "우측"
        var isFirstRadarCast = true

        // 💡 [최초 1회 캐싱]
        var stats = calculateClusterStats(currentCluster)

        while (expand && loops < 8) {
            loops++
            
            val cachedCluster = stats.sortedChars
            val medianH = stats.medianHeight
            val medianW = stats.medianWidth
            
            val outerChar = if (isLeft) cachedCluster.first() else cachedCluster.last()
            
            var topSlope = 0.0; var centerSlope = 0.0; var bottomSlope = 0.0
            if (cachedCluster.size >= 2) {
                val innerChar = if (isLeft) cachedCluster[1] else cachedCluster[cachedCluster.size - 2]
                val dx = outerChar.center.x - innerChar.center.x
                if (abs(dx) > 1.0) {
                    topSlope = (outerChar.rect.y - innerChar.rect.y) / dx
                    centerSlope = (outerChar.center.y - innerChar.center.y) / dx
                    bottomSlope = ((outerChar.rect.y + outerChar.rect.height) - (innerChar.rect.y + innerChar.rect.height)) / dx
                }
            }
            
            val searchW = (medianW * 2.5).toInt()
            val searchX = if (isLeft) outerChar.rect.x - searchW else outerChar.rect.x + outerChar.rect.width
            
            val probeRect = getSafeRect(searchX, outerChar.rect.y, searchW, outerChar.rect.height, fullMat.cols(), fullMat.rows())
            if (probeRect.width < 10) break

            // 💡 [디버그 3/9 (좌) or 5/9 (우)] 최초 3선 레이더 투사 
            if (isFirstRadarCast && probeRect.width >= 10) {
                isFirstRadarCast = false
                debugListener?.let {
                    val debugMat = fullMat.clone()
                    cachedCluster.forEach { c -> Imgproc.rectangle(debugMat, c.rect, Scalar(150.0, 150.0, 150.0, 255.0), 2) }
                    Imgproc.rectangle(debugMat, probeRect, Scalar(255.0, 0.0, 255.0, 255.0), 3)
                    
                    val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(debugMat, debugBmp)
                    
                    val stepNum = if (isLeft) "3" else "5"
                    val hudBmp = addDebugHUD(debugBmp, "[$stepNum/9] $dirStr 3단 레이더 투사", listOf("-> (마젠타) 관통 레이더 박스 생성", "[진단] 문자가 있을 것으로 예상되는 곳에 박스가 투사되었는지 확인"), screenRatio)
                    it.pauseAndShowStep("디버그 $stepNum/9: $dirStr 레이더", hudBmp)
                    debugMat.release(); debugBmp.recycle()
                }
            }

            // 첫 번째 이진화 (레이더용)
            val probeMat = Mat()
            fullGray.submat(probeRect).copyTo(probeMat)
            Imgproc.adaptiveThreshold(probeMat, probeMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 19, 12.0)

            var finalSearchY = outerChar.rect.y
            var finalSearchH = outerChar.rect.height

            if (probeMat.rows() >= 5) {
                val occupancyThreshold = max(medianW * 0.20, 6.0)

                val tRowStart = (probeMat.rows() * 0.10).toInt().coerceIn(0, probeMat.rows() - 1)
                val tRowEnd = (probeMat.rows() * 0.20).toInt().coerceIn(tRowStart, probeMat.rows() - 1)
                
                val mRowStart = (probeMat.rows() * 0.45).toInt().coerceIn(0, probeMat.rows() - 1)
                val mRowEnd = (probeMat.rows() * 0.55).toInt().coerceIn(mRowStart, probeMat.rows() - 1)
                
                val bRowStart = (probeMat.rows() * 0.80).toInt().coerceIn(0, probeMat.rows() - 1)
                val bRowEnd = (probeMat.rows() * 0.90).toInt().coerceIn(bRowStart, probeMat.rows() - 1)

                val topHit = checkLineIntersection(probeMat, tRowStart, tRowEnd, occupancyThreshold)
                val midHit = checkLineIntersection(probeMat, mRowStart, mRowEnd, occupancyThreshold)
                val botHit = checkLineIntersection(probeMat, bRowStart, bRowEnd, occupancyThreshold)

                if (topHit && midHit && !botHit) {
                    stepLogs.add(" -> [$dirStr 3선 레이더] 상단 O / 중앙 O / 하단 X ➔ 위로 상승 상자 배치")
                    finalSearchY = outerChar.rect.y - (medianH * 0.5).toInt()
                    finalSearchH = (medianH * 1.3).toInt()
                } else if (!topHit && midHit && botHit) {
                    stepLogs.add(" -> [$dirStr 3선 레이더] 상단 X / 중앙 O / 하단 O ➔ 아래로 하강 상자 배치")
                    finalSearchY = outerChar.rect.y - (medianH * 0.1).toInt()
                    finalSearchH = (medianH * 1.4).toInt()
                } else {
                    val pad = (medianH * 0.15).toInt()
                    finalSearchY = outerChar.rect.y - pad
                    finalSearchH = outerChar.rect.height + pad * 2
                }
            }
            probeMat.release()

            val searchRect = getSafeRect(searchX, finalSearchY, searchW, finalSearchH, fullMat.cols(), fullMat.rows())
            
            // 💡 단일 전처리 파이프라인으로 추출 
            val params = listOf(Pair(31, 7.0), Pair(19, 15.0))
            val newCandidates = scanRegion(fullMat, fullGray, searchRect, params)

            val sortedCandidates = if (isLeft) newCandidates.sortedByDescending { it.rect.x } else newCandidates.sortedBy { it.rect.x }
            var foundValid = false

            for (candidate in sortedCandidates) {
                if (cachedCluster.any { abs(it.center.x - candidate.center.x) < it.width * 0.5 }) continue
                
                if (isValidNextChar(candidate, cachedCluster, stepLogs, dirStr, centerSlope, topSlope, bottomSlope)) {
                    currentCluster.add(candidate)
                    stats = calculateClusterStats(currentCluster) // 💡 [상태 갱신] 추가 시 캐시 재계산
                    foundValid = true
                    stepLogs.add(" -> [$dirStr 연결] X:${candidate.center.x.toInt()} 문자 추가 성공")
                    break
                }
            }
            if (!foundValid) expand = false
        }

        // 💡 [디버그 4/9 (좌) or 6/9 (우)] 탐색 완료 결과
        debugListener?.let {
            val debugMat = fullMat.clone()
            currentCluster.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(255.0, 255.0, 0.0, 255.0), 3) }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val stepNum = if (isLeft) "4" else "6"
            val hudBmp = addDebugHUD(debugBmp, "[$stepNum/9] $dirStr 확장 루프 완료", stepLogs.takeLast(3), screenRatio)
            it.pauseAndShowStep("디버그 $stepNum/9: $dirStr 완료", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }
    }

    // ==========================================================================================
    // [엔진 2-1] 4대 하드 스톱 방어선
    // ==========================================================================================
    private fun isValidNextChar(
        newChar: CharData, currentCluster: List<CharData>, logs: MutableList<String>, dir: String,
        centerSlope: Double, topSlope: Double, bottomSlope: Double
    ): Boolean {
        if (currentCluster.isEmpty()) return true
        
        val sortedArea = currentCluster.map { it.rect.area() }.sorted()
        val sortedHeight = currentCluster.map { it.height }.sorted()
        val medianArea = sortedArea[currentCluster.size / 2]
        val medianH = sortedHeight[currentCluster.size / 2]

        val outerChar = if (dir == "좌측") currentCluster.first() else currentCluster.last()
        val distToNext = newChar.center.x - outerChar.center.x

        val expectedTopY = outerChar.rect.y + topSlope * distToNext
        val expectedBottomY = (outerChar.rect.y + outerChar.rect.height) + bottomSlope * distToNext
        val expectedCenterY = outerChar.center.y + centerSlope * distToNext

        val topDeviation = abs(newChar.rect.y - expectedTopY)
        val bottomDeviation = abs((newChar.rect.y + newChar.rect.height) - expectedBottomY)
        val centerDeviation = abs(newChar.center.y - expectedCenterY)

        val isBoundsDeviated = topDeviation > (medianH * 0.35) || bottomDeviation > (medianH * 0.35)

        if (newChar.rect.area() < medianArea * 0.35 && isBoundsDeviated) {
            logs.add(" -> [$dir 중단] 볼트/먼지 차단 (면적 급감 + 3단 궤도 이탈)")
            return false 
        }
        if (newChar.rect.area() > medianArea * 2.5 && isBoundsDeviated) {
            logs.add(" -> [$dir 중단] 범퍼/프레임 차단 (면적 급증 + 3단 궤도 이탈)")
            return false
        }
        if (centerDeviation > medianH * 0.6) {
            logs.add(" -> [$dir 중단] 중심점 3단 궤도 완전 이탈")
            return false
        }
        if (newChar.height < medianH * 0.50 || newChar.height > medianH * 1.50) {
            logs.add(" -> [$dir 중단] 원근 오차를 넘는 높이 급변 차단")
            return false
        }
        return true
    }

    // ==========================================================================================
    // [엔진 3] CharacterScanner: 단일 전처리 캐싱 구조 
    // ==========================================================================================
    private fun scanRegion(
        fullMat: Mat, fullGray: Mat, roi: Rect, 
        thresholdParams: List<Pair<Int, Double>> 
    ): List<CharData> {
        val roiGray = Mat()
        fullGray.submat(roi).copyTo(roiGray)
        
        // 💡 [블러 전처리 1회 수행 및 캐싱]
        val blurred = Mat()
        Imgproc.medianBlur(roiGray, roiGray, 3)
        Imgproc.GaussianBlur(roiGray, blurred, Size(5.0, 5.0), 0.0)
        roiGray.release() 

        val finalCandidates = mutableListOf<CharData>()

        for ((blockSize, cValue) in thresholdParams) {
            val thresh = Mat()
            
            // 캐싱된 blurred 이미지로 이진화만 재시도
            Imgproc.adaptiveThreshold(blurred, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, blockSize, cValue)

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
                
                if (ratio in 0.15..7.0) {
                    val globalRect = Rect(localRect.x + roi.x, localRect.y + roi.y, localRect.width, localRect.height)
                    val globalCenter = Point(globalRect.x + globalRect.width / 2.0, globalRect.y + globalRect.height / 2.0)
                    rawBlobs.add(CharData(globalCenter, globalRect.width.toDouble(), globalRect.height.toDouble(), globalRect))
                }
            }
            
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

                val isVerticalSplit = xOverlap > min(curr.width, next.width) * 0.3 && 
                                      yGap < (curr.height * 0.45) &&
                                      (currRatio < 1.0 || nextRatio < 1.0 || currRatio > 1.2)
                
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

            for (blob in rawBlobs) {
                val ratio = blob.height / max(blob.width, 1.0)
                if (ratio in 0.25..5.5 && blob.height >= 12) {
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
            
            hierarchy.release()
            contours.forEach { it.release() }
            thresh.release()
            
            // 💡 후보를 찾았다면 루프 즉시 탈출 (불필요한 2차 시도 방지)
            if (finalCandidates.isNotEmpty()) {
                break
            }
        }
        
        blurred.release()
        return finalCandidates
    }

    // ==========================================================================================
    // [엔진 4] GeometryBuilder: 최종 검증 및 와이어프레임 계산 (디버그 8/9, 9/9)
    // ==========================================================================================
    private fun buildWireframe(
        collectedChars: List<CharData>, fullMat: Mat, 
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<ImmutablePoint>? {
        
        val stepLogs = mutableListOf<String>()
        stepLogs.add("[진단] 와이어프레임 기하학 조립 전 최종 검증 시작")

        var validChars = collectedChars.toMutableList()

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

        // 💡 [디버그 8/9] 선형 궤도 및 KOR 마크 검증 완료
        debugListener?.let {
            val debugMat = fullMat.clone()
            validChars.forEach { char -> 
                Imgproc.rectangle(debugMat, char.rect, Scalar(255.0, 100.0, 100.0, 255.0), 3) 
                Imgproc.circle(debugMat, char.center, 5, Scalar(0.0, 255.0, 0.0, 255.0), -1)
            }
            if (validChars.size >= 2) {
                Imgproc.line(debugMat, validChars.first().center, validChars.last().center, Scalar(0.0, 255.0, 255.0, 255.0), 2)
            }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "[8/9] 선형 궤도 검증 & KOR 마크 삭제", stepLogs, screenRatio)
            it.pauseAndShowStep("디버그 8/9: 최종 궤도 검증", hudBmp)
            debugMat.release(); debugBmp.recycle()
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

        // 💡 [디버그 9/9] 최종 와이어프레임 적용
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
            val hudBmp = addDebugHUD(debugBmp, "[9/9] 디버깅 완료: 최종 와이어프레임 기하학 도출", listOf("-> (초록 테두리) 1.35배 대칭 팽창된 최종 번호판 영역", "[진단] 이 영역이 PerspectiveTransform 을 통해 최종 크롭될 영역입니다."), screenRatio)
            it.pauseAndShowStep("디버그 9/9: 최종 렌더링", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
    }
}
