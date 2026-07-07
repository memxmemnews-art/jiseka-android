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

    // 외부에서 비동기로 ML Kit를 호출하여 결과를 받아오기 위한 인터페이스
    interface MLKitScanner {
        suspend fun scanSingleCharacter(bitmap: Bitmap): android.graphics.Rect?
    }

    // 실제 픽셀 윤곽선 기반의 최상단(Top) 및 최하단(Bottom) 좌표 속성 포함
    class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect, var contrast: Double = 0.0, var density: Double = 0.0) {
        val topCenter: Point = Point(center.x, rect.y.toDouble())
        val bottomCenter: Point = Point(center.x, rect.y + height.toDouble())
    }

    data class SeedCropResult(
        val offsetX: Int,
        val offsetY: Int,
        val croppedBitmap: Bitmap,
        val roiRect: Rect
    )

    private data class ClusterStats(
        val medianHeight: Double,
        val medianWidth: Double,
        val sortedChars: List<CharData>
    )

    // ==========================================================================================
    // [공통 유틸리티]
    // ==========================================================================================
    
    // [최적화 적용] 불필요한 List 객체 생성 제거 및 DoubleArray 활용
    private fun calculateClusterStats(cluster: MutableList<CharData>): ClusterStats {
        cluster.sortBy { it.rect.x }
        
        val size = cluster.size
        val heights = DoubleArray(size) { cluster[it].height }
        val widths = DoubleArray(size) { cluster[it].width }
        
        heights.sort()
        widths.sort()
        
        return ClusterStats(
            medianHeight = heights[size / 2],
            medianWidth = widths[size / 2],
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

    private fun addDebugHUD(original: Bitmap, title: String, logs: List<String>, screenRatio: Float, focusY: Float? = null): Bitmap {
        val resultBmp = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBmp)

        val imgW = canvas.width.toFloat()
        val imgH = canvas.height.toFloat()

        val baseTextSize = max(24f, imgW * 0.035f)
        val titleTextSize = baseTextSize * 1.15f
        val padding = imgW * 0.03f
        val lineHeight = baseTextSize * 1.3f
        val maxTextWidth = imgW - (padding * 2)

        val paint = Paint().apply {
            color = Color.WHITE
            textSize = baseTextSize
            isAntiAlias = true
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }

        var totalTextHeight = padding + lineHeight + (padding * 0.5f)
        for (log in logs) {
            var currentLine = ""
            for (word in log.split(" ")) {
                val testLine = if (currentLine.isEmpty()) word else "$currentLine $word"
                if (paint.measureText(testLine) > maxTextWidth && currentLine.isNotEmpty()) {
                    totalTextHeight += lineHeight
                    currentLine = word
                } else {
                    currentLine = testLine
                }
            }
            if (currentLine.isNotEmpty()) totalTextHeight += lineHeight
        }
        totalTextHeight += padding

        var startY = padding
        if (focusY != null) {
            val spaceAbove = focusY - (imgH * 0.1f)
            if (spaceAbove > totalTextHeight) {
                startY = spaceAbove - totalTextHeight
            } else {
                startY = focusY + (imgH * 0.1f)
            }
        }

        startY = startY.coerceIn(padding, max(padding, imgH - totalTextHeight - padding))

        val bgPaint = Paint().apply { color = Color.parseColor("#CC000000") }
        canvas.drawRect(0f, startY, imgW, startY + totalTextHeight, bgPaint)

        var currentY = startY + padding + (titleTextSize * 0.8f)

        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        paint.textSize = titleTextSize
        currentY = drawTextWithWrap(canvas, title, padding, currentY, paint, maxTextWidth, lineHeight)
        currentY += (padding * 0.5f)

        paint.isFakeBoldText = false
        paint.textSize = baseTextSize
        for (log in logs) {
            if (log.startsWith("->") || log.startsWith("[경고]")) paint.color = Color.parseColor("#FF8888")
            else if (log.startsWith("[진단")) paint.color = Color.parseColor("#55FF55")
            else if (log.startsWith("[정보]") || log.startsWith("[기준]")) paint.color = Color.parseColor("#55FFFF")
            else if (log.contains("FAIL") || log.contains("중단") || log.contains("삭제") || log.contains("손절")) paint.color = Color.parseColor("#FF5555")
            else if (log.contains("통과") || log.contains("성공") || log.contains("조립")) paint.color = Color.parseColor("#55FF55")
            else paint.color = Color.WHITE

            currentY = drawTextWithWrap(canvas, log, padding, currentY, paint, maxTextWidth, lineHeight)
        }

        return resultBmp
    }

    private fun getSafeRect(x: Int, y: Int, w: Int, h: Int, maxW: Int, maxH: Int): Rect {
        val safeX = x.coerceIn(0, maxW - 1)
        val safeY = y.coerceIn(0, maxH - 1)
        val safeW = w.coerceAtMost(maxW - safeX)
        val safeH = h.coerceAtMost(maxH - safeY)
        return Rect(safeX, safeY, safeW, safeH)
    }

    // ==========================================================================================
    // 💡 [유틸리티] OpenCV 타이트 바운딩 전담 함수 (블러 연산 최적화 적용)
    // ==========================================================================================
    private fun tightenWithOpenCV(fullGray: Mat, mlKitGlobalRect: Rect, fullCols: Int, fullRows: Int): CharData {
        val pad = (mlKitGlobalRect.width * 0.15).toInt()
        val searchRect = getSafeRect(
            mlKitGlobalRect.x - pad,
            mlKitGlobalRect.y - pad,
            mlKitGlobalRect.width + pad * 2,
            mlKitGlobalRect.height + pad * 2,
            fullCols, fullRows
        )

        val roiGray = Mat()
        fullGray.submat(searchRect).copyTo(roiGray)

        val blurred = Mat()
        // [최적화] medianBlur 삭제, MORPH_CLOSE를 믿고 GaussianBlur만 수행하여 연산량 감소
        Imgproc.GaussianBlur(roiGray, blurred, Size(5.0, 5.0), 0.0)

        val thresh = Mat()
        Imgproc.adaptiveThreshold(blurred, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 19, 12.0)

        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        tempClose.release()

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var minX = Int.MAX_VALUE; var minY = Int.MAX_VALUE
        var maxX = Int.MIN_VALUE; var maxY = Int.MIN_VALUE
        var foundValid = false

        for (contour in contours) {
            val rect = Imgproc.boundingRect(contour)
            if (rect.area() > 30) {
                minX = min(minX, rect.x)
                minY = min(minY, rect.y)
                maxX = max(maxX, rect.x + rect.width)
                maxY = max(maxY, rect.y + rect.height)
                foundValid = true
            }
        }

        roiGray.release(); blurred.release(); thresh.release(); hierarchy.release()
        contours.forEach { it.release() }

        val tightRect = if (foundValid && maxX > minX && maxY > minY) {
            Rect(searchRect.x + minX, searchRect.y + minY, maxX - minX, maxY - minY)
        } else {
            mlKitGlobalRect 
        }

        val globalCenter = Point(tightRect.x + tightRect.width / 2.0, tightRect.y + tightRect.height / 2.0)
        return CharData(globalCenter, tightRect.width.toDouble(), tightRect.height.toDouble(), tightRect)
    }

    // ==========================================================================================
    // 💡 단일 파이프라인 시작점
    // ==========================================================================================
    fun prepareSeedCrop(
        fullBitmap: Bitmap, 
        touchX: Float, touchY: Float, 
        debugListener: DetectionDebugListener? = null
    ): SeedCropResult? {
        val fullMat = Mat(); val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)
        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

        val seedRect = createSeedROI(fullMat, fullGray, touchX.toInt(), touchY.toInt(), debugListener, screenRatio)
        
        val croppedBitmap = Bitmap.createBitmap(fullBitmap, seedRect.x, seedRect.y, seedRect.width, seedRect.height)

        fullMat.release(); fullGray.release()
        return SeedCropResult(seedRect.x, seedRect.y, croppedBitmap, seedRect)
    }

    fun prepareDumbCrop(
        fullBitmap: Bitmap, 
        touchX: Float, touchY: Float, 
        debugListener: DetectionDebugListener? = null
    ): SeedCropResult {
        val cropW = (fullBitmap.width * 0.25f).toInt()
        val cropH = (fullBitmap.height * 0.15f).toInt()

        val safeRect = getSafeRect(
            (touchX - cropW / 2).toInt(),
            (touchY - cropH / 2).toInt(),
            cropW,
            cropH,
            fullBitmap.width, fullBitmap.height
        )

        val croppedBitmap = Bitmap.createBitmap(fullBitmap, safeRect.x, safeRect.y, safeRect.width, safeRect.height)

        debugListener?.let {
            val debugMat = Mat(); Utils.bitmapToMat(fullBitmap, debugMat)
            val screenRatio = debugMat.rows().toFloat() / debugMat.cols().toFloat()
            Imgproc.circle(debugMat, Point(touchX.toDouble(), touchY.toDouble()), 10, Scalar(0.0, 0.0, 255.0, 255.0), -1)
            
            Imgproc.rectangle(debugMat, safeRect, Scalar(255.0, 100.0, 0.0, 255.0), 5)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "[Fallback] 2차 강제 크롭 발동", listOf("[경고] 1차 스마트 탐색 실패", "-> (주황) 터치 주변 강제 고정 영역으로 ML Kit 2차 시도"), screenRatio, touchY)
            it.pauseAndShowStep("디버그 Fallback: 강제 크롭", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return SeedCropResult(safeRect.x, safeRect.y, croppedBitmap, safeRect)
    }

    // ML Kit 결과를 바탕으로 궤도 조립 및 타이트 바운딩 진행 (Suspend 함수)
    suspend fun processWithMLKitResult(
        fullBitmap: Bitmap, 
        offsetX: Int, offsetY: Int, 
        localMlKitBox: android.graphics.Rect, 
        mlKitScanner: MLKitScanner, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        val fullMat = Mat(); val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)
        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

        val globalMlKitBox = Rect(
            offsetX + localMlKitBox.left,
            offsetY + localMlKitBox.top,
            localMlKitBox.width(),
            localMlKitBox.height()
        )

        val seedChar = tightenWithOpenCV(fullGray, globalMlKitBox, fullMat.cols(), fullMat.rows())
        val seedChars = listOf(seedChar)

        debugListener?.let {
            val debugMat = fullMat.clone()
            
            val oldRect = Rect(globalMlKitBox.x, globalMlKitBox.y, globalMlKitBox.width, globalMlKitBox.height)
            Imgproc.rectangle(debugMat, oldRect, Scalar(255.0, 0.0, 0.0, 255.0), 2) 
            Imgproc.rectangle(debugMat, seedChar.rect, Scalar(0.0, 255.0, 0.0, 255.0), 4)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val logs = listOf(
                "-> (빨강) 기존 ML Kit 원본 박스 (엉성함)",
                "-> (초록) OpenCV로 재계산된 타이트한 바운드 박스",
                "[진단] 정확한 중심/상단/하단 좌표 확보 완료. 레이더 궤도 정상화."
            )
            val hudBmp = addDebugHUD(debugBmp, "[2/9] Seed 문자 타이트 바운딩", logs, screenRatio, seedChar.center.y.toFloat())
            it.pauseAndShowStep("디버그 2/9: 타이트 Seed", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        val collectedChars = expandAndCollect(seedChars, fullBitmap, fullMat, fullGray, mlKitScanner, debugListener, screenRatio)
        
        var resultPoints: List<ImmutablePoint>? = null
        if (collectedChars.size >= 4) { 
            resultPoints = buildWireframe(collectedChars, fullMat, debugListener, screenRatio)
        }

        fullMat.release(); fullGray.release()
        return resultPoints
    }

    private fun createSeedROI(
        fullMat: Mat, fullGray: Mat, cx: Int, cy: Int, 
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): Rect {
        val cropSize = (fullMat.cols() * 0.06).toInt() 
        val currentX = cx - cropSize / 2
        val currentY = cy - cropSize / 2

        val finalRect = getSafeRect(currentX, currentY, cropSize, cropSize, fullMat.cols(), fullMat.rows())

        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.circle(debugMat, Point(cx.toDouble(), cy.toDouble()), 10, Scalar(0.0, 0.0, 255.0, 255.0), -1)
            Imgproc.rectangle(debugMat, finalRect, Scalar(0.0, 255.0, 0.0, 255.0), 5)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "[1/9] 고정 크기 Seed ROI 생성 (스마트 확장 제거)", listOf("-> (초록) 사용자의 2차 터치(조준) 지점 기준 6% 고정 크롭 영역", "[진단] OpenCV 개입 없이 이 영역 내에서 ML Kit가 단일 문자를 직접 찾습니다."), screenRatio, cy.toFloat())
            it.pauseAndShowStep("디버그 1/9: 고정 ROI", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return finalRect
    }

    private suspend fun expandAndCollect(
        seeds: List<CharData>, fullBitmap: Bitmap, fullMat: Mat, fullGray: Mat, 
        mlKitScanner: MLKitScanner,
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<CharData> {
        
        val currentCluster = seeds.toMutableList()
        val stepLogs = mutableListOf<String>()
        stepLogs.add("[진단] 초기 Seed 확보 완료. 방향별 통합 확장 루프 기동.")

        expandOneDirection(currentCluster, fullBitmap, fullMat, fullGray, true, mlKitScanner, stepLogs, debugListener, screenRatio)
        expandOneDirection(currentCluster, fullBitmap, fullMat, fullGray, false, mlKitScanner, stepLogs, debugListener, screenRatio)

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

        debugListener?.let {
            val debugMat = fullMat.clone()
            currentCluster.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(0.0, 255.0, 255.0, 255.0), 3) }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val focusY = currentCluster.firstOrNull()?.center?.y?.toFloat()
            val hudBmp = addDebugHUD(debugBmp, "[7/9] 글로벌 대청소 (노이즈/볼트 제거)", stepLogs, screenRatio, focusY)
            it.pauseAndShowStep("디버그 7/9: 클러스터 정돈", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return currentCluster
    }

    // ==========================================================================================
    // 💡 [핵심 엔진] 3점 기울기 기반 예측 -> ML Kit로 텍스트 확인 -> OpenCV 타이트 바운딩
    // ==========================================================================================
    private suspend fun expandOneDirection(
        currentCluster: MutableList<CharData>, fullBitmap: Bitmap, fullMat: Mat, fullGray: Mat, 
        isLeft: Boolean, mlKitScanner: MLKitScanner, stepLogs: MutableList<String>,
        debugListener: DetectionDebugListener?, screenRatio: Float
    ) {
        var expand = true
        var loops = 0
        val dirStr = if (isLeft) "좌측" else "우측"

        var stats = calculateClusterStats(currentCluster)

        while (expand && loops < 8) {
            loops++
            
            val cachedCluster = stats.sortedChars
            val medianH = stats.medianHeight
            val medianW = stats.medianWidth
            
            val outerChar = if (isLeft) cachedCluster.first() else cachedCluster.last()
            
            val searchW = (medianW * 1.5).toInt()
            val searchX = if (isLeft) outerChar.rect.x - searchW else outerChar.rect.x + outerChar.rect.width
            
            var topSlope = 0.0; var centerSlope = 0.0; var bottomSlope = 0.0
            
            if (cachedCluster.size >= 2) {
                val innerChar = if (isLeft) cachedCluster.last() else cachedCluster.first()
                val dx = outerChar.center.x - innerChar.center.x
                
                if (abs(dx) > 1.0) {
                    topSlope = (outerChar.topCenter.y - innerChar.topCenter.y) / dx
                    centerSlope = (outerChar.center.y - innerChar.center.y) / dx
                    bottomSlope = (outerChar.bottomCenter.y - innerChar.bottomCenter.y) / dx
                }
            }

            val expectedDistX = if (isLeft) -medianW else medianW
            val predictedTopY = outerChar.topCenter.y + topSlope * expectedDistX
            val predictedBottomY = outerChar.bottomCenter.y + bottomSlope * expectedDistX

            val padY = (medianH * 0.15).toInt()
            val finalSearchY = predictedTopY.toInt() - padY
            val finalSearchH = max(10, (predictedBottomY - predictedTopY).toInt() + (padY * 2))

            val searchRect = getSafeRect(searchX, finalSearchY, searchW, finalSearchH, fullMat.cols(), fullMat.rows())
            if (searchRect.width < 10) break

            debugListener?.let {
                val debugMat = fullMat.clone()
                cachedCluster.forEach { c -> Imgproc.rectangle(debugMat, c.rect, Scalar(150.0, 150.0, 150.0, 255.0), 2) }
                Imgproc.rectangle(debugMat, searchRect, Scalar(255.0, 0.0, 255.0, 255.0), 3)
                
                Imgproc.line(debugMat, Point(outerChar.center.x, predictedTopY), Point(outerChar.center.x + expectedDistX, predictedTopY), Scalar(0.0, 255.0, 0.0, 255.0), 2)
                Imgproc.line(debugMat, Point(outerChar.center.x, predictedBottomY), Point(outerChar.center.x + expectedDistX, predictedBottomY), Scalar(0.0, 255.0, 0.0, 255.0), 2)
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                
                val stepNum = if (isLeft) "3" else "5"
                val hudBmp = addDebugHUD(debugBmp, "[$stepNum/9] $dirStr 3점 기울기 궤도 투사", listOf("-> (마젠타) 1글자 너비 제한 탐색 박스", "-> (초록선) 이전 문자와 시드 문자를 이은 예측 기울기"), screenRatio, searchRect.y.toFloat())
                it.pauseAndShowStep("디버그 $stepNum/9: $dirStr 궤도", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // 비동기로 ML Kit 호출하여 문자 인식 수행
            val probeBmp = Bitmap.createBitmap(fullBitmap, searchRect.x, searchRect.y, searchRect.width, searchRect.height)
            val localMlKitBox = mlKitScanner.scanSingleCharacter(probeBmp)
            probeBmp.recycle()

            var foundValid = false

            if (localMlKitBox != null) {
                val globalMlKitRect = Rect(
                    searchRect.x + localMlKitBox.left,
                    searchRect.y + localMlKitBox.top,
                    localMlKitBox.width(),
                    localMlKitBox.height()
                )

                // ML Kit가 찾은 박스를 OpenCV로 타이트하게 조이기
                val tightChar = tightenWithOpenCV(fullGray, globalMlKitRect, fullMat.cols(), fullMat.rows())
                
                if (isValidNextChar(tightChar, cachedCluster, stepLogs, dirStr, centerSlope, topSlope, bottomSlope)) {
                    currentCluster.add(tightChar)
                    stats = calculateClusterStats(currentCluster) 
                    foundValid = true
                    stepLogs.add(" -> [$dirStr 연결] ML Kit 탐색 성공 -> OpenCV 타이트 바운딩 완료")
                }
            } else {
                stepLogs.add(" -> [$dirStr 중단] ML Kit가 해당 영역에서 문자를 찾지 못함")
            }

            if (!foundValid) expand = false
        }

        debugListener?.let {
            val debugMat = fullMat.clone()
            currentCluster.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(255.0, 255.0, 0.0, 255.0), 3) }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val stepNum = if (isLeft) "4" else "6"
            val focusY = currentCluster.firstOrNull()?.center?.y?.toFloat()
            val hudBmp = addDebugHUD(debugBmp, "[$stepNum/9] $dirStr 확장 루프 완료", stepLogs.takeLast(3), screenRatio, focusY)
            it.pauseAndShowStep("디버그 $stepNum/9: $dirStr 완료", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }
    }

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

        val expectedTopY = outerChar.topCenter.y + topSlope * distToNext
        val expectedBottomY = outerChar.bottomCenter.y + bottomSlope * distToNext
        val expectedCenterY = outerChar.center.y + centerSlope * distToNext

        val topDeviation = abs(newChar.topCenter.y - expectedTopY)
        val bottomDeviation = abs(newChar.bottomCenter.y - expectedBottomY)
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
            
            val focusY = validChars.firstOrNull()?.center?.y?.toFloat()
            val hudBmp = addDebugHUD(debugBmp, "[8/9] 선형 궤도 검증 & KOR 마크 삭제", stepLogs, screenRatio, focusY)
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
            
            val hudBmp = addDebugHUD(debugBmp, "[9/9] 디버깅 완료: 최종 와이어프레임 기하학 도출", listOf("-> (초록 테두리) 1.35배 대칭 팽창된 최종 번호판 영역", "[진단] 이 영역이 PerspectiveTransform 을 통해 최종 크롭될 영역입니다."), screenRatio, midY.toFloat())
            it.pauseAndShowStep("디버그 9/9: 최종 렌더링", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        return finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
    }
}
