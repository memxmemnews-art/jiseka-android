package com.jiseka.app

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(stageName: String, bitmap: Bitmap, title: String, logs: List<String>)
    }

    interface MLKitScanner {
        suspend fun scanCharacters(bitmap: Bitmap): List<android.graphics.Rect>
    }

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

    private fun getSafeRect(x: Int, y: Int, w: Int, h: Int, maxW: Int, maxH: Int): Rect {
        val safeX = x.coerceIn(0, maxW - 1)
        val safeY = y.coerceIn(0, maxH - 1)
        val safeW = w.coerceAtMost(maxW - safeX)
        val safeH = h.coerceAtMost(maxH - safeY)
        return Rect(safeX, safeY, safeW, safeH)
    }

    private fun tightenWithOpenCV(fullGray: Mat, mlKitGlobalRect: Rect, fullCols: Int, fullRows: Int): CharData {
        val padY = (mlKitGlobalRect.height * 0.10).toInt()
        val padX = (mlKitGlobalRect.width * 0.02).toInt()
        
        val searchRect = getSafeRect(
            mlKitGlobalRect.x - padX,
            mlKitGlobalRect.y - padY,
            mlKitGlobalRect.width + padX * 2,
            mlKitGlobalRect.height + padY * 2,
            fullCols, fullRows
        )

        val roiGray = Mat()
        fullGray.submat(searchRect).copyTo(roiGray)

        val blurred = Mat()
        Imgproc.GaussianBlur(roiGray, blurred, Size(5.0, 5.0), 0.0)

        val thresh = Mat()
        Imgproc.adaptiveThreshold(blurred, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 19, 12.0)

        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        tempClose.release()

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val mlKitLocalCenterX = padX + mlKitGlobalRect.width / 2.0
        
        var minX = Int.MAX_VALUE; var minY = Int.MAX_VALUE
        var maxX = Int.MIN_VALUE; var maxY = Int.MIN_VALUE
        var foundValid = false

        for (contour in contours) {
            val rect = Imgproc.boundingRect(contour)
            if (rect.area() > 20) {
                val contourCenterX = rect.x + rect.width / 2.0
                if (abs(contourCenterX - mlKitLocalCenterX) <= mlKitGlobalRect.width * 0.6) {
                    minX = min(minX, rect.x)
                    minY = min(minY, rect.y)
                    maxX = max(maxX, rect.x + rect.width)
                    maxY = max(maxY, rect.y + rect.height)
                    foundValid = true
                }
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
            Imgproc.circle(debugMat, Point(touchX.toDouble(), touchY.toDouble()), 10, Scalar(0.0, 0.0, 255.0, 255.0), -1)
            Imgproc.rectangle(debugMat, safeRect, Scalar(255.0, 100.0, 0.0, 255.0), 5)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            it.pauseAndShowStep(
                "디버그 Fallback: 강제 크롭", debugBmp,
                "[Fallback] 2차 강제 크롭 발동",
                listOf("[경고] 1차 스마트 탐색 실패", "-> (주황) 터치 주변 강제 고정 영역으로 ML Kit 2차 시도")
            )
            debugMat.release(); debugBmp.recycle()
        }

        return SeedCropResult(safeRect.x, safeRect.y, croppedBitmap, safeRect)
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
            
            it.pauseAndShowStep(
                "디버그 1/9: 고정 ROI", debugBmp,
                "[1/9] 고정 크기 Seed ROI 생성",
                listOf("-> (초록) 사용자의 터치 지점 기준 고정 크롭 영역", "[진단] 이 영역 내에서 ML Kit가 단일 문자를 찾습니다.")
            )
            debugMat.release(); debugBmp.recycle()
        }

        return finalRect
    }

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

        val seedGlobalBox = android.graphics.Rect(
            offsetX + localMlKitBox.left,
            offsetY + localMlKitBox.top,
            offsetX + localMlKitBox.right,
            offsetY + localMlKitBox.bottom
        )

        val allMlKitBoxes = collectMLKitCharacters(seedGlobalBox, fullBitmap, mlKitScanner, debugListener, fullMat.cols(), fullMat.rows())

        val collectedChars = mutableListOf<CharData>()
        for (box in allMlKitBoxes) {
            val cvRect = org.opencv.core.Rect(box.left, box.top, box.width(), box.height())
            val tightChar = tightenWithOpenCV(fullGray, cvRect, fullMat.cols(), fullMat.rows())
            collectedChars.add(tightChar)
        }

        if (collectedChars.size > 2) {
            collectedChars.sortBy { it.rect.x }
            val heightsDesc = collectedChars.map { it.height }.sortedDescending()
            val trueHeight = if (heightsDesc.size >= 3) heightsDesc[2] else heightsDesc.first()
            val purgeHeightLimit = trueHeight * 0.50
            
            val purgeIterator = collectedChars.iterator()
            var purgedCount = 0
            while (purgeIterator.hasNext()) {
                val c = purgeIterator.next()
                if (c.height < purgeHeightLimit) {
                    purgeIterator.remove()
                    purgedCount++
                }
            }
        }

        debugListener?.let {
            val debugMat = fullMat.clone()
            collectedChars.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(0.0, 255.0, 255.0, 255.0), 3) }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            it.pauseAndShowStep(
                "디버그 7/9: 타이트 바운딩 일괄 적용", debugBmp,
                "[7/9] ML Kit 탐색 완료 후 일괄 OpenCV 적용",
                listOf("-> (노랑) ${collectedChars.size}개 문자에 대해 일괄 타이트 바운딩 완료")
            )
            debugMat.release(); debugBmp.recycle()
        }

        var resultPoints: List<ImmutablePoint>? = null
        if (collectedChars.size >= 4) { 
            resultPoints = buildWireframe(collectedChars, fullMat, debugListener, screenRatio)
        }

        fullMat.release(); fullGray.release()
        return resultPoints
    }

    private suspend fun collectMLKitCharacters(
        seedGlobalBox: android.graphics.Rect,
        fullBitmap: Bitmap,
        mlKitScanner: MLKitScanner,
        debugListener: DetectionDebugListener?,
        fullW: Int, fullH: Int
    ): List<android.graphics.Rect> {
        val stepLogs = mutableListOf<String>()
        stepLogs.add("[진단] ML Kit 전용 방향별 탐색 루프 기동.")

        val visited = mutableListOf<android.graphics.Rect>()
        visited.add(seedGlobalBox)

        val leftBoxes = expandOneDirectionMLKit(seedGlobalBox, fullBitmap, true, mlKitScanner, visited, stepLogs, debugListener, fullW, fullH)
        val rightBoxes = expandOneDirectionMLKit(seedGlobalBox, fullBitmap, false, mlKitScanner, visited, stepLogs, debugListener, fullW, fullH)

        val allBoxes = mutableListOf<android.graphics.Rect>()
        allBoxes.addAll(leftBoxes.reversed())
        allBoxes.add(seedGlobalBox)
        allBoxes.addAll(rightBoxes)

        return allBoxes
    }

    private suspend fun expandOneDirectionMLKit(
        seedBox: android.graphics.Rect, fullBitmap: Bitmap, 
        isLeft: Boolean, mlKitScanner: MLKitScanner, visited: MutableList<android.graphics.Rect>, stepLogs: MutableList<String>,
        debugListener: DetectionDebugListener?, fullW: Int, fullH: Int
    ): List<android.graphics.Rect> {
        val result = mutableListOf<android.graphics.Rect>()
        val recentBoxes = mutableListOf<android.graphics.Rect>(seedBox) 
        
        var currentBox = seedBox
        var expand = true
        var loops = 0
        val dirStr = if (isLeft) "좌측" else "우측"

        while (expand && loops < 8) {
            loops++
            val searchRect = buildSearchROI(currentBox, recentBoxes, isLeft, fullW, fullH)
            if (searchRect.width < 10) break

            val probeBmp = Bitmap.createBitmap(fullBitmap, searchRect.x, searchRect.y, searchRect.width, searchRect.height)
            val localBoxes = mlKitScanner.scanCharacters(probeBmp)
            probeBmp.recycle()

            if (localBoxes.isNotEmpty()) {
                val currentCenter = Point(currentBox.exactCenterX().toDouble(), currentBox.exactCenterY().toDouble())
                val bestCandidate = chooseNearestCandidate(currentCenter, localBoxes, searchRect, isLeft, visited)

                if (bestCandidate != null) {
                    result.add(bestCandidate)
                    visited.add(bestCandidate)
                    recentBoxes.add(bestCandidate) 
                    currentBox = bestCandidate
                    stepLogs.add(" -> [$dirStr 확장] ML Kit 후보 선택 완료")
                } else {
                    stepLogs.add(" -> [$dirStr 중단] 모든 후보가 이미 방문(visited) 됨")
                    expand = false
                }
            } else {
                stepLogs.add(" -> [$dirStr 중단] 탐색 ROI 내 ML Kit 결과 없음")
                expand = false
            }
        }
        return result
    }

    private fun buildSearchROI(
        currentBox: android.graphics.Rect, 
        recentBoxes: List<android.graphics.Rect>, 
        isLeft: Boolean, 
        fullW: Int, fullH: Int
    ): org.opencv.core.Rect {
        
        val historyToUse = recentBoxes.takeLast(3)
        val avgWidth = historyToUse.map { it.width() }.average()
        val avgHeight = historyToUse.map { it.height() }.average()

        var avgDistance = avgWidth * 0.8
        var expectedDy = 0.0

        if (historyToUse.size >= 2) {
            val dists = mutableListOf<Double>()
            for (i in 1 until historyToUse.size) {
                val dx = abs(historyToUse[i].exactCenterX() - historyToUse[i - 1].exactCenterX())
                dists.add(dx.toDouble())
            }
            val latestAvgDist = dists.average()
            avgDistance = min(latestAvgDist, avgWidth * 3.0) 

            val firstBox = historyToUse.first()
            val lastBox = historyToUse.last()
            
            val dxTotal = lastBox.exactCenterX() - firstBox.exactCenterX()
            val dyTotal = lastBox.exactCenterY() - firstBox.exactCenterY()
            
            if (abs(dxTotal) > 1.0) {
                val slope = dyTotal.toDouble() / dxTotal.toDouble()
                val stepDx = if (isLeft) -avgDistance else avgDistance
                expectedDy = stepDx * slope
            }
        }

        val searchWidth = (avgDistance + avgWidth * 1.2).toInt()
        val searchHeight = (avgHeight * 1.5).toInt()
        val overlapX = (avgWidth * 0.2).toInt()
        
        val expectedCenterY = currentBox.exactCenterY() + expectedDy
        
        val searchX = if (isLeft) {
            currentBox.left - searchWidth + overlapX
        } else {
            currentBox.right - overlapX
        }
        
        val searchY = (expectedCenterY - searchHeight / 2.0).toInt()
        
        return getSafeRect(searchX, searchY, searchWidth, searchHeight, fullW, fullH)
    }

    private fun chooseNearestCandidate(
        currentCenter: Point,
        candidates: List<android.graphics.Rect>,
        searchRect: org.opencv.core.Rect,
        isLeft: Boolean,
        visited: List<android.graphics.Rect>
    ): android.graphics.Rect? {
        var bestCandidate: android.graphics.Rect? = null
        var minDiff = Double.MAX_VALUE

        for (localBox in candidates) {
            val globalBox = android.graphics.Rect(
                searchRect.x + localBox.left,
                searchRect.y + localBox.top,
                searchRect.x + localBox.right,
                searchRect.y + localBox.bottom
            )
            
            if (isVisited(globalBox, visited)) continue

            val candidateCenterX = globalBox.exactCenterX().toDouble()
            val candidateCenterY = globalBox.exactCenterY().toDouble()
            
            // X축 방향 조건 검증 (좌/우측으로 진행하고 있는지)
            val isValidDirection = if (isLeft) {
                candidateCenterX < currentCenter.x
            } else {
                candidateCenterX > currentCenter.x
            }

            if (isValidDirection) {
                // 수정 반영: 1D 거리 계산 방식 -> 2D 유클리드 빗변 거리 계산으로 변경
                val dx = candidateCenterX - currentCenter.x
                val dy = candidateCenterY - currentCenter.y
                val dist = hypot(dx, dy)
                
                if (dist < minDiff) {
                    minDiff = dist
                    bestCandidate = globalBox
                }
            }
        }
        return bestCandidate
    }

    private fun isVisited(candidate: android.graphics.Rect, visited: List<android.graphics.Rect>): Boolean {
        val cx1 = candidate.exactCenterX()
        val cy1 = candidate.exactCenterY()
        
        // 수정 반영: 폭(Width) 대신 넓은 방향(Max)을 기준으로 삼아 좁은 숫자('1')도 안정적으로 필터링
        val maxDimension = max(candidate.width(), candidate.height())
        
        for (v in visited) {
            val cx2 = v.exactCenterX()
            val cy2 = v.exactCenterY()
            val dist = hypot((cx1 - cx2).toDouble(), (cy1 - cy2).toDouble())
            
            if (dist < maxDimension * 0.4) {
                return true
            }
        }
        return false
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
            
            it.pauseAndShowStep(
                "디버그 8/9: 최종 궤도 검증", debugBmp,
                "[8/9] 선형 궤도 검증 & KOR 마크 삭제",
                stepLogs
            )
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
            
            it.pauseAndShowStep(
                "디버그 9/9: 최종 렌더링", debugBmp,
                "[9/9] 디버깅 완료: 최종 와이어프레임 기하학 도출",
                listOf("-> (초록 테두리) 1.35배 대칭 팽창된 최종 번호판 영역", "[진단] 이 영역이 PerspectiveTransform 을 통해 최종 크롭될 영역입니다.")
            )
            debugMat.release(); debugBmp.recycle()
        }

        return finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
    }
}
