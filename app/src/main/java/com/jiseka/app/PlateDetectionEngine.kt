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

    private fun checkDensity(fullGray: Mat, roi: Rect): Double {
        val roiMat = fullGray.submat(roi)
        val thresh = Mat()
        Imgproc.adaptiveThreshold(roiMat, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 19, 12.0)
        val nonZero = Core.countNonZero(thresh)
        val density = nonZero.toDouble() / max(1, roi.width * roi.height).toDouble()
        thresh.release()
        roiMat.release()
        return density
    }

    fun prepareWideCrop(fullBitmap: Bitmap, touchX: Float, touchY: Float): SeedCropResult {
        val cropW = (fullBitmap.width * 0.10f).toInt()
        val cropH = (fullBitmap.height * 0.05f).toInt()

        val safeRect = getSafeRect(
            (touchX - cropW / 2).toInt(),
            (touchY - cropH / 2).toInt(),
            cropW,
            cropH,
            fullBitmap.width, fullBitmap.height
        )

        val croppedBitmap = Bitmap.createBitmap(fullBitmap, safeRect.x, safeRect.y, safeRect.width, safeRect.height)
        return SeedCropResult(safeRect.x, safeRect.y, croppedBitmap, safeRect)
    }

    private fun splitAndTightenWithOpenCV(
        fullGray: Mat, mlKitGlobalRect: Rect, fullCols: Int, fullRows: Int, seedWidth: Double?
    ): List<CharData> {
        val resultChars = mutableListOf<CharData>()

        val padY = (mlKitGlobalRect.height * 0.10).toInt()
        val padX = (mlKitGlobalRect.width * 0.05).toInt() 
        
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

        val nonZeroPixels = Core.countNonZero(thresh)
        val totalPixels = thresh.rows() * thresh.cols()
        val density = nonZeroPixels.toDouble() / max(1, totalPixels).toDouble()

        if (density < 0.08 || density > 0.75) {
            roiGray.release(); blurred.release(); thresh.release()
            return emptyList()
        }

        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        tempClose.release()

        val projection = IntArray(thresh.cols())
        for (col in 0 until thresh.cols()) {
            var colSum = 0
            for (row in 0 until thresh.rows()) {
                if (thresh.get(row, col)[0] > 128) {
                    colSum++
                }
            }
            projection[col] = colSum
        }

        val maxProj = projection.maxOrNull() ?: 1
        val noiseThreshold = maxProj * 0.10 

        val segments = mutableListOf<Pair<Int, Int>>() 
        var isText = false
        var startX = 0

        for (col in 0 until projection.size) {
            if (!isText && projection[col] > noiseThreshold) {
                isText = true
                startX = col
            } else if (isText && projection[col] <= noiseThreshold) {
                isText = false
                val segmentWidth = col - startX
                if (segmentWidth > (seedWidth ?: 10.0) * 0.15) { 
                    segments.add(Pair(startX, col))
                }
            }
        }
        
        if (isText) {
             if ((projection.size - startX) > (seedWidth ?: 10.0) * 0.15) {
                 segments.add(Pair(startX, projection.size - 1))
             }
        }

        for (segment in segments) {
            val segX = segment.first
            val segEndX = segment.second
            val segWidth = segEndX - segX
            
            val segRect = Rect(segX, 0, segWidth, thresh.rows())
            val segMat = Mat()
            thresh.submat(segRect).copyTo(segMat)

            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(segMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            if (contours.isNotEmpty()) {
                var minX = Int.MAX_VALUE; var minY = Int.MAX_VALUE
                var maxX = Int.MIN_VALUE; var maxY = Int.MIN_VALUE

                for (contour in contours) {
                    val rect = Imgproc.boundingRect(contour)
                    if (rect.area() > 10) { 
                        minX = min(minX, rect.x)
                        minY = min(minY, rect.y)
                        maxX = max(maxX, rect.x + rect.width)
                        maxY = max(maxY, rect.y + rect.height)
                    }
                }

                if (maxX > minX && maxY > minY) {
                    val finalTightRect = Rect(
                        searchRect.x + segX + minX, 
                        searchRect.y + minY, 
                        maxX - minX, 
                        maxY - minY
                    )
                    
                    if (finalTightRect.height > searchRect.height * 0.3) {
                        val globalCenter = Point(finalTightRect.x + finalTightRect.width / 2.0, finalTightRect.y + finalTightRect.height / 2.0)
                        resultChars.add(CharData(globalCenter, finalTightRect.width.toDouble(), finalTightRect.height.toDouble(), finalTightRect))
                    }
                }
            }
            segMat.release()
            hierarchy.release()
            contours.forEach { it.release() }
        }

        roiGray.release(); blurred.release(); thresh.release()
        return resultChars
    }

    suspend fun processWithMLKitResult(
        fullBitmap: Bitmap, 
        lineGlobalBox: android.graphics.Rect, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        val fullMat = Mat(); val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)
        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

        val seedWidthEstimate = lineGlobalBox.width().toDouble() / 7.0

        val cvRect = org.opencv.core.Rect(lineGlobalBox.left, lineGlobalBox.top, lineGlobalBox.width(), lineGlobalBox.height())
        val uniqueChars = splitAndTightenWithOpenCV(fullGray, cvRect, fullMat.cols(), fullMat.rows(), seedWidthEstimate).toMutableList()

        uniqueChars.sortBy { it.rect.x }

        if (uniqueChars.size > 2) {
            val heightsDesc = uniqueChars.map { it.height }.sortedDescending()
            val trueHeight = if (heightsDesc.size >= 3) heightsDesc[2] else heightsDesc.first()
            val purgeHeightLimit = trueHeight * 0.50
            
            val purgeIterator = uniqueChars.iterator()
            while (purgeIterator.hasNext()) {
                if (purgeIterator.next().height < purgeHeightLimit) {
                    purgeIterator.remove()
                }
            }
        }

        var resultPoints: List<ImmutablePoint>? = null
        if (uniqueChars.isNotEmpty()) { 
            resultPoints = buildWireframe(uniqueChars, fullMat, fullGray, debugListener, screenRatio)
        }

        fullMat.release(); fullGray.release()
        return resultPoints
    }

    // 내부 후보군 점수 클래스 정의
    private class CandidateScore(val pts: List<Point>, val score: Double, val log: String)

    private fun buildWireframe(
        collectedChars: List<CharData>, fullMat: Mat, fullGray: Mat,
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
                stepLogs.add(" -> [검증] 좌측 KOR 파랑 마크 식별 및 완전 제거 완료")
            }
        }

        if (validChars.size >= 2) {
            val pointsMatTemp = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
            val lineTemp = Mat()
            Imgproc.fitLine(pointsMatTemp, lineTemp, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
            
            val vxTemp = lineTemp.get(0, 0)[0]; val vyTemp = lineTemp.get(1, 0)[0]
            val x0Temp = lineTemp.get(2, 0)[0]; val y0Temp = lineTemp.get(3, 0)[0]
            pointsMatTemp.release(); lineTemp.release()

            val mag = hypot(vxTemp, vyTemp)
            var normVx = vxTemp / mag
            var normVy = vyTemp / mag
            if (normVx < 0) { normVx = -normVx; normVy = -normVy } 

            val avgW = validChars.map { it.width }.average()
            val avgH = validChars.map { it.height }.average()
            val stepX = normVx * avgW * 1.15 
            val stepY = normVy * avgW * 1.15

            stepLogs.add("[추가 탐색] 산출된 궤도(${String.format("%.2f", normVy/normVx)})를 따라 좌우 레이더 가동")

            var currentPtL = validChars.first().center
            for(i in 0..2) {
                currentPtL = Point(currentPtL.x - stepX, currentPtL.y - stepY)
                val roi = getSafeRect((currentPtL.x - avgW/2).toInt(), (currentPtL.y - avgH/2).toInt(), avgW.toInt(), avgH.toInt(), fullGray.cols(), fullGray.rows())
                if (roi.width < avgW * 0.5) break 
                
                if (checkDensity(fullGray, roi) in 0.08..0.60) {
                    validChars.add(0, CharData(currentPtL, avgW, avgH, roi))
                    stepLogs.add(" -> [성공] 좌측 연장선에서 숨겨진 문자 1개 추가 적출")
                } else break 
            }

            var currentPtR = validChars.last().center
            for(i in 0..3) {
                currentPtR = Point(currentPtR.x + stepX, currentPtR.y + stepY)
                val roi = getSafeRect((currentPtR.x - avgW/2).toInt(), (currentPtR.y - avgH/2).toInt(), avgW.toInt(), avgH.toInt(), fullGray.cols(), fullGray.rows())
                if (roi.width < avgW * 0.5) break
                
                if (checkDensity(fullGray, roi) in 0.08..0.60) {
                    validChars.add(CharData(currentPtR, avgW, avgH, roi))
                    stepLogs.add(" -> [성공] 우측 연장선에서 숨겨진 문자 1개 추가 적출")
                } else break
            }

            val A = vyTemp; val B = -vxTemp; val C = vxTemp * y0Temp - vyTemp * x0Temp
            val denominator = hypot(A, B)
            val fitLineLimit = avgH * 0.20

            val iterator = validChars.iterator()
            while (iterator.hasNext()) {
                val charData = iterator.next()
                val dist = abs(A * charData.center.x + B * charData.center.y + C) / denominator
                
                if (dist > fitLineLimit) {
                    stepLogs.add(" -> [검증] 선형 궤도 이탈 노이즈 삭제")
                    iterator.remove()
                }
            }
        }

        if (validChars.size < 4) {
            stepLogs.add(" -> [FAIL] 추가 탐색 후에도 유효 문자가 4개 미만입니다.")
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

        stepLogs.add("[성공] 대칭 팽창(Scale: 1.35) 가상 테두리 생성 완료")

        // --------------------------------------------------------------------------------------
        // ⭐️ 새로운 로직: 점수(가중치) 기반 후보군 비교 알고리즘
        // --------------------------------------------------------------------------------------
        
        // 1. 기준점(Baseline) 특징 추출
        val baseTl = finalPts[0]; val baseTr = finalPts[1]
        val baseBr = finalPts[2]; val baseBl = finalPts[3]
        
        val baseW = (hypot(baseTr.x - baseTl.x, baseTr.y - baseTl.y) + hypot(baseBr.x - baseBl.x, baseBr.y - baseBl.y)) / 2.0
        val baseH = (hypot(baseBl.x - baseTl.x, baseBl.y - baseTl.y) + hypot(baseBr.x - baseTr.x, baseBr.y - baseTr.y)) / 2.0
        val baseAR = baseW / baseH
        val baseCx = midX
        val baseCy = midY
        val baseAngle = Math.toDegrees(Math.atan2(baseTr.y - baseTl.y, baseTr.x - baseTl.x))

        // 2. ROI 엣지 탐색
        val roiMinX = finalPts.minOf { it.x }.toInt()
        val roiMinY = finalPts.minOf { it.y }.toInt()
        val roiMaxX = finalPts.maxOf { it.x }.toInt()
        val roiMaxY = finalPts.maxOf { it.y }.toInt()

        val pad = 10
        val plateRoiRect = getSafeRect(roiMinX - pad, roiMinY - pad, roiMaxX - roiMinX + pad * 2, roiMaxY - roiMinY + pad * 2, fullGray.cols(), fullGray.rows())
        val roiArea = plateRoiRect.width * plateRoiRect.height

        val roiGray = Mat()
        fullGray.submat(plateRoiRect).copyTo(roiGray) 

        val edges = Mat()
        Imgproc.GaussianBlur(roiGray, edges, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(edges, edges, 50.0, 150.0)
        
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        val plateContours = ArrayList<MatOfPoint>()
        val hierarchy2 = Mat()
        
        // RETR_LIST 모드로 안쪽/바깥쪽 가리지 않고 모든 엣지를 긁어모읍니다.
        Imgproc.findContours(edges, plateContours, hierarchy2, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        val candidates = mutableListOf<CandidateScore>()

        for (contour in plateContours) {
            val peri = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, 0.03 * peri, true)

            // 노이즈(조각) 제거를 위해 최소한 ROI 면적의 10% 이상인 볼록 사각형만 취급
            if (approx.toArray().size == 4 && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                val area = Imgproc.contourArea(approx)
                if (area < roiArea * 0.10) { approx.release(); continue }
                
                // 글로벌 좌표로 복원 및 정렬 (TL, TR, BR, BL)
                val pts = approx.toArray().map { Point(it.x + plateRoiRect.x, it.y + plateRoiRect.y) }
                val sortedBySum = pts.sortedBy { it.x + it.y }
                val tl = sortedBySum.first()
                val br = sortedBySum.last()
                val remaining = pts.filter { it != tl && it != br }
                val tr = if (remaining[0].x > remaining[1].x) remaining[0] else remaining[1]
                val bl = if (remaining[0].x < remaining[1].x) remaining[0] else remaining[1]
                
                val candPts = listOf(tl, tr, br, bl)

                // 🚨 절대 방어막 (Hard Filter): 문자 포함률 검사
                val contourMat = MatOfPoint2f(*candPts.toTypedArray())
                var includedChars = 0
                for (char in validChars) {
                    if (Imgproc.pointPolygonTest(contourMat, char.center, false) >= 0) {
                        includedChars++
                    }
                }
                contourMat.release()
                
                val inclusionRate = includedChars.toDouble() / validChars.size
                if (inclusionRate < 0.5) { approx.release(); continue } // 문자를 50% 이상 못 품으면 즉시 버림

                // 3. 후보군 특징 산출 및 채점 (Scoring)
                val candW = (hypot(tr.x - tl.x, tr.y - tl.y) + hypot(br.x - bl.x, br.y - bl.y)) / 2.0
                val candH = (hypot(bl.x - tl.x, bl.y - tl.y) + hypot(br.x - tr.x, br.y - tr.y)) / 2.0
                val candAR = candW / candH
                val candCx = (tl.x + tr.x + br.x + bl.x) / 4.0
                val candCy = (tl.y + tr.y + br.y + bl.y) / 4.0
                val candAngle = Math.toDegrees(Math.atan2(tr.y - tl.y, tr.x - tl.x))

                // 가중치 1: 꼭짓점 일치도 (IoU 대체, 45%)
                val avgCornerDist = (hypot(tl.x - baseTl.x, tl.y - baseTl.y) + 
                                     hypot(tr.x - baseTr.x, tr.y - baseTr.y) + 
                                     hypot(br.x - baseBr.x, br.y - baseBr.y) + 
                                     hypot(bl.x - baseBl.x, bl.y - baseBl.y)) / 4.0
                val cornerScore = max(0.0, 1.0 - (avgCornerDist / baseH)) * 100.0

                // 가중치 2: 중심점 일치도 (25%)
                val centerDist = hypot(candCx - baseCx, candCy - baseCy)
                val centerScore = max(0.0, 1.0 - (centerDist / baseH)) * 100.0

                // 가중치 3: 회전각 일치도 (15%)
                var angleDiff = abs(candAngle - baseAngle)
                if (angleDiff > 180) angleDiff = 360.0 - angleDiff
                val angleScore = max(0.0, 1.0 - (angleDiff / 15.0)) * 100.0 // 15도 이상 차이나면 0점

                // 가중치 4: 가로세로 비율(AR) 일치도 (15%)
                val arDiff = abs(candAR - baseAR)
                val arScore = max(0.0, 1.0 - (arDiff / baseAR)) * 100.0

                val totalScore = (0.45 * cornerScore) + (0.25 * centerScore) + (0.15 * angleScore) + (0.15 * arScore)
                
                val logStr = "총점:${String.format("%.1f", totalScore)} (코너:${cornerScore.toInt()}, 중심:${centerScore.toInt()}, 각도:${angleScore.toInt()})"
                candidates.add(CandidateScore(candPts, totalScore, logStr))
                
                approx.release()
            }
        }
        
        // 4. 최종 판단
        var refinedPts = finalPts 
        if (candidates.isNotEmpty()) {
            candidates.sortByDescending { it.score }
            val best = candidates.first()
            refinedPts = best.pts
            stepLogs.add(" -> [정밀 적출 성공] 가장 높은 점수(${String.format("%.1f", best.score)}점)의 물리 테두리 채택!")
            candidates.forEachIndexed { index, cand -> 
                stepLogs.add("     - 후보${index+1}: ${cand.log}") 
            }
        } else {
            stepLogs.add(" -> [정밀 적출 실패] 조건을 만족하는 물리 테두리가 없어 1.35배 영역 유지.")
        }

        roiGray.release(); edges.release(); kernel.release(); hierarchy2.release()
        plateContours.forEach { it.release() }
        // --------------------------------------------------------------------------------------


        debugListener?.let {
            val debugMat = fullMat.clone()
            val colors = arrayOf(Scalar(255.0, 0.0, 0.0, 255.0), Scalar(0.0, 255.0, 0.0, 255.0), 
                                 Scalar(0.0, 0.0, 255.0, 255.0), Scalar(255.0, 255.0, 0.0, 255.0))
            val labels = arrayOf("TL", "TR", "BR", "BL")

            for (i in 0..3) {
                Imgproc.line(debugMat, refinedPts[i], refinedPts[(i + 1) % 4], Scalar(0.0, 255.0, 0.0, 255.0), 5)
                Imgproc.circle(debugMat, refinedPts[i], 15, colors[i], -1)
                Imgproc.putText(debugMat, labels[i], Point(refinedPts[i].x - 20, refinedPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 1.8, colors[i], 4)
            }
            Imgproc.circle(debugMat, Point(midX, midY), 8, Scalar(0.0, 255.0, 255.0, 255.0), -1)

            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            it.pauseAndShowStep(
                "디버그 9/9: 점수 기반 테두리 채택", debugBmp,
                "[9/9] 디버깅 완료: 종합 점수 기반 최적 테두리 적출",
                stepLogs
            )
            debugMat.release(); debugBmp.recycle()
        }

        return refinedPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
    }
}
