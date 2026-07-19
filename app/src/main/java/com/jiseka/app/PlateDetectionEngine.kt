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

    // ⭐️ 수정: 디버그 리스너 추가하여 실패 원인 시각화
    private fun splitAndTightenWithOpenCV(
        fullGray: Mat, mlKitGlobalRect: Rect, fullCols: Int, fullRows: Int, seedWidth: Double?, debugListener: DetectionDebugListener?
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

        // 🚨 실패 관문 1: 픽셀 밀도 검사 (노이즈, 그릴, 아스팔트 등)
        if (density < 0.08 || density > 0.75) {
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(thresh, debugBmp)
                it.pauseAndShowStep(
                    "디버그 3단계: [FAIL] 밀도 검사 실패", debugBmp,
                    "[FAIL] 텍스트 밀도 미달 또는 초과",
                    listOf("-> 기준 밀도: 0.08 ~ 0.75", "-> 현재 밀도: ${String.format("%.3f", density)}", "-> 원인: 노이즈가 너무 많거나 글자가 아닙니다.")
                )
                debugBmp.recycle()
            }
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
        
        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.rectangle(debugMat, cvRect, Scalar(0.0, 255.0, 0.0, 255.0), 4)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            it.pauseAndShowStep("디버그 2단계: 텍스트 글로벌 ROI", debugBmp, "[진단] 크롭 영역 기준 글로벌 좌표 복원", listOf("-> (초록) 이 영역을 투영 분할기에 넘깁니다."))
            debugMat.release(); debugBmp.recycle()
        }

        val uniqueChars = splitAndTightenWithOpenCV(fullGray, cvRect, fullMat.cols(), fullMat.rows(), seedWidthEstimate, debugListener).toMutableList()

        if (uniqueChars.isEmpty()) {
            fullMat.release(); fullGray.release()
            return null // 밀도 검사에서 이미 실패 화면을 띄웠음
        }

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

        // 🚨 실패 관문 2: 노이즈 청소 후 글자가 2개 미만일 때 (기울기 선을 그을 수 없음)
        if (uniqueChars.size < 2) {
            debugListener?.let {
                val debugMat = fullMat.clone()
                uniqueChars.forEach { char -> Imgproc.rectangle(debugMat, char.rect, Scalar(0.0, 0.0, 255.0, 255.0), 4) }
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                it.pauseAndShowStep(
                    "디버그 4단계: [FAIL] 유효 문자 부족", debugBmp,
                    "[FAIL] 노이즈 청소 후 문자가 다 지워짐",
                    listOf("-> 남은 문자: ${uniqueChars.size}개", "-> 원인: 크기가 들쭉날쭉한 쓰레기 엣지였을 확률이 높습니다.")
                )
                debugMat.release(); debugBmp.recycle()
            }
            fullMat.release(); fullGray.release()
            return null
        }

        var resultPoints: List<ImmutablePoint>? = null
        if (uniqueChars.isNotEmpty()) { 
            resultPoints = buildWireframe(uniqueChars, fullMat, fullGray, debugListener, screenRatio)
        }

        fullMat.release(); fullGray.release()
        return resultPoints
    }

    private class CandidateScore(val pts: List<Point>, val score: Double, val log: String)

    private fun buildWireframe(
        collectedChars: List<CharData>, fullMat: Mat, fullGray: Mat,
        debugListener: DetectionDebugListener?, screenRatio: Float
    ): List<ImmutablePoint>? {
        
        val stepLogs = mutableListOf<String>()

        var validChars = collectedChars.toMutableList()

        if (validChars.isNotEmpty()) {
            val firstChar = validChars.first() 
            val roiMat = fullMat.submat(firstChar.rect)
            val meanColor = Core.mean(roiMat)
            roiMat.release() 
            
            val r = meanColor.`val`[0].toInt(); val g = meanColor.`val`[1].toInt(); val b = meanColor.`val`[2].toInt()
            if (b > r + 25 && b > g + 15) {
                validChars.removeAt(0) 
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

            stepLogs.add("[탐색] 산출된 기울기를 따라 좌우 숨겨진 문자 적출 진행")

            var currentPtL = validChars.first().center
            for(i in 0..2) {
                currentPtL = Point(currentPtL.x - stepX, currentPtL.y - stepY)
                val roi = getSafeRect((currentPtL.x - avgW/2).toInt(), (currentPtL.y - avgH/2).toInt(), avgW.toInt(), avgH.toInt(), fullGray.cols(), fullGray.rows())
                if (roi.width < avgW * 0.5) break 
                
                if (checkDensity(fullGray, roi) in 0.08..0.60) {
                    validChars.add(0, CharData(currentPtL, avgW, avgH, roi))
                    stepLogs.add(" -> 좌측 숨겨진 문자 1개 추가 적출")
                } else break 
            }

            var currentPtR = validChars.last().center
            for(i in 0..3) {
                currentPtR = Point(currentPtR.x + stepX, currentPtR.y + stepY)
                val roi = getSafeRect((currentPtR.x - avgW/2).toInt(), (currentPtR.y - avgH/2).toInt(), avgW.toInt(), avgH.toInt(), fullGray.cols(), fullGray.rows())
                if (roi.width < avgW * 0.5) break
                
                if (checkDensity(fullGray, roi) in 0.08..0.60) {
                    validChars.add(CharData(currentPtR, avgW, avgH, roi))
                    stepLogs.add(" -> 우측 숨겨진 문자 1개 추가 적출")
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
                    iterator.remove()
                }
            }
        }

        // 🚨 실패 관문 3: 레이더 탐색 후에도 4개가 안될 때 
        val isFailed = validChars.size < 4
        if (isFailed) {
            stepLogs.add(" -> [FAIL] 유효 문자가 ${validChars.size}개뿐이므로 기하학 조립을 포기합니다.")
        } else {
            stepLogs.add("[진단] 최종 ${validChars.size}개 문자로 와이어프레임 렌더링을 진행합니다.")
        }

        // ⭐️ 실패하든 성공하든 무조건 화면을 띄워서 결과를 보여줌
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
            
            val title = if (isFailed) "[FAIL] 번호판 문자 개수 미달" else "[진행] 기울기 탐색 및 궤도 검증"
            it.pauseAndShowStep("디버그 5단계: 궤도 검증", debugBmp, title, stepLogs)
            debugMat.release(); debugBmp.recycle()
        }

        if (isFailed) return null

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

        val scoreLogs = mutableListOf<String>()
        
        val baseTl = finalPts[0]; val baseTr = finalPts[1]
        val baseBr = finalPts[2]; val baseBl = finalPts[3]
        
        val baseW = (hypot(baseTr.x - baseTl.x, baseTr.y - baseTl.y) + hypot(baseBr.x - baseBl.x, baseBr.y - baseBl.y)) / 2.0
        val baseH = (hypot(baseBl.x - baseTl.x, baseBl.y - baseTl.y) + hypot(baseBr.x - baseTr.x, baseBr.y - baseTr.y)) / 2.0
        val baseAR = baseW / baseH
        val baseCx = midX
        val baseCy = midY
        val baseAngle = Math.toDegrees(Math.atan2(baseTr.y - baseTl.y, baseTr.x - baseTl.x))

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
        
        Imgproc.findContours(edges, plateContours, hierarchy2, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        val candidates = mutableListOf<CandidateScore>()

        for (contour in plateContours) {
            val peri = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, 0.03 * peri, true)

            if (approx.toArray().size == 4 && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                val area = Imgproc.contourArea(approx)
                if (area < roiArea * 0.10) { approx.release(); continue }
                
                val pts = approx.toArray().map { Point(it.x + plateRoiRect.x, it.y + plateRoiRect.y) }
                val sortedBySum = pts.sortedBy { it.x + it.y }
                val tl = sortedBySum.first()
                val br = sortedBySum.last()
                val remaining = pts.filter { it != tl && it != br }
                val tr = if (remaining[0].x > remaining[1].x) remaining[0] else remaining[1]
                val bl = if (remaining[0].x < remaining[1].x) remaining[0] else remaining[1]
                
                val candPts = listOf(tl, tr, br, bl)

                val contourMat = MatOfPoint2f(*candPts.toTypedArray())
                var includedChars = 0
                for (char in validChars) {
                    if (Imgproc.pointPolygonTest(contourMat, char.center, false) >= 0) {
                        includedChars++
                    }
                }
                contourMat.release()
                
                val inclusionRate = includedChars.toDouble() / validChars.size
                if (inclusionRate < 0.5) { approx.release(); continue }

                val candW = (hypot(tr.x - tl.x, tr.y - tl.y) + hypot(br.x - bl.x, br.y - bl.y)) / 2.0
                val candH = (hypot(bl.x - tl.x, bl.y - tl.y) + hypot(br.x - tr.x, br.y - tr.y)) / 2.0
                val candAR = candW / candH
                val candCx = (tl.x + tr.x + br.x + bl.x) / 4.0
                val candCy = (tl.y + tr.y + br.y + bl.y) / 4.0
                val candAngle = Math.toDegrees(Math.atan2(tr.y - tl.y, tr.x - tl.x))

                val avgCornerDist = (hypot(tl.x - baseTl.x, tl.y - baseTl.y) + 
                                     hypot(tr.x - baseTr.x, tr.y - baseTr.y) + 
                                     hypot(br.x - baseBr.x, br.y - baseBr.y) + 
                                     hypot(bl.x - baseBl.x, bl.y - baseBl.y)) / 4.0
                val cornerScore = max(0.0, 1.0 - (avgCornerDist / baseH)) * 100.0

                val centerDist = hypot(candCx - baseCx, candCy - baseCy)
                val centerScore = max(0.0, 1.0 - (centerDist / baseH)) * 100.0

                var angleDiff = abs(candAngle - baseAngle)
                if (angleDiff > 180) angleDiff = 360.0 - angleDiff
                val angleScore = max(0.0, 1.0 - (angleDiff / 15.0)) * 100.0 

                val arDiff = abs(candAR - baseAR)
                val arScore = max(0.0, 1.0 - (arDiff / baseAR)) * 100.0

                val totalScore = (0.45 * cornerScore) + (0.25 * centerScore) + (0.15 * angleScore) + (0.15 * arScore)
                
                val logStr = "총점:${String.format("%.1f", totalScore)} (코너:${cornerScore.toInt()}, 중심:${centerScore.toInt()}, 각도:${angleScore.toInt()})"
                candidates.add(CandidateScore(candPts, totalScore, logStr))
                
                approx.release()
            }
        }
        
        var refinedPts = finalPts 
        if (candidates.isNotEmpty()) {
            candidates.sortByDescending { it.score }
            val best = candidates.first()
            refinedPts = best.pts
            scoreLogs.add(" -> [정밀 적출 성공] 가장 높은 점수(${String.format("%.1f", best.score)}점)의 테두리 채택")
            candidates.forEachIndexed { index, cand -> 
                scoreLogs.add("     - 후보${index+1}: ${cand.log}") 
            }
        } else {
            scoreLogs.add(" -> [정밀 적출 포기] 조건을 만족하는 테두리가 없어 1.35배 영역 유지.")
        }

        roiGray.release(); edges.release(); kernel.release(); hierarchy2.release()
        plateContours.forEach { it.release() }

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
                "디버그 6단계: 점수 기반 테두리 채택", debugBmp,
                "[최종] 종합 점수 기반 최적 테두리 적출",
                scoreLogs
            )
            debugMat.release(); debugBmp.recycle()
        }

        return refinedPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
    }
}
