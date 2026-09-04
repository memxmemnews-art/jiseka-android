package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Rect as AndroidRect // 1. 명확한 분리를 위한 Alias 적용
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(
            stageName: String,
            bitmap: Bitmap,
            title: String,
            logs: List<String>
        )
    }

    data class SeedCropResult(
        val offsetX: Int,
        val offsetY: Int,
        val croppedBitmap: Bitmap,
        val roiRect: AndroidRect
    )

    private data class PlateCandidate(
        val pts: List<Point>,
        val score: Double,
        val debugLog: String
    )

    private data class ScoreResult(
        val score: Double,
        val log: String
    )

    private data class HoughLine(
        val p1: Point,
        val p2: Point,
        val length: Double,
        val angle: Double
    )

    // =========================================================
    // 디버그 이미지 렌더링 최적화 헬퍼
    // =========================================================
    private fun emitDebug(
        listener: DetectionDebugListener,
        mat: Mat,
        stageName: String,
        title: String,
        logs: List<String>
    ) {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        listener.pauseAndShowStep(stageName, bmp, title, logs)
        bmp.recycle() 
    }

    // ---------------------------------------------------------
    // ⭐️ 안전한 Rect (AndroidRect 반환) - 회원님 제안 로직 완벽 적용
    // ---------------------------------------------------------
    private fun getSafeRect(
        x: Int,
        y: Int,
        w: Int,
        h: Int,
        maxW: Int,
        maxH: Int
    ): AndroidRect {
        // 원본 자체가 비정상(0 이하)인 경우 방어
        if (maxW <= 0 || maxH <= 0) {
            return AndroidRect(0, 0, 1, 1)
        }

        // 1. 시작점은 실제 이미지 내부로 제한
        val left = x.coerceIn(0, maxW - 1)
        val top = y.coerceIn(0, maxH - 1)

        // 2. 남아 있는 실제 공간
        val availableW = maxW - left
        val availableH = maxH - top

        // 3. width / height는 반드시 1 이상, 남은 공간 이하 보장
        val safeW = w.coerceIn(1, availableW)
        val safeH = h.coerceIn(1, availableH)

        // 4. 정상적인 right, bottom 도출
        val right = left + safeW
        val bottom = top + safeH

        return AndroidRect(left, top, right, bottom)
    }

    // ---------------------------------------------------------
    // 1. 터치 주변 AI용 Crop
    // ---------------------------------------------------------
    fun prepareWideCrop(
        fullBitmap: Bitmap,
        touchX: Float,
        touchY: Float
    ): SeedCropResult {

        var cropW = (fullBitmap.width * 0.25f).toInt()
        var cropH = (fullBitmap.height * 0.15f).toInt()

        val minPhysicalWidth = 400
        val minPhysicalHeight = 200

        cropW = max(cropW, minPhysicalWidth).coerceAtMost(fullBitmap.width)
        cropH = max(cropH, minPhysicalHeight).coerceAtMost(fullBitmap.height)

        val safeRect = getSafeRect(
            (touchX - cropW / 2).toInt(),
            (touchY - cropH / 2).toInt(),
            cropW,
            cropH,
            fullBitmap.width,
            fullBitmap.height
        )

        // 안전한 Rect가 보장되므로 createBitmap에서 예외가 발생하지 않음
        val croppedBitmap = Bitmap.createBitmap(
            fullBitmap, 
            safeRect.left, 
            safeRect.top, 
            safeRect.width(), 
            safeRect.height()
        )

        return SeedCropResult(
            safeRect.left, 
            safeRect.top, 
            croppedBitmap, 
            safeRect
        )
    }

    // ---------------------------------------------------------
    // 2. AI Box → OpenCV 정밀화
    // ---------------------------------------------------------
    suspend fun processWithMLKitResult(
        fullBitmap: Bitmap,
        aiGlobalBox: AndroidRect, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {

        if (aiGlobalBox.width() <= 0 || aiGlobalBox.height() <= 0) {
            return null
        }

        val fullMat = Mat()
        val fullGray = Mat()

        try {
            Utils.bitmapToMat(fullBitmap, fullMat)
            Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

            val safeRoi = getSafeRect(
                aiGlobalBox.left,
                aiGlobalBox.top,
                aiGlobalBox.width(),
                aiGlobalBox.height(),
                fullMat.cols(),
                fullMat.rows()
            )

            if (safeRoi.width() <= 5 || safeRoi.height() <= 5) {
                return null
            }

            val cvSafeRoi = Rect(
                safeRoi.left,
                safeRoi.top,
                safeRoi.width(),
                safeRoi.height()
            )

            debugListener?.let {
                val debugMat = fullMat.clone()
                Imgproc.rectangle(debugMat, cvSafeRoi, Scalar(0.0, 255.0, 0.0, 255.0), 5)
                emitDebug(it, debugMat, "1. AI Box 탐색 영역", "ROI 설정 완료", listOf(
                    "X: ${safeRoi.left}, Y: ${safeRoi.top}",
                    "Width: ${safeRoi.width()}, Height: ${safeRoi.height()}",
                    "이 영역 안에서만 연산을 수행합니다."
                ))
                debugMat.release()
            }

            val roiGray = Mat()
            fullGray.submat(cvSafeRoi).copyTo(roiGray) 

            val localAiRect = AndroidRect(
                0, 
                0, 
                safeRoi.width(), 
                safeRoi.height()
            )
            
            val candidates = mutableListOf<PlateCandidate>()

            // A. Contour 후보
            val contourCandidates = extractContourCandidates(roiGray, debugListener)
            candidates.addAll(contourCandidates.mapNotNull { pts ->
                val score = evaluateCandidate(pts, localAiRect, safeRoi.width(), safeRoi.height())
                if (score.score >= MIN_CANDIDATE_SCORE) PlateCandidate(pts, score.score, score.log) else null
            })

            // B. Hough 후보
            val houghCandidates = extractHoughQuadCandidates(roiGray, debugListener)
            candidates.addAll(houghCandidates.mapNotNull { pts ->
                val score = evaluateCandidate(pts, localAiRect, safeRoi.width(), safeRoi.height())
                if (score.score >= MIN_CANDIDATE_SCORE) PlateCandidate(pts, score.score, score.log) else null
            })

            val bestCandidate = candidates.maxByOrNull { it.score }

            debugListener?.let { listener ->
                val debugMat = Mat()
                Imgproc.cvtColor(roiGray, debugMat, Imgproc.COLOR_GRAY2RGBA)

                candidates.forEach { c ->
                    val poly = MatOfPoint(*c.pts.toTypedArray())
                    Imgproc.polylines(debugMat, listOf(poly), true, Scalar(180.0, 180.0, 180.0, 255.0), 1)
                    poly.release()
                }

                bestCandidate?.let { best ->
                    val poly = MatOfPoint(*best.pts.toTypedArray())
                    Imgproc.polylines(debugMat, listOf(poly), true, Scalar(0.0, 255.0, 0.0, 255.0), 3)
                    poly.release()
                    for (p in best.pts) {
                        Imgproc.circle(debugMat, p, 4, Scalar(0.0, 200.0, 255.0, 255.0), -1)
                    }
                }

                emitDebug(listener, debugMat, "4. 전체 후보 평가", "점수 계산 및 최적 후보 선정", listOf(
                    "통과 기준점: $MIN_FINAL_SCORE",
                    "총 도출된 후보 수: ${candidates.size}",
                    "최고 점수: ${bestCandidate?.score?.let { s -> String.format("%.1f", s) } ?: "없음"}",
                    bestCandidate?.debugLog ?: "평가된 후보가 없습니다."
                ))
                debugMat.release()
            }

            if (bestCandidate == null || bestCandidate.score < MIN_FINAL_SCORE) {
                debugListener?.let {
                    val debugMat = fullMat.clone()
                    Imgproc.rectangle(debugMat, cvSafeRoi, Scalar(255.0, 0.0, 0.0, 255.0), 5) 
                    emitDebug(it, debugMat, "5. OpenCV 정밀화 실패", "번호판 4점 확정 실패", listOf(
                        "AI Box는 정상적으로 검출됨",
                        "하지만 신뢰할 수 있는 4점 후보를 찾지 못함",
                        "기준 점수 미달: ${bestCandidate?.score?.let { s -> String.format("%.1f", s) } ?: "0.0"} < $MIN_FINAL_SCORE"
                    ))
                    debugMat.release()
                }
                roiGray.release()
                return null
            }

            val finalLocalPts = sortCorners(bestCandidate.pts)
            val globalPts = finalLocalPts.map {
                ImmutablePoint((it.x + safeRoi.left).toFloat(), (it.y + safeRoi.top).toFloat())
            }

            debugListener?.let {
                val debugMat = fullMat.clone()
                Imgproc.rectangle(debugMat, cvSafeRoi, Scalar(0.0, 255.0, 255.0, 255.0), 3) 

                for (i in 0 until 4) {
                    val p1 = Point(globalPts[i].x.toDouble(), globalPts[i].y.toDouble())
                    val p2 = Point(globalPts[(i + 1) % 4].x.toDouble(), globalPts[(i + 1) % 4].y.toDouble())
                    Imgproc.line(debugMat, p1, p2, Scalar(0.0, 255.0, 0.0, 255.0), 5)
                    Imgproc.circle(debugMat, p1, 10, Scalar(255.0, 0.0, 255.0, 255.0), -1)
                }

                emitDebug(it, debugMat, "5. AI → OpenCV 정밀화 완료", "최종 번호판 영역 확정", listOf(
                    "AI Box 내부 정밀화 성공",
                    "최종 점수: ${String.format("%.1f", bestCandidate.score)}",
                    bestCandidate.debugLog
                ))
                debugMat.release()
            }

            roiGray.release()
            return globalPts

        } finally {
            fullGray.release()
            fullMat.release()
        }
    }

    private fun extractContourCandidates(gray: Mat, debugListener: DetectionDebugListener?): List<List<Point>> {
        val result = mutableListOf<List<Point>>()
        val edgeMats = mutableListOf<Mat>()

        run {
            val blurred = Mat()
            val edges = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(3.0, 3.0), 0.0)
            Imgproc.Canny(blurred, edges, 40.0, 120.0)
            
            debugListener?.let {
                emitDebug(it, edges, "2-1. Contour - 기본 Canny", "Canny 엣지 추출", listOf("Blur(3x3) -> Canny(40, 120)"))
            }
            
            edgeMats.add(edges)
            blurred.release()
        }

        run {
            val blurred = Mat()
            val edges = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(3.0, 3.0), 0.0)
            Imgproc.Canny(blurred, edges, 40.0, 120.0)

            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)
            
            debugListener?.let {
                emitDebug(it, edges, "2-2. Contour - Morph Close", "끊어진 선 연결", listOf("Morphology Close (2x2) 적용됨"))
            }

            kernel.release()
            edgeMats.add(edges)
            blurred.release()
        }

        for (edges in edgeMats) {
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            hierarchy.release()

            for (contour in contours) {
                if (contour.total() < 4) {
                    contour.release()
                    continue
                }

                val contour2f = MatOfPoint2f(*contour.toArray())
                val peri = Imgproc.arcLength(contour2f, true)
                val approx = MatOfPoint2f()

                Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true)
                val points = approx.toArray().toList()

                if (points.size == 4 && Imgproc.isContourConvex(MatOfPoint(*points.toTypedArray()))) {
                    result.add(sortCorners(points))
                }

                approx.release()
                contour2f.release()
                contour.release()
            }
            edges.release()
        }

        debugListener?.let { listener ->
            val debugMat = Mat()
            Imgproc.cvtColor(gray, debugMat, Imgproc.COLOR_GRAY2RGBA)
            result.forEach { pts ->
                val poly = MatOfPoint(*pts.toTypedArray())
                Imgproc.polylines(debugMat, listOf(poly), true, Scalar(0.0, 255.0, 0.0, 255.0), 2)
                poly.release()
            }
            emitDebug(listener, debugMat, "2-3. Contour - 다각형", "4점 다각형 필터링", listOf(
                "검출된 볼록 다각형(Convex) 수: ${result.size}"
            ))
            debugMat.release()
        }

        return result
    }

    private fun extractHoughQuadCandidates(gray: Mat, debugListener: DetectionDebugListener?): List<List<Point>> {
        val result = mutableListOf<List<Point>>()
        val edges = Mat()

        Imgproc.Canny(gray, edges, 40.0, 120.0)

        debugListener?.let {
            emitDebug(it, edges, "3-1. Hough - Canny", "직선 추출용 엣지", listOf("Canny(40, 120)"))
        }

        val lines = Mat()
        Imgproc.HoughLinesP(
            edges, lines, 1.0, Math.PI / 180.0, 25,
            min(gray.cols(), gray.rows()) * 0.18, 8.0
        )

        val houghLines = mutableListOf<HoughLine>()
        for (i in 0 until lines.rows()) {
            val v = lines.get(i, 0) ?: continue
            val p1 = Point(v[0], v[1])
            val p2 = Point(v[2], v[3])

            val dx = p2.x - p1.x
            val dy = p2.y - p1.y
            val length = hypot(dx, dy)

            if (length < 20.0) continue

            var angle = Math.toDegrees(atan2(dy, dx))
            if (angle < 0) angle += 180.0

            houghLines.add(HoughLine(p1, p2, length, angle))
        }

        lines.release()
        edges.release()

        debugListener?.let { listener ->
            val debugMat = Mat()
            Imgproc.cvtColor(gray, debugMat, Imgproc.COLOR_GRAY2RGBA)
            houghLines.forEach { hl ->
                Imgproc.line(debugMat, hl.p1, hl.p2, Scalar(255.0, 0.0, 0.0, 255.0), 1)
            }
            emitDebug(listener, debugMat, "3-2. Hough - 선 검출", "직선 성분 추출", listOf(
                "검출된 선분 수: ${houghLines.size}", 
                "상위 24개 추출 및 필터링 후 교점 탐색 진행"
            ))
            debugMat.release()
        }

        val selectedLines = houghLines.sortedByDescending { it.length }.take(24)
        val parallelPairs = mutableListOf<Pair<HoughLine, HoughLine>>()

        for (i in selectedLines.indices) {
            for (j in i + 1 until selectedLines.size) {
                val a = selectedLines[i]
                val b = selectedLines[j]
                val diff = angleDifference(a.angle, b.angle)
                if (diff <= 12.0) parallelPairs.add(Pair(a, b))
            }
        }

        for (i in parallelPairs.indices) {
            val pairA = parallelPairs[i]
            val angleA = pairA.first.angle
            for (j in i + 1 until parallelPairs.size) {
                val pairB = parallelPairs[j]
                val angleB = pairB.first.angle
                val perpendicularDiff = abs(angleDifference(angleA, angleB) - 90.0)

                if (perpendicularDiff > 20.0) continue

                val quad = buildQuadFromLines(pairA.first, pairA.second, pairB.first, pairB.second) ?: continue

                if (isReasonableQuad(quad, gray.cols(), gray.rows())) {
                    result.add(sortCorners(quad))
                }
            }
        }

        debugListener?.let { listener ->
            val debugMat = Mat()
            Imgproc.cvtColor(gray, debugMat, Imgproc.COLOR_GRAY2RGBA)
            result.forEach { pts ->
                val poly = MatOfPoint(*pts.toTypedArray())
                Imgproc.polylines(debugMat, listOf(poly), true, Scalar(255.0, 255.0, 0.0, 255.0), 2)
                poly.release()
            }
            emitDebug(listener, debugMat, "3-3. Hough - 조합 사각형", "교점 기반 4점 조합", listOf(
                "교점으로 생성된 사각형 수: ${result.size}"
            ))
            debugMat.release()
        }

        return result
    }

    private fun buildQuadFromLines(a1: HoughLine, a2: HoughLine, b1: HoughLine, b2: HoughLine): List<Point>? {
        val p1 = intersection(a1.p1, a1.p2, b1.p1, b1.p2)
        val p2 = intersection(a1.p1, a1.p2, b2.p1, b2.p2)
        val p3 = intersection(a2.p1, a2.p2, b2.p1, b2.p2)
        val p4 = intersection(a2.p1, a2.p2, b1.p1, b1.p2)

        if (p1 == null || p2 == null || p3 == null || p4 == null) return null
        return listOf(p1, p2, p3, p4)
    }

    private fun intersection(p1: Point, p2: Point, p3: Point, p4: Point): Point? {
        val x1 = p1.x; val y1 = p1.y
        val x2 = p2.x; val y2 = p2.y
        val x3 = p3.x; val y3 = p3.y
        val x4 = p4.x; val y4 = p4.y

        val denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (abs(denominator) < 1e-6) return null

        val px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        val py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return Point(px, py)
    }

    private fun isReasonableQuad(pts: List<Point>, width: Int, height: Int): Boolean {
        if (pts.size != 4) return false
        for (p in pts) {
            if (p.x < -width * 0.15 || p.x > width * 1.15 ||
                p.y < -height * 0.15 || p.y > height * 1.15) {
                return false
            }
        }
        val ordered = sortCorners(pts)
        val area = abs(polygonArea(ordered))
        if (area < width * height * 0.03) return false
        return true
    }

    private fun evaluateCandidate(pts: List<Point>, aiRect: AndroidRect, roiWidth: Int, roiHeight: Int): ScoreResult {
        if (pts.size != 4) {
            return ScoreResult(0.0, "4점 아님")
        }

        val p = sortCorners(pts)
        val tl = p[0]; val tr = p[1]; val br = p[2]; val bl = p[3]

        val top = hypot(tr.x - tl.x, tr.y - tl.y)
        val bottom = hypot(br.x - bl.x, br.y - bl.y)
        val left = hypot(bl.x - tl.x, bl.y - tl.y)
        val right = hypot(br.x - tr.x, br.y - tr.y)

        if (top <= 1 || bottom <= 1 || left <= 1 || right <= 1) {
            return ScoreResult(0.0, "변 길이 오류")
        }

        val widthAvg = (top + bottom) / 2.0
        val heightAvg = (left + right) / 2.0
        val aspectRatio = widthAvg / heightAvg

        val aspectScore = if (aspectRatio in 1.8..6.0) {
            100.0 - abs(aspectRatio - 3.0) * 12.0
        } else {
            max(0.0, 100.0 - abs(aspectRatio - 3.0) * 30.0)
        }

        val topAngle = lineAngle(tl, tr)
        val bottomAngle = lineAngle(bl, br)
        val horizontalParallel = angleDifference(topAngle, bottomAngle)
        val horizontalScore = max(0.0, 100.0 - horizontalParallel * 6.0)

        val leftAngle = lineAngle(tl, bl)
        val rightAngle = lineAngle(tr, br)
        val verticalParallel = angleDifference(leftAngle, rightAngle)
        val verticalScore = max(0.0, 100.0 - verticalParallel * 6.0)

        val parallelScore = (horizontalScore + verticalScore) / 2.0

        val candidateArea = abs(polygonArea(p))
        val aiWidth = aiRect.width().toDouble()
        val aiHeight = aiRect.height().toDouble()
        val aiArea = aiWidth * aiHeight

        if (aiArea <= 0) {
            return ScoreResult(0.0, "AI Box 면적 오류")
        }

        val areaRatio = candidateArea / aiArea
        val areaFitScore = when {
            areaRatio in 0.45..1.05 -> 100.0
            areaRatio < 0.45 -> max(0.0, areaRatio / 0.45 * 100.0)
            else -> max(0.0, 100.0 - (areaRatio - 1.05) * 150.0)
        }

        val centerX = p.map { it.x }.average()
        val centerY = p.map { it.y }.average()
        val aiCenterX = (aiRect.left + aiRect.right) / 2.0
        val aiCenterY = (aiRect.top + aiRect.bottom) / 2.0

        val centerDistance = hypot(centerX - aiCenterX, centerY - aiCenterY)
        val maxCenterDistance = hypot(aiWidth, aiHeight) / 2.0

        val centerScore = max(0.0, 100.0 - (centerDistance / maxCenterDistance.coerceAtLeast(1.0)) * 100.0)

        var overflow = 0.0
        for (point in p) {
            val dx = when {
                point.x < aiRect.left -> aiRect.left - point.x
                point.x > aiRect.right -> point.x - aiRect.right
                else -> 0.0
            }
            val dy = when {
                point.y < aiRect.top -> aiRect.top - point.y
                point.y > aiRect.bottom -> point.y - aiRect.bottom
                else -> 0.0
            }
            overflow += hypot(dx, dy)
        }

        val overflowRatio = overflow / (aiWidth + aiHeight)
        val overflowScore = max(0.0, 100.0 - overflowRatio * 200.0)

        val finalScore = (
            aspectScore * 0.20 +
            parallelScore * 0.25 +
            areaFitScore * 0.20 +
            centerScore * 0.20 +
            overflowScore * 0.15
        )

        val log = "점수=${String.format("%.1f", finalScore)} AR=${aspectScore.toInt()} 평행=${parallelScore.toInt()} 크기=${areaFitScore.toInt()} 중심=${centerScore.toInt()} Over=${overflowScore.toInt()}"
        return ScoreResult(finalScore, log)
    }

    private fun lineAngle(a: Point, b: Point): Double {
        var angle = Math.toDegrees(atan2(b.y - a.y, b.x - a.x))
        if (angle < 0) angle += 180.0
        return angle
    }

    private fun angleDifference(a: Double, b: Double): Double {
        var diff = abs(a - b)
        while (diff > 180.0) diff -= 180.0
        return min(diff, 180.0 - diff)
    }

    private fun polygonArea(pts: List<Point>): Double {
        var area = 0.0
        for (i in pts.indices) {
            val j = (i + 1) % pts.size
            area += pts[i].x * pts[j].y - pts[j].x * pts[i].y
        }
        return area / 2.0
    }

    private fun sortCorners(pts: List<Point>): List<Point> {
        if (pts.size != 4) return pts
        val centerX = pts.map { it.x }.average()
        val centerY = pts.map { it.y }.average()
        val sorted = pts.sortedBy { atan2(it.y - centerY, it.x - centerX) }
        val startIndex = sorted.indices.minByOrNull { i -> sorted[i].x + sorted[i].y } ?: 0
        return List(4) { index -> sorted[(startIndex + index) % 4] }
    }

    private const val MIN_CANDIDATE_SCORE = 35.0
    private const val MIN_FINAL_SCORE = 65.0
}
