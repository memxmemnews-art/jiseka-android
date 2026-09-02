package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

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
        val roiRect: Rect
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

    // ---------------------------------------------------------
    // 안전한 Rect
    // ---------------------------------------------------------

    private fun getSafeRect(
        x: Int,
        y: Int,
        w: Int,
        h: Int,
        maxW: Int,
        maxH: Int
    ): Rect {

        if (maxW <= 0 || maxH <= 0) {
            return Rect(0, 0, 0, 0)
        }

        val safeX = x.coerceIn(0, maxW - 1)
        val safeY = y.coerceIn(0, maxH - 1)

        val safeW = w.coerceAtMost(maxW - safeX)
        val safeH = h.coerceAtMost(maxH - safeY)

        return Rect(
            safeX,
            safeY,
            safeW,
            safeH
        )
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

        cropW = max(cropW, minPhysicalWidth)
            .coerceAtMost(fullBitmap.width)

        cropH = max(cropH, minPhysicalHeight)
            .coerceAtMost(fullBitmap.height)

        val safeRect = getSafeRect(
            (touchX - cropW / 2).toInt(),
            (touchY - cropH / 2).toInt(),
            cropW,
            cropH,
            fullBitmap.width,
            fullBitmap.height
        )

        val croppedBitmap = Bitmap.createBitmap(
            fullBitmap,
            safeRect.x,
            safeRect.y,
            safeRect.width,
            safeRect.height
        )

        return SeedCropResult(
            safeRect.x,
            safeRect.y,
            croppedBitmap,
            safeRect
        )
    }

    // ---------------------------------------------------------
    // 2. AI Box → OpenCV 정밀화
    // ---------------------------------------------------------

    suspend fun processWithMLKitResult(
        fullBitmap: Bitmap,
        aiGlobalBox: Rect,
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {

        if (aiGlobalBox.width() <= 0 || aiGlobalBox.height() <= 0) {
            return null
        }

        val fullMat = Mat()
        val fullGray = Mat()

        try {

            Utils.bitmapToMat(fullBitmap, fullMat)

            Imgproc.cvtColor(
                fullMat,
                fullGray,
                Imgproc.COLOR_RGBA2GRAY
            )

            // =================================================
            // 핵심 변경
            //
            // AI Box를 확장하지 않는다.
            // AI가 정한 영역 자체를 OpenCV 검색 영역으로 사용한다.
            // =================================================

            val safeRoi = getSafeRect(
                aiGlobalBox.left,
                aiGlobalBox.top,
                aiGlobalBox.width(),
                aiGlobalBox.height(),
                fullMat.cols(),
                fullMat.rows()
            )

            if (safeRoi.width <= 5 || safeRoi.height <= 5) {
                return null
            }

            val roiGray = Mat()

            fullGray
                .submat(safeRoi)
                .copyTo(roiGray)

            // =================================================
            // AI Box는 그대로 유지
            // =================================================

            val localAiRect = Rect(
                0,
                0,
                safeRoi.width,
                safeRoi.height
            )

            // =================================================
            // 후보 생성
            //
            // A. 원본 Canny
            // B. 약한 Close Canny
            // C. HoughLinesP
            // =================================================

            val candidates = mutableListOf<PlateCandidate>()

            // -------------------------------------------------
            // A. Contour 후보
            // -------------------------------------------------

            val contourCandidates =
                extractContourCandidates(roiGray)

            candidates.addAll(
                contourCandidates.mapNotNull { pts ->

                    val score =
                        evaluateCandidate(
                            pts,
                            localAiRect,
                            safeRoi.width,
                            safeRoi.height
                        )

                    if (score.score >= MIN_CANDIDATE_SCORE) {
                        PlateCandidate(
                            pts,
                            score.score,
                            score.log
                        )
                    } else {
                        null
                    }
                }
            )

            // -------------------------------------------------
            // B. Hough 기반 실제 4선 조합
            // -------------------------------------------------

            val houghCandidates =
                extractHoughQuadCandidates(roiGray)

            candidates.addAll(
                houghCandidates.mapNotNull { pts ->

                    val score =
                        evaluateCandidate(
                            pts,
                            localAiRect,
                            safeRoi.width,
                            safeRoi.height
                        )

                    if (score.score >= MIN_CANDIDATE_SCORE) {
                        PlateCandidate(
                            pts,
                            score.score,
                            score.log
                        )
                    } else {
                        null
                    }
                }
            )

            // =================================================
            // 최고 후보
            // =================================================

            val bestCandidate =
                candidates
                    .maxByOrNull { it.score }

            // =================================================
            // 매우 중요
            //
            // 후보가 없거나 신뢰도가 낮으면 실패.
            // AI Box로 fallback 하지 않는다.
            // =================================================

            if (
                bestCandidate == null ||
                bestCandidate.score < MIN_FINAL_SCORE
            ) {

                debugListener?.let {

                    val debugMat = fullMat.clone()

                    Imgproc.rectangle(
                        debugMat,
                        safeRoi,
                        Scalar(
                            0.0,
                            255.0,
                            255.0,
                            255.0
                        ),
                        3
                    )

                    val debugBmp =
                        Bitmap.createBitmap(
                            debugMat.cols(),
                            debugMat.rows(),
                            Bitmap.Config.ARGB_8888
                        )

                    Utils.matToBitmap(
                        debugMat,
                        debugBmp
                    )

                    it.pauseAndShowStep(
                        "OpenCV 정밀화 실패",
                        debugBmp,
                        "번호판 4점 확정 실패",
                        listOf(
                            "AI Box는 정상적으로 검출됨",
                            "하지만 신뢰할 수 있는 4점 후보를 찾지 못함",
                            "→ AI Box fallback을 사용하지 않음",
                            "→ 마스킹하지 않음"
                        )
                    )

                    debugMat.release()
                    debugBmp.recycle()
                }

                roiGray.release()

                return null
            }

            // =================================================
            // 최종 4점
            // =================================================

            val finalLocalPts =
                sortCorners(bestCandidate.pts)

            val globalPts =
                finalLocalPts.map {

                    ImmutablePoint(
                        (it.x + safeRoi.x).toFloat(),
                        (it.y + safeRoi.y).toFloat()
                    )
                }

            // =================================================
            // 디버그
            // =================================================

            debugListener?.let {

                val debugMat =
                    fullMat.clone()

                // AI Box
                Imgproc.rectangle(
                    debugMat,
                    safeRoi,
                    Scalar(
                        0.0,
                        255.0,
                        255.0,
                        255.0
                    ),
                    3
                )

                // 최종 4점
                for (i in 0 until 4) {

                    val p1 =
                        Point(
                            globalPts[i].x.toDouble(),
                            globalPts[i].y.toDouble()
                        )

                    val p2 =
                        Point(
                            globalPts[(i + 1) % 4].x.toDouble(),
                            globalPts[(i + 1) % 4].y.toDouble()
                        )

                    Imgproc.line(
                        debugMat,
                        p1,
                        p2,
                        Scalar(
                            0.0,
                            255.0,
                            0.0,
                            255.0
                        ),
                        5
                    )

                    Imgproc.circle(
                        debugMat,
                        p1,
                        10,
                        Scalar(
                            255.0,
                            0.0,
                            255.0,
                            255.0
                        ),
                        -1
                    )
                }

                val debugBmp =
                    Bitmap.createBitmap(
                        debugMat.cols(),
                        debugMat.rows(),
                        Bitmap.Config.ARGB_8888
                    )

                Utils.matToBitmap(
                    debugMat,
                    debugBmp
                )

                it.pauseAndShowStep(
                    "AI → OpenCV 정밀화",
                    debugBmp,
                    "최종 번호판 4점",
                    listOf(
                        "AI Box 내부에서만 정밀화",
                        "AI Box 확장 없음",
                        "후보 수: ${candidates.size}",
                        "최종 점수: ${
                            String.format(
                                "%.1f",
                                bestCandidate.score
                            )
                        }",
                        bestCandidate.debugLog
                    )
                )

                debugMat.release()
                debugBmp.recycle()
            }

            roiGray.release()

            return globalPts

        } finally {

            fullGray.release()
            fullMat.release()
        }
    }

    // =========================================================
    // Contour 후보 생성
    // =========================================================

    private fun extractContourCandidates(
        gray: Mat
    ): List<List<Point>> {

        val result =
            mutableListOf<List<Point>>()

        // ---------------------------------------------
        // 두 가지 edge 경로
        // ---------------------------------------------

        val edgeMats =
            mutableListOf<Mat>()

        // A. 기본 Canny
        run {

            val blurred = Mat()
            val edges = Mat()

            Imgproc.GaussianBlur(
                gray,
                blurred,
                Size(3.0, 3.0),
                0.0
            )

            Imgproc.Canny(
                blurred,
                edges,
                40.0,
                120.0
            )

            edgeMats.add(edges)

            blurred.release()
        }

        // B. 약한 Close
        run {

            val blurred = Mat()
            val edges = Mat()

            Imgproc.GaussianBlur(
                gray,
                blurred,
                Size(3.0, 3.0),
                0.0
            )

            Imgproc.Canny(
                blurred,
                edges,
                40.0,
                120.0
            )

            val kernel =
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    Size(2.0, 2.0)
                )

            Imgproc.morphologyEx(
                edges,
                edges,
                Imgproc.MORPH_CLOSE,
                kernel
            )

            kernel.release()

            edgeMats.add(edges)

            blurred.release()
        }

        // ---------------------------------------------
        // contour
        // ---------------------------------------------

        for (edges in edgeMats) {

            val contours =
                ArrayList<MatOfPoint>()

            val hierarchy = Mat()

            Imgproc.findContours(
                edges,
                contours,
                hierarchy,
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            hierarchy.release()

            for (contour in contours) {

                if (contour.total() < 4) {
                    contour.release()
                    continue
                }

                val contour2f =
                    MatOfPoint2f(
                        *contour.toArray()
                    )

                val peri =
                    Imgproc.arcLength(
                        contour2f,
                        true
                    )

                val approx =
                    MatOfPoint2f()

                Imgproc.approxPolyDP(
                    contour2f,
                    approx,
                    0.02 * peri,
                    true
                )

                val points =
                    approx.toArray().toList()

                if (
                    points.size == 4 &&
                    Imgproc.isContourConvex(
                        MatOfPoint(*points.toTypedArray())
                    )
                ) {

                    result.add(
                        sortCorners(points)
                    )
                }

                approx.release()
                contour2f.release()
                contour.release()
            }

            edges.release()
        }

        return result
    }

    // =========================================================
    // HoughLinesP → 실제 4개 선 조합
    // =========================================================

    private fun extractHoughQuadCandidates(
        gray: Mat
    ): List<List<Point>> {

        val result =
            mutableListOf<List<Point>>()

        val edges = Mat()

        Imgproc.Canny(
            gray,
            edges,
            40.0,
            120.0
        )

        val lines = Mat()

        Imgproc.HoughLinesP(
            edges,
            lines,
            1.0,
            Math.PI / 180.0,
            25,
            min(gray.cols(), gray.rows()) * 0.18,
            8.0
        )

        val houghLines =
            mutableListOf<HoughLine>()

        for (i in 0 until lines.rows()) {

            val v =
                lines.get(i, 0)
                    ?: continue

            val p1 =
                Point(
                    v[0],
                    v[1]
                )

            val p2 =
                Point(
                    v[2],
                    v[3]
                )

            val dx = p2.x - p1.x
            val dy = p2.y - p1.y

            val length =
                hypot(dx, dy)

            if (length < 20.0) {
                continue
            }

            var angle =
                Math.toDegrees(
                    atan2(dy, dx)
                )

            if (angle < 0) {
                angle += 180.0
            }

            houghLines.add(
                HoughLine(
                    p1,
                    p2,
                    length,
                    angle
                )
            )
        }

        lines.release()
        edges.release()

        // 너무 많은 조합 방지
        val selectedLines =
            houghLines
                .sortedByDescending { it.length }
                .take(24)

        // ---------------------------------------------
        // 서로 거의 평행한 두 선을 찾는다.
        // ---------------------------------------------

        val parallelPairs =
            mutableListOf<Pair<HoughLine, HoughLine>>()

        for (i in selectedLines.indices) {

            for (j in i + 1 until selectedLines.size) {

                val a =
                    selectedLines[i]

                val b =
                    selectedLines[j]

                val diff =
                    angleDifference(
                        a.angle,
                        b.angle
                    )

                if (diff <= 12.0) {
                    parallelPairs.add(
                        Pair(a, b)
                    )
                }
            }
        }

        // ---------------------------------------------
        // 서로 거의 직각인 두 방향의 pair를 결합
        // ---------------------------------------------

        for (i in parallelPairs.indices) {

            val pairA =
                parallelPairs[i]

            val angleA =
                pairA.first.angle

            for (j in i + 1 until parallelPairs.size) {

                val pairB =
                    parallelPairs[j]

                val angleB =
                    pairB.first.angle

                val perpendicularDiff =
                    abs(
                        angleDifference(
                            angleA,
                            angleB
                        ) - 90.0
                    )

                if (perpendicularDiff > 20.0) {
                    continue
                }

                val quad =
                    buildQuadFromLines(
                        pairA.first,
                        pairA.second,
                        pairB.first,
                        pairB.second
                    )
                    ?: continue

                if (
                    isReasonableQuad(
                        quad,
                        gray.cols(),
                        gray.rows()
                    )
                ) {

                    result.add(
                        sortCorners(quad)
                    )
                }
            }
        }

        return result
    }

    // =========================================================
    // 4개 선의 교점으로 사각형 생성
    // =========================================================

    private fun buildQuadFromLines(
        a1: HoughLine,
        a2: HoughLine,
        b1: HoughLine,
        b2: HoughLine
    ): List<Point>? {

        val p1 =
            intersection(
                a1.p1,
                a1.p2,
                b1.p1,
                b1.p2
            )

        val p2 =
            intersection(
                a1.p1,
                a1.p2,
                b2.p1,
                b2.p2
            )

        val p3 =
            intersection(
                a2.p1,
                a2.p2,
                b2.p1,
                b2.p2
            )

        val p4 =
            intersection(
                a2.p1,
                a2.p2,
                b1.p1,
                b1.p2
            )

        if (
            p1 == null ||
            p2 == null ||
            p3 == null ||
            p4 == null
        ) {
            return null
        }

        return listOf(
            p1,
            p2,
            p3,
            p4
        )
    }

    // =========================================================
    // 두 직선 교점
    // =========================================================

    private fun intersection(
        p1: Point,
        p2: Point,
        p3: Point,
        p4: Point
    ): Point? {

        val x1 = p1.x
        val y1 = p1.y
        val x2 = p2.x
        val y2 = p2.y

        val x3 = p3.x
        val y3 = p3.y
        val x4 = p4.x
        val y4 = p4.y

        val denominator =
            (x1 - x2) * (y3 - y4) -
            (y1 - y2) * (x3 - x4)

        if (abs(denominator) < 1e-6) {
            return null
        }

        val px =
            (
                (x1 * y2 - y1 * x2) * (x3 - x4) -
                (x1 - x2) * (x3 * y4 - y3 * x4)
            ) / denominator

        val py =
            (
                (x1 * y2 - y1 * x2) * (y3 - y4) -
                (y1 - y2) * (x3 * y4 - y3 * x4)
            ) / denominator

        return Point(px, py)
    }

    // =========================================================
    // 사각형 기본 유효성
    // =========================================================

    private fun isReasonableQuad(
        pts: List<Point>,
        width: Int,
        height: Int
    ): Boolean {

        if (pts.size != 4) {
            return false
        }

        for (p in pts) {

            if (
                p.x < -width * 0.15 ||
                p.x > width * 1.15 ||
                p.y < -height * 0.15 ||
                p.y > height * 1.15
            ) {
                return false
            }
        }

        val ordered =
            sortCorners(pts)

        val area =
            abs(
                polygonArea(ordered)
            )

        if (area < width * height * 0.03) {
            return false
        }

        return true
    }

    // =========================================================
    // 후보 평가
    // =========================================================

    private fun evaluateCandidate(
        pts: List<Point>,
        aiRect: Rect,
        roiWidth: Int,
        roiHeight: Int
    ): ScoreResult {

        if (pts.size != 4) {
            return ScoreResult(
                0.0,
                "4점 아님"
            )
        }

        val p =
            sortCorners(pts)

        val tl = p[0]
        val tr = p[1]
        val br = p[2]
        val bl = p[3]

        // ---------------------------------------------
        // 변 길이
        // ---------------------------------------------

        val top =
            hypot(
                tr.x - tl.x,
                tr.y - tl.y
            )

        val bottom =
            hypot(
                br.x - bl.x,
                br.y - bl.y
            )

        val left =
            hypot(
                bl.x - tl.x,
                bl.y - tl.y
            )

        val right =
            hypot(
                br.x - tr.x,
                br.y - tr.y
            )

        if (
            top <= 1 ||
            bottom <= 1 ||
            left <= 1 ||
            right <= 1
        ) {
            return ScoreResult(
                0.0,
                "변 길이 오류"
            )
        }

        val widthAvg =
            (top + bottom) / 2.0

        val heightAvg =
            (left + right) / 2.0

        val aspectRatio =
            widthAvg / heightAvg

        // ---------------------------------------------
        // 1. 종횡비
        // ---------------------------------------------

        val aspectScore =
            if (aspectRatio in 1.8..6.0) {

                100.0 -
                    abs(aspectRatio - 3.0) * 12.0

            } else {

                max(
                    0.0,
                    100.0 -
                        abs(aspectRatio - 3.0) * 30.0
                )
            }

        // ---------------------------------------------
        // 2. 상하변 평행성
        // ---------------------------------------------

        val topAngle =
            lineAngle(tl, tr)

        val bottomAngle =
            lineAngle(bl, br)

        val horizontalParallel =
            angleDifference(
                topAngle,
                bottomAngle
            )

        val horizontalScore =
            max(
                0.0,
                100.0 -
                    horizontalParallel * 6.0
            )

        // ---------------------------------------------
        // 3. 좌우변 평행성
        // ---------------------------------------------

        val leftAngle =
            lineAngle(tl, bl)

        val rightAngle =
            lineAngle(tr, br)

        val verticalParallel =
            angleDifference(
                leftAngle,
                rightAngle
            )

        val verticalScore =
            max(
                0.0,
                100.0 -
                    verticalParallel * 6.0
            )

        val parallelScore =
            (horizontalScore + verticalScore) / 2.0

        // ---------------------------------------------
        // 4. AI Box 내부 적합도
        //
        // 후보의 실제 4점이 AI Box를 얼마나 잘 채우는가
        // ---------------------------------------------

        val candidateArea =
            abs(
                polygonArea(p)
            )

        val aiArea =
            aiRect.width.toDouble() *
            aiRect.height.toDouble()

        if (aiArea <= 0) {
            return ScoreResult(
                0.0,
                "AI Box 면적 오류"
            )
        }

        val areaRatio =
            candidateArea / aiArea

        val areaFitScore =
            when {
                areaRatio in 0.45..1.05 ->
                    100.0

                areaRatio < 0.45 ->
                    max(
                        0.0,
                        areaRatio / 0.45 * 100.0
                    )

                else ->
                    max(
                        0.0,
                        100.0 -
                            (areaRatio - 1.05) * 150.0
                    )
            }

        // ---------------------------------------------
        // 5. 중심 위치
        // ---------------------------------------------

        val centerX =
            p.map { it.x }.average()

        val centerY =
            p.map { it.y }.average()

        val aiCenterX =
            aiRect.x +
                aiRect.width / 2.0

        val aiCenterY =
            aiRect.y +
                aiRect.height / 2.0

        val centerDistance =
            hypot(
                centerX - aiCenterX,
                centerY - aiCenterY
            )

        val maxCenterDistance =
            hypot(
                aiRect.width.toDouble(),
                aiRect.height.toDouble()
            ) / 2.0

        val centerScore =
            max(
                0.0,
                100.0 -
                    (
                        centerDistance /
                            maxCenterDistance.coerceAtLeast(1.0)
                    ) * 100.0
            )

        // ---------------------------------------------
        // 6. AI Box 밖으로 나간 점
        // ---------------------------------------------

        var overflow = 0.0

        for (point in p) {

            val dx =
                when {
                    point.x < aiRect.x ->
                        aiRect.x - point.x

                    point.x >
                        aiRect.x + aiRect.width ->
                        point.x -
                            (aiRect.x + aiRect.width)

                    else -> 0.0
                }

            val dy =
                when {
                    point.y < aiRect.y ->
                        aiRect.y - point.y

                    point.y >
                        aiRect.y + aiRect.height ->
                        point.y -
                            (aiRect.y + aiRect.height)

                    else -> 0.0
                }

            overflow += hypot(dx, dy)
        }

        val overflowRatio =
            overflow /
                (
                    aiRect.width +
                        aiRect.height
                ).toDouble()

        val overflowScore =
            max(
                0.0,
                100.0 -
                    overflowRatio * 200.0
            )

        // ---------------------------------------------
        // 최종 점수
        // ---------------------------------------------

        val finalScore =
            (
                aspectScore * 0.20 +
                parallelScore * 0.25 +
                areaFitScore * 0.20 +
                centerScore * 0.20 +
                overflowScore * 0.15
            )

        val log =
            "점수=${String.format("%.1f", finalScore)} " +
            "AR=${aspectScore.toInt()} " +
            "평행=${parallelScore.toInt()} " +
            "크기=${areaFitScore.toInt()} " +
            "중심=${centerScore.toInt()} " +
            "Overflow=${overflowScore.toInt()}"

        return ScoreResult(
            finalScore,
            log
        )
    }

    // =========================================================
    // 선 각도
    // =========================================================

    private fun lineAngle(
        a: Point,
        b: Point
    ): Double {

        var angle =
            Math.toDegrees(
                atan2(
                    b.y - a.y,
                    b.x - a.x
                )
            )

        if (angle < 0) {
            angle += 180.0
        }

        return angle
    }

    // =========================================================
    // 각도 차이
    // =========================================================

    private fun angleDifference(
        a: Double,
        b: Double
    ): Double {

        var diff =
            abs(a - b)

        while (diff > 180.0) {
            diff -= 180.0
        }

        return min(
            diff,
            180.0 - diff
        )
    }

    // =========================================================
    // Polygon 면적
    // =========================================================

    private fun polygonArea(
        pts: List<Point>
    ): Double {

        var area = 0.0

        for (i in pts.indices) {

            val j =
                (i + 1) % pts.size

            area +=
                pts[i].x * pts[j].y -
                pts[j].x * pts[i].y
        }

        return area / 2.0
    }

    // =========================================================
    // 꼭지점 정렬
    // =========================================================

    private fun sortCorners(
        pts: List<Point>
    ): List<Point> {

        if (pts.size != 4) {
            return pts
        }

        val centerX =
            pts.map { it.x }.average()

        val centerY =
            pts.map { it.y }.average()

        val sorted =
            pts.sortedBy {

                atan2(
                    it.y - centerY,
                    it.x - centerX
                )
            }

        // 시계 방향으로 정렬된 뒤
        // 가장 좌상단에 가까운 점을 첫 점으로 이동

        val startIndex =
            sorted.indices.minByOrNull { i ->
                sorted[i].x +
                    sorted[i].y
            } ?: 0

        return List(4) { index ->
            sorted[
                (startIndex + index) % 4
            ]
        }
    }

    // =========================================================
    // 최소 점수
    // =========================================================

    private const val MIN_CANDIDATE_SCORE = 35.0

    private const val MIN_FINAL_SCORE = 65.0
}
