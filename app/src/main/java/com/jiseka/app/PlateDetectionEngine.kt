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

    data class SeedCropResult(
        val offsetX: Int,
        val offsetY: Int,
        val croppedBitmap: Bitmap,
        val roiRect: Rect
    )

    // 스코어링된 사각형 후보 데이터 클래스
    private data class PlateCandidate(
        val pts: List<Point>,
        val score: Double,
        val debugLog: String
    )

    private fun getSafeRect(x: Int, y: Int, w: Int, h: Int, maxW: Int, maxH: Int): Rect {
        val safeX = x.coerceIn(0, maxW - 1)
        val safeY = y.coerceIn(0, maxH - 1)
        val safeW = w.coerceAtMost(maxW - safeX)
        val safeH = h.coerceAtMost(maxH - safeY)
        return Rect(safeX, safeY, safeW, safeH)
    }

    // 1. 터치 주변 크롭 (최소 픽셀 보장 및 작은 3~5% 마진 적용)
    fun prepareWideCrop(fullBitmap: Bitmap, touchX: Float, touchY: Float): SeedCropResult {
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
            fullBitmap.width, fullBitmap.height
        )

        val croppedBitmap = Bitmap.createBitmap(fullBitmap, safeRect.x, safeRect.y, safeRect.width, safeRect.height)
        return SeedCropResult(safeRect.x, safeRect.y, croppedBitmap, safeRect)
    }

    // 2. 3중 방어 파이프라인 메인 진입점
    suspend fun processWithMLKitResult(
        fullBitmap: Bitmap, 
        aiGlobalBox: android.graphics.Rect, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat()
        val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        // [1차 방어선: AI Box 제한] 그릴 유입을 막기 위해 15%가 아닌 5%의 최소한의 마진만 확장
        val marginX = (aiGlobalBox.width() * 0.05f).toInt()
        val marginY = (aiGlobalBox.height() * 0.05f).toInt()
        val safeRoi = getSafeRect(
            aiGlobalBox.left - marginX,
            aiGlobalBox.top - marginY,
            aiGlobalBox.width() + marginX * 2,
            aiGlobalBox.height() + marginY * 2,
            fullMat.cols(), fullMat.rows()
        )

        val roiGray = Mat()
        fullGray.submat(safeRoi).copyTo(roiGray)

        // [2차 방어선: 3-Way 다중 후보 생성]
        val rawContours = mutableListOf<MatOfPoint>()
        
        // 경로 A: 원본 Canny
        val edgesA = Mat()
        Imgproc.GaussianBlur(roiGray, edgesA, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(edgesA, edgesA, 50.0, 150.0)
        extractContoursTo(edgesA, rawContours)
        edgesA.release()

        // 경로 B: 약한 Close 연산 적용
        val edgesB = Mat()
        Imgproc.GaussianBlur(roiGray, edgesB, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(edgesB, edgesB, 50.0, 150.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(edgesB, edgesB, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        extractContoursTo(edgesB, rawContours)
        edgesB.release()

        // 경로 C: HoughLinesP 기반 직선 조합 후보 생성
        val houghContours = extractHoughRects(roiGray)
        rawContours.addAll(houghContours)

        // [3차 방어선: Geometry 스코어링 평가 시스템]
        val candidates = mutableListOf<PlateCandidate>()
        val roiArea = (safeRoi.width * safeRoi.height).toDouble()
        
        // AI 기준 박스 (로컬 좌표계)
        val localAiRect = Rect(
            aiGlobalBox.left - safeRoi.x,
            aiGlobalBox.top - safeRoi.y,
            aiGlobalBox.width(),
            aiGlobalBox.height()
        )

        for (contour in rawContours) {
            val peri = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, 0.03 * peri, true)

            if (approx.toArray().size == 4 && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                val area = Imgproc.contourArea(approx)
                
                // [필터] 너무 작거나 ROI 전체를 다 먹어버리는 거대한 contour는 제거
                if (area < roiArea * 0.08 || area > roiArea * 0.95) {
                    approx.release()
                    continue
                }

                val pts = sortCorners(approx.toArray().toList())
                val scoreResult = evaluateCandidate(pts, localAiRect, roiWidth = safeRoi.width, roiHeight = safeRoi.height)

                if (scoreResult.score > 30.0) { // 최소 커트라인
                    candidates.add(PlateCandidate(pts, scoreResult.score, scoreResult.log))
                }
            }
            approx.release()
        }

        roiGray.release()
        rawContours.forEach { it.release() }

        // 최고 점수 후보 채택 (없을 경우 AI Box 기본 사각형으로 폴백)
        val bestCandidate = candidates.maxByOrNull { it.score }
        
        val finalLocalPts = if (bestCandidate != null) {
            bestCandidate.pts
        } else {
            listOf(
                Point(localAiRect.x.toDouble(), localAiRect.y.toDouble()),
                Point((localAiRect.x + localAiRect.width).toDouble(), localAiRect.y.toDouble()),
                Point((localAiRect.x + localAiRect.width).toDouble(), (localAiRect.y + localAiRect.height).toDouble()),
                Point(localAiRect.x.toDouble(), (localAiRect.y + localAiRect.height).toDouble())
            )
        }

        // 글로벌 화면 좌표로 복원
        val globalPts = finalLocalPts.map {
            ImmutablePoint((it.x + safeRoi.x).toFloat(), (it.y + safeRoi.y).toFloat())
        }

        // 디버그 리스너 로깅
        debugListener?.let {
            val debugMat = fullMat.clone()
            val cvRect = Rect(aiGlobalBox.left, aiGlobalBox.top, aiGlobalBox.width(), aiGlobalBox.height())
            Imgproc.rectangle(debugMat, cvRect, Scalar(0.0, 255.0, 255.0, 255.0), 3)

            for (i in 0..3) {
                val pt1 = Point(globalPts[i].x.toDouble(), globalPts[i].y.toDouble())
                val pt2 = Point(globalPts[(i+1)%4].x.toDouble(), globalPts[(i+1)%4].y.toDouble())
                Imgproc.line(debugMat, pt1, pt2, Scalar(0.0, 255.0, 0.0, 255.0), 5)
                Imgproc.circle(debugMat, pt1, 10, Scalar(255.0, 0.0, 255.0, 255.0), -1)
            }

            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)

            val logList = mutableListOf<String>()
            if (bestCandidate != null) {
                logList.add("-> [성공] 3중 방어 통과 최고 점수 채택")
                logList.add("-> ${bestCandidate.debugLog}")
            } else {
                logList.add("-> [경고] 기하학 조건을 만족하는 후보가 없어 AI Box로 대체합니다.")
            }

            it.pauseAndShowStep("디버그: 3중 방어 스코어링", debugBmp, "Geometry Scoring 결과", logList)
            debugMat.release(); debugBmp.recycle()
        }

        fullMat.release(); fullGray.release()
        return globalPts
    }

    // 보조 함수: Canny 에지에서 Contour 수집
    private fun extractContoursTo(edges: Mat, outList: MutableList<MatOfPoint>) {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        outList.addAll(contours)
        hierarchy.release()
    }

    // 보조 함수: HoughLinesP를 이용해 사각형 구조물 후보(MatOfPoint) 생성
    private fun extractHoughRects(gray: Mat): List<MatOfPoint> {
        val lines = Mat()
        Imgproc.HoughLinesP(gray, lines, 1.0, Math.PI / 180, 50, 30.0, 10.0)
        
        val pts = mutableListOf<Point>()
        for (i in 0 until lines.rows()) {
            val v = lines.get(i, 0)
            pts.add(Point(v[0], v[1]))
            pts.add(Point(v[2], v[3]))
        }
        lines.release()

        if (pts.size < 4) return emptyList()

        // 수집된 선들의 양 끝점을 감싸는 최소 외곽 사각형을 후보로 등록
        val matOfPt = MatOfPoint(*pts.toTypedArray())
        val rect = Imgproc.boundingRect(matOfPt)
        matOfPt.release()

        val rectContour = MatOfPoint(
            Point(rect.x.toDouble(), rect.y.toDouble()),
            Point((rect.x + rect.width).toDouble(), rect.y.toDouble()),
            Point((rect.x + rect.width).toDouble(), (rect.y + rect.height).toDouble()),
            Point(rect.x.toDouble(), (rect.y + rect.height).toDouble())
        )
        return listOf(rectContour)
    }

    // 7-Factor 기하학적 스코어링 평가 엔진
    private data class ScoreResult(val score: Double, val log: String)

    private fun evaluateCandidate(pts: List<Point>, aiRect: Rect, roiWidth: Int, roiHeight: Int): ScoreResult {
        val tl = pts[0]; val tr = pts[1]; val br = pts[2]; val bl = pts[3]

        val w = (hypot(tr.x - tl.x, tr.y - tl.y) + hypot(br.x - bl.x, br.y - bl.y)) / 2.0
        val h = (hypot(bl.x - tl.x, bl.y - tl.y) + hypot(br.x - tr.x, br.y - tr.y)) / 2.0
        if (h <= 0 || w <= 0) return ScoreResult(0.0, "Invalid dimension")

        // 1. 종횡비 검사 (한국 번호판 표준 비율: 약 2.0 ~ 5.5 범위 허용)
        val aspectRatio = w / h
        val arScore = if (aspectRatio in 1.8..6.0) 100.0 else max(0.0, 100.0 - abs(aspectRatio - 3.0) * 25.0)

        // 2. 평행성 검사 (상단 변과 하단 변의 각도 차이)
        val angleTop = Math.toDegrees(Math.atan2(tr.y - tl.y, tr.x - tl.x))
        val angleBottom = Math.toDegrees(Math.atan2(br.x - bl.x, br.y - bl.y)) // 수정된 각도 계산
        val angleDiff = abs(angleTop - angleBottom).let { if (it > 180) 360 - it else it }
        val parallelismScore = max(0.0, 100.0 - (angleDiff * 5.0))

        // 3. AI Box 일치도 (Overlap Score)
        val candRect = Rect(
            min(tl.x, bl.x).toInt(),
            min(tl.y, tr.y).toInt(),
            w.toInt(),
            h.toInt()
        )
        
        // 교집합 영역 계산
        val intersect = aiRect.intersect(candRect)
        val overlapArea = if (intersect) {
            val xOverlap = max(0, min(aiRect.x + aiRect.width, candRect.x + candRect.width) - max(aiRect.x, candRect.x))
            val yOverlap = max(0, min(aiRect.y + aiRect.height, candRect.y + candRect.height) - max(aiRect.y, candRect.y))
            (xOverlap * yOverlap).toDouble()
        } else 0.0

        val aiBoxArea = (aiRect.width * aiRect.height).toDouble()
        val overlapRatio = if (aiBoxArea > 0) overlapArea / aiBoxArea else 0.0
        val aiOverlapScore = overlapRatio * 100.0

        // 4. Overflow Penalty (AI Box 밖으로 튀어나간 면적에 대한 강력한 감점)
        val candArea = w * h
        val excessArea = max(0.0, candArea - overlapArea)
        val overflowPenalty = (excessArea / aiBoxArea) * 50.0

        // 최종 종합 점수 산출 (가중치 적용)
        val finalScore = (0.35 * aiOverlapScore) + (0.25 * arScore) + (0.25 * parallelismScore) - (0.15 * overflowPenalty)

        val logStr = "점수:${String.format("%.1f", finalScore)} (종횡비:${arScore.toInt()}, 평행:${parallelismScore.toInt()}, AI일치:${aiOverlapScore.toInt()})"
        return ScoreResult(max(0.0, finalScore), logStr)
    }

    // 꼭지점 정렬 유틸리티
    private fun sortCorners(pts: List<Point>): List<Point> {
        val sumSorted = pts.sortedBy { it.x + it.y }
        val tl = sumSorted.first()
        val br = sumSorted.last()

        val remaining = pts.filter { it != tl && it != br }
        val diffSorted = remaining.sortedBy { it.y - it.x }
        val tr = diffSorted.first()
        val bl = diffSorted.last()

        return listOf(tl, tr, br, bl)
    }
}
