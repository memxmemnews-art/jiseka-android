package com.jiseka.app

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(stageName: String, bitmap: Bitmap)
    }

    fun rescuePlateFromLine(
        fullBitmap: Bitmap, 
        startX: Float, startY: Float, 
        endX: Float, endY: Float,
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat()
        val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        // 방향 정규화 (무조건 좌 -> 우)
        var p1x = startX.toDouble(); var p1y = startY.toDouble()
        var p2x = endX.toDouble(); var p2y = endY.toDouble()
        if (p1x > p2x) {
            val tx = p1x; val ty = p1y; p1x = p2x; p1y = p2y; p2x = tx; p2y = ty
        }

        val dx = p2x - p1x
        val dy = p2y - p1y
        val lineLen = hypot(dx, dy)
        val cx = (p1x + p2x) / 2.0
        val cy = (p1y + p2y) / 2.0

        // =====================================================================
        // [1단계] 안전 영역(Padded ROI) 1차 추출
        // 회전 시 번호판이 잘리지 않도록 선 길이의 2배 크기로 넉넉하게 자릅니다.
        // =====================================================================
        val paddedSize = lineLen * 2.0 
        val paddedLeft = (cx - paddedSize / 2.0).toInt().coerceIn(0, fullMat.cols() - 1)
        val paddedTop = (cy - paddedSize / 2.0).toInt().coerceIn(0, fullMat.rows() - 1)
        val paddedRight = (cx + paddedSize / 2.0).toInt().coerceIn(1, fullMat.cols())
        val paddedBottom = (cy + paddedSize / 2.0).toInt().coerceIn(1, fullMat.rows())

        val paddedRect = Rect(paddedLeft, paddedTop, paddedRight - paddedLeft, paddedBottom - paddedTop)
        
        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.line(debugMat, Point(p1x, p1y), Point(p2x, p2y), Scalar(255.0, 255.0, 0.0, 255.0), 8)
            Imgproc.rectangle(debugMat, paddedRect, Scalar(255.0, 0.0, 0.0, 255.0), 12)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            it.pauseAndShowStep("1단계: 회전 대비 안전 영역(Padded ROI) 확보", debugBmp)
            debugMat.release()
        }

        val paddedMat = Mat(); val paddedGray = Mat()
        fullMat.submat(paddedRect).copyTo(paddedMat)
        fullGray.submat(paddedRect).copyTo(paddedGray)

        // 안전 영역 내부에서의 선 중심점 좌표 재계산
        val roiCx = cx - paddedLeft
        val roiCy = cy - paddedTop

        // =====================================================================
        // [2단계] 작은 안전 영역만 회전시킨 후 타이트한 2차 추출
        // =====================================================================
        val angle = Math.toDegrees(Math.atan2(dy, dx))
        val rotMat = Imgproc.getRotationMatrix2D(Point(roiCx, roiCy), -angle, 1.0)

        val rotatedPaddedMat = Mat(); val rotatedPaddedGray = Mat()
        Imgproc.warpAffine(paddedMat, rotatedPaddedMat, rotMat, paddedMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(paddedGray, rotatedPaddedGray, rotMat, paddedGray.size(), Imgproc.INTER_LINEAR)

        val expectedPlateHeight = lineLen / 4.7
        val tightWidth = max(100.0, lineLen * 1.1)
        val tightHeight = max(60.0, expectedPlateHeight * 2.0)

        val tightLeft = (roiCx - tightWidth / 2.0).toInt().coerceIn(0, rotatedPaddedMat.cols() - 1)
        val tightTop = (roiCy - tightHeight / 2.0).toInt().coerceIn(0, rotatedPaddedMat.rows() - 1)
        val tightRight = (roiCx + tightWidth / 2.0).toInt().coerceIn(1, rotatedPaddedMat.cols())
        val tightBottom = (roiCy + tightHeight / 2.0).toInt().coerceIn(1, rotatedPaddedMat.rows())

        if (tightRight - tightLeft <= 10 || tightBottom - tightTop <= 10) {
            fullMat.release(); fullGray.release(); rotMat.release()
            paddedMat.release(); paddedGray.release()
            rotatedPaddedMat.release(); rotatedPaddedGray.release()
            return null
        }

        val safeTightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)
        val tightGray = Mat()
        rotatedPaddedGray.submat(safeTightRect).copyTo(tightGray)
        val tightImageArea = safeTightRect.width * safeTightRect.height.toDouble()

        debugListener?.let {
            val debugMat = rotatedPaddedMat.clone()
            Imgproc.rectangle(debugMat, safeTightRect, Scalar(0.0, 255.0, 0.0, 255.0), 12)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            it.pauseAndShowStep("2단계: 수평 정렬 후 타이트한 정밀 ROI 추출", debugBmp)
            debugMat.release()
        }

        val roiEdge = Mat(); val combinedEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>(); val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            // =====================================================================
            // [3단계] CLAHE 적용
            // =====================================================================
            val clahe = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
            clahe.apply(tightGray, tightGray)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(tightGray.cols(), tightGray.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(tightGray, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                it.pauseAndShowStep("3단계: 흑백 CLAHE (대비 강화)", debugBmp)
                tempRgb.release()
            }

            // =====================================================================
            // [4단계] Canny Edge
            // =====================================================================
            val meanVal = Core.mean(tightGray).`val`[0]
            Imgproc.Canny(tightGray, roiEdge, max(0.0, 0.33 * meanVal), min(255.0, 1.33 * meanVal))
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(roiEdge.cols(), roiEdge.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(roiEdge, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                it.pauseAndShowStep("4단계: Canny Edge (경계선 추출)", debugBmp)
                tempRgb.release()
            }

            // =====================================================================
            // [5단계] 모폴로지 닫기
            // =====================================================================
            val kernelLen = (safeTightRect.width * 0.08).toInt().coerceIn(10, 60)
            val kernel0 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelLen.toDouble(), 2.0))
            Imgproc.morphologyEx(roiEdge, combinedEdge, Imgproc.MORPH_CLOSE, kernel0)
            kernel0.release()

            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(combinedEdge.cols(), combinedEdge.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(combinedEdge, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                it.pauseAndShowStep("5단계: 모폴로지 연산 (끊어진 선 잇기)", debugBmp)
                tempRgb.release()
            }

            // =====================================================================
            // [6단계] 다각형 윤곽선 추려내기
            // =====================================================================
            Imgproc.findContours(combinedEdge, roiContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            
            val validPolygons = ArrayList<MatOfPoint2f>()
            for (contour in roiContours) {
                val approx = extractRobustPolygon(contour) ?: continue
                validPolygons.add(approx)
            }

            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(tightGray, debugMat, Imgproc.COLOR_GRAY2RGBA)
                for (poly in validPolygons) {
                    val pts = MatOfPoint(*poly.toArray())
                    Imgproc.drawContours(debugMat, listOf(pts), -1, Scalar(0.0, 255.0, 255.0, 255.0), 3)
                    pts.release()
                }
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                it.pauseAndShowStep("6단계: 다각형 후보군 추려내기 (${validPolygons.size}개 발견)", debugBmp)
                debugMat.release()
            }

            // =====================================================================
            // [7단계] 깐깐한 형태 검사
            // =====================================================================
            var maxArea = -1.0
            var bestApprox2f: MatOfPoint2f? = null

            for (poly in validPolygons) {
                val contourArea = Imgproc.contourArea(poly)
                if (isValidRescueGeometry(contourArea, poly.toArray(), tightImageArea)) {
                    if (contourArea > maxArea) {
                        maxArea = contourArea
                        bestApprox2f?.release()
                        bestApprox2f = poly.clone() as MatOfPoint2f
                    }
                }
            }

            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(tightGray, debugMat, Imgproc.COLOR_GRAY2RGBA)
                if (bestApprox2f != null) {
                    val pts = MatOfPoint(*bestApprox2f!!.toArray())
                    Imgproc.drawContours(debugMat, listOf(pts), -1, Scalar(0.0, 255.0, 0.0, 255.0), 4)
                    pts.release()
                }
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                it.pauseAndShowStep("7단계: 최종 번호판 형태 결정", debugBmp)
                debugMat.release()
            }

            // =====================================================================
            // [8단계] 이중 역변환 (타이트 ROI -> 안전 영역 -> 글로벌 원본)
            // =====================================================================
            if (bestApprox2f != null) {
                val invRotMat = Mat()
                Imgproc.invertAffineTransform(rotMat, invRotMat)

                // 1. 타이트 좌표를 안전 영역 좌표계로 이동
                val rotatedPoints = bestApprox2f!!.toArray().map { Point(it.x + safeTightRect.x, it.y + safeTightRect.y) }.toTypedArray()
                val srcMat = MatOfPoint2f(*rotatedPoints)
                val dstMat = MatOfPoint2f()

                // 2. 안전 영역 내에서 역회전 적용
                Core.transform(srcMat, dstMat, invRotMat)

                // 3. 글로벌 좌표계로 다시 이동
                resultPoints = dstMat.toArray().map { 
                    ImmutablePoint((it.x + paddedRect.x).toFloat(), (it.y + paddedRect.y).toFloat()) 
                }

                debugListener?.let {
                    val debugMat = fullMat.clone()
                    val finalPts = resultPoints!!.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()
                    val restoredPts = MatOfPoint(*finalPts)
                    Imgproc.drawContours(debugMat, listOf(restoredPts), -1, Scalar(255.0, 0.0, 255.0, 255.0), 6)
                    val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(debugMat, debugBmp)
                    it.pauseAndShowStep("8단계: 원래 각도와 좌표로 이중 역변환 완료", debugBmp)
                    debugMat.release(); restoredPts.release()
                }

                invRotMat.release(); srcMat.release(); dstMat.release()
            }

            validPolygons.forEach { it.release() }
            bestApprox2f?.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            // 메모리 최적화 완료: 더 이상 존재하지 않는 변수 오류가 발생하지 않습니다.
            roiEdge.release(); combinedEdge.release()
            roiContours.forEach { it.release() }; hierarchy.release()
            
            rotMat.release()
            paddedMat.release(); paddedGray.release()
            rotatedPaddedMat.release(); rotatedPaddedGray.release()
            tightGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }

    private fun extractRobustPolygon(contour: MatOfPoint): MatOfPoint2f? {
        val contour2f = MatOfPoint2f(*contour.toArray())
        val approx2f = MatOfPoint2f()
        val perimeter = Imgproc.arcLength(contour2f, true)
        Imgproc.approxPolyDP(contour2f, approx2f, perimeter * 0.02, true)
        
        val approxPointMat = MatOfPoint(*approx2f.toArray())
        val hullIndices = MatOfInt()
        Imgproc.convexHull(approxPointMat, hullIndices)
        
        val hullPoints = mutableListOf<Point>()
        val approxPointsArray = approxPointMat.toArray()
        for (index in hullIndices.toArray()) {
            hullPoints.add(approxPointsArray[index])
        }
        approx2f.release(); contour2f.release(); approxPointMat.release(); hullIndices.release()

        if (hullPoints.size in 4..12) { 
            return MatOfPoint2f(*sortPolygonPoints(hullPoints).toTypedArray())
        }
        return null
    }

    private fun isValidRescueGeometry(originalContourArea: Double, hullPoints: Array<Point>, roiArea: Double): Boolean {
        val hullMat = MatOfPoint(*hullPoints)
        val hullArea = Imgproc.contourArea(hullMat)
        val solidity = originalContourArea / hullArea
        if (solidity < 0.85) { hullMat.release(); return false }

        val normalizedArea = originalContourArea / roiArea
        if (normalizedArea < 0.05 || normalizedArea > 0.95) { hullMat.release(); return false }

        val hullMat2f = MatOfPoint2f(*hullPoints)
        val minAreaRect = Imgproc.minAreaRect(hullMat2f)
        val rectArea = minAreaRect.size.width * minAreaRect.size.height
        val rectangularity = if (rectArea > 0) originalContourArea / rectArea else 0.0
        if (rectangularity < 0.30) { hullMat.release(); hullMat2f.release(); return false }

        var w = minAreaRect.size.width; var h = minAreaRect.size.height
        if (w < h) { val temp = w; w = h; h = temp }
        if (h < 1e-6 || w / h !in 2.2..6.5) { hullMat.release(); hullMat2f.release(); return false }

        hullMat.release(); hullMat2f.release()
        return true
    }

    private fun sortPolygonPoints(points: List<Point>): List<Point> {
        val cx = points.map { it.x }.average()
        val cy = points.map { it.y }.average()
        var sorted = points.sortedBy { Math.atan2(it.y - cy, it.x - cx) }
        var area = 0.0
        for (i in sorted.indices) {
            val p1 = sorted[i]; val p2 = sorted[(i + 1) % sorted.size]
            area += (p1.x * p2.y - p2.x * p1.y)
        }
        if (area < 0) sorted = sorted.reversed()
        return sorted
    }
}
