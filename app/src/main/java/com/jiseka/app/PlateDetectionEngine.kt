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

        val roiCx = cx - paddedLeft
        val roiCy = cy - paddedTop

        // =====================================================================
        // [2단계] 수평 정렬
        // =====================================================================
        val angle = Math.toDegrees(Math.atan2(dy, dx))
        val rotMat = Imgproc.getRotationMatrix2D(Point(roiCx, roiCy), -angle, 1.0)

        val rotatedPaddedMat = Mat(); val rotatedPaddedGray = Mat()
        Imgproc.warpAffine(paddedMat, rotatedPaddedMat, rotMat, paddedMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(paddedGray, rotatedPaddedGray, rotMat, paddedGray.size(), Imgproc.INTER_LINEAR)

        // =====================================================================
        // [2.5단계] Edge Density 동적 탐색 (세로창 그릴 방어 및 디버그)
        // =====================================================================
        var tightTop = 0
        var tightBottom = rotatedPaddedGray.rows()
        
        val sobelX = Mat()
        val searchMat = Mat()
        val projection = Mat()
        
        try {
            // 1. 세로선(수직 엣지)만 강하게 추출
            Imgproc.Sobel(rotatedPaddedGray, sobelX, CvType.CV_32F, 1, 0, 3)
            Core.convertScaleAbs(sobelX, sobelX)
            Imgproc.threshold(sobelX, sobelX, 50.0, 255.0, Imgproc.THRESH_BINARY)

            // 🛠️ [디버그 2.5-1] 수직 엣지 추출 상태 확인
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(sobelX.cols(), sobelX.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(sobelX, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                it.pauseAndShowStep("2.5단계 (1/3): 수직 엣지(문자/세로그릴) 추출 및 이진화", debugBmp)
                tempRgb.release()
            }

            // 2. 가로 노이즈(헤드라이트 등)를 배제하기 위해 폭 제한
            val searchWidth = max(50.0, lineLen * 1.0).toInt()
            val searchLeft = (roiCx - searchWidth / 2.0).toInt().coerceIn(0, sobelX.cols() - 1)
            val searchRight = (roiCx + searchWidth / 2.0).toInt().coerceIn(1, sobelX.cols())
            
            val searchRect = Rect(searchLeft, 0, searchRight - searchLeft, sobelX.rows())
            sobelX.submat(searchRect).copyTo(searchMat)

            // 🛠️ [디버그 2.5-2] 좌우 탐색 범위 확인
            debugListener?.let {
                val debugMat = rotatedPaddedMat.clone()
                Imgproc.rectangle(debugMat, searchRect, Scalar(255.0, 0.0, 255.0, 255.0), 6) // 마젠타색 박스
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                it.pauseAndShowStep("2.5단계 (2/3): 스캔 영역 제한 (가로 노이즈 배제)", debugBmp)
                debugMat.release()
            }

            // 3. 가로줄 단위로 엣지 픽셀 합산 (고속 투영)
            Core.reduce(searchMat, projection, 1, Core.REDUCE_SUM, CvType.CV_32S)
            val profile = IntArray(projection.rows())
            projection.get(0, 0, profile)

            // 4. 중심 파생 탐색 (Center-Out Expansion)
            val startY = roiCy.toInt().coerceIn(0, profile.lastIndex)
            
            var centerSum = 0L
            var count = 0
            for(i in max(0, startY - 3)..min(profile.lastIndex, startY + 3)) {
                centerSum += profile[i]
                count++
            }
            val centerDensity = centerSum / count.toDouble()
            val threshold = centerDensity * 0.25 

            var detectedTop = startY
            for (y in startY downTo 0) {
                if (profile[y] < threshold) { detectedTop = y; break }
            }

            var detectedBottom = startY
            for (y in startY..profile.lastIndex) {
                if (profile[y] < threshold) { detectedBottom = y; break }
            }

            // 5. 기하학적 퓨즈 (예외 처리)
            var dynamicHeight = (detectedBottom - detectedTop).toDouble()
            val expectedGeometricHeight = lineLen / 4.7
            var usedFallback = false

            if (dynamicHeight > expectedGeometricHeight * 2.5 || dynamicHeight < expectedGeometricHeight * 0.4) {
                val fallbackHeight = expectedGeometricHeight * 2.0
                tightTop = max(0, (roiCy - fallbackHeight / 2.0).toInt())
                tightBottom = min(rotatedPaddedGray.rows(), (roiCy + fallbackHeight / 2.0).toInt())
                usedFallback = true
            } else {
                val padding = dynamicHeight * 0.2
                tightTop = max(0, (detectedTop - padding).toInt())
                tightBottom = min(rotatedPaddedGray.rows(), (detectedBottom + padding).toInt())
            }

            // 🛠️ [디버그 2.5-3] 최종 판단 결과 및 롤백 여부 확인
            debugListener?.let {
                val debugMat = rotatedPaddedMat.clone()
                
                // 탐색 시작점 (초록색 선)
                Imgproc.line(debugMat, Point(0.0, startY.toDouble()), Point(debugMat.cols().toDouble(), startY.toDouble()), Scalar(0.0, 255.0, 0.0, 255.0), 3)
                
                // 찾아낸 타이트 ROI의 상하단 (정상: 노란색, 롤백: 빨간색)
                val color = if(usedFallback) Scalar(255.0, 0.0, 0.0, 255.0) else Scalar(255.0, 255.0, 0.0, 255.0)
                Imgproc.line(debugMat, Point(0.0, tightTop.toDouble()), Point(debugMat.cols().toDouble(), tightTop.toDouble()), color, 5)
                Imgproc.line(debugMat, Point(0.0, tightBottom.toDouble()), Point(debugMat.cols().toDouble(), tightBottom.toDouble()), color, 5)
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                
                val msg = if(usedFallback) "2.5단계 (3/3): [경고] 비정상 감지 -> 기하학적 롤백 적용" 
                          else "2.5단계 (3/3): Edge Density 탐색 성공 -> 최종 높이 확정"
                it.pauseAndShowStep(msg, debugBmp)
                debugMat.release()
            }

        } finally {
            sobelX.release(); searchMat.release(); projection.release()
        }

        // =====================================================================
        // [3단계] 최종 타이트 ROI 생성
        // =====================================================================
        val tightWidth = max(100.0, lineLen * 1.1)
        val tightLeft = (roiCx - tightWidth / 2.0).toInt().coerceIn(0, rotatedPaddedGray.cols() - 1)
        val tightRight = (roiCx + tightWidth / 2.0).toInt().coerceIn(1, rotatedPaddedGray.cols())
        
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
            it.pauseAndShowStep("3단계: Edge Density가 반영된 최종 타이트 크롭 확인", debugBmp)
            debugMat.release()
        }

        val roiEdge = Mat(); val combinedEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>(); val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            // [4단계] CLAHE 적용
            val clahe = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
            clahe.apply(tightGray, tightGray)
            
            // [5단계] Canny Edge
            val meanVal = Core.mean(tightGray).`val`[0]
            Imgproc.Canny(tightGray, roiEdge, max(0.0, 0.33 * meanVal), min(255.0, 1.33 * meanVal))
            
            // [6단계] 모폴로지 닫기
            val kernelLen = (safeTightRect.width * 0.08).toInt().coerceIn(10, 60)
            val kernel0 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelLen.toDouble(), 2.0))
            Imgproc.morphologyEx(roiEdge, combinedEdge, Imgproc.MORPH_CLOSE, kernel0)
            kernel0.release()

            // [7단계] 다각형 윤곽선 추려내기
            Imgproc.findContours(combinedEdge, roiContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            
            val validPolygons = ArrayList<MatOfPoint2f>()
            for (contour in roiContours) {
                val approx = extractRobustPolygon(contour) ?: continue
                validPolygons.add(approx)
            }

            // [8단계] 깐깐한 형태 검사
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

            // [9단계] 이중 역변환 (타이트 ROI -> 안전 영역 -> 글로벌 원본)
            if (bestApprox2f != null) {
                val invRotMat = Mat()
                Imgproc.invertAffineTransform(rotMat, invRotMat)

                val rotatedPoints = bestApprox2f!!.toArray().map { Point(it.x + safeTightRect.x, it.y + safeTightRect.y) }.toTypedArray()
                val srcMat = MatOfPoint2f(*rotatedPoints)
                val dstMat = MatOfPoint2f()

                Core.transform(srcMat, dstMat, invRotMat)

                resultPoints = dstMat.toArray().map { 
                    ImmutablePoint((it.x + paddedRect.x).toFloat(), (it.y + paddedRect.y).toFloat()) 
                }

                invRotMat.release(); srcMat.release(); dstMat.release()
            }

            validPolygons.forEach { it.release() }
            bestApprox2f?.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
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
