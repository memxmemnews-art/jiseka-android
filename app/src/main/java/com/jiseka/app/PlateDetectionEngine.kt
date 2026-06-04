package com.jiseka.app

import android.graphics.Bitmap
import android.os.SystemClock
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    fun rescuePlateFromCrosshair(fullBitmap: Bitmap, crosshairX: Float, crosshairY: Float): List<ImmutablePoint>? {
        val imageArea = fullBitmap.width * fullBitmap.height.toDouble()

        // 🌟 JNI 병목 원천 차단: 루프 외부에서 전체 이미지를 딱 1번만 흑백 Mat으로 변환
        val fullMat = Mat()
        val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        fun attemptSearch(
            roiPercent: Double, 
            targetRatio: Double, 
            claheLim: Double, 
            cannyLow: Double, 
            cannyUp: Double, 
            kernelRatio: Double, 
            useMultiAngle: Boolean
        ): List<ImmutablePoint>? {
            
            val targetRoiArea = imageArea * roiPercent
            val roiHeight = Math.sqrt(targetRoiArea / targetRatio)
            val roiWidth = roiHeight * targetRatio

            // 안전한 정수형 경계 박스 (크래시 방지)
            var left = floor(crosshairX - roiWidth / 2.0).toInt()
            var top = floor(crosshairY - roiHeight / 2.0).toInt()
            var right = ceil(crosshairX + roiWidth / 2.0).toInt()
            var bottom = ceil(crosshairY + roiHeight / 2.0).toInt()

            left = max(0, left)
            top = max(0, top)
            right = min(fullBitmap.width, right)
            bottom = min(fullBitmap.height, bottom)

            if (right - left <= 10 || bottom - top <= 10) return null

            val safeRect = org.opencv.core.Rect(left, top, right - left, bottom - top)
            
            val roiImageArea = safeRect.width * safeRect.height.toDouble()
            val roiCenter = Point(safeRect.width / 2.0, safeRect.height / 2.0)

            // 🌟 Bitmap 재생성 제거: 전체 Mat에서 submat으로 즉시 획득
            val roiGray = Mat()
            fullGray.submat(safeRect).copyTo(roiGray)

            val roiEdge = Mat()
            val combinedEdge = Mat(); val closed0 = Mat(); val closed15 = Mat(); val closedMinus15 = Mat()
            var kernel0: Mat? = null; var kernel15: Mat? = null; var kernelMinus15: Mat? = null
            val roiContours = ArrayList<MatOfPoint>()
            var bestApprox2f: MatOfPoint2f? = null
            var resultPoints: List<ImmutablePoint>? = null

            val clahe = Imgproc.createCLAHE(claheLim, Size(4.0, 4.0))
            val hierarchy = Mat()

            try {
                clahe.apply(roiGray, roiGray)

                val meanVal = Core.mean(roiGray).`val`[0]
                Imgproc.Canny(roiGray, roiEdge, max(0.0, cannyLow * meanVal), min(255.0, cannyUp * meanVal))

                val kernelLen = max(10.0, min(60.0, safeRect.width * kernelRatio)).toInt()
                kernel0 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelLen.toDouble(), 2.0))

                if (useMultiAngle) {
                    kernel15 = createAngledKernel(kernelLen, 15.0)
                    kernelMinus15 = createAngledKernel(kernelLen, -15.0)

                    Imgproc.morphologyEx(roiEdge, closed0, Imgproc.MORPH_CLOSE, kernel0)
                    Imgproc.morphologyEx(roiEdge, closed15, Imgproc.MORPH_CLOSE, kernel15)
                    Imgproc.morphologyEx(roiEdge, closedMinus15, Imgproc.MORPH_CLOSE, kernelMinus15)

                    Core.bitwise_or(closed0, closed15, combinedEdge)
                    Core.bitwise_or(combinedEdge, closedMinus15, combinedEdge)
                } else {
                    Imgproc.morphologyEx(roiEdge, combinedEdge, Imgproc.MORPH_CLOSE, kernel0)
                }

                Imgproc.findContours(combinedEdge, roiContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

                var maxArea = -1.0

                for (contour in roiContours) {
                    val contourArea = Imgproc.contourArea(contour)
                    val approx = extractRobustPolygon(contour) ?: continue
                    val candidateArray = approx.toArray()

                    if (isValidRescueGeometry(contourArea, candidateArray, roiImageArea)) {
                        val candCx = candidateArray.map { it.x }.average()
                        val candCy = candidateArray.map { it.y }.average()
                        val distToCenter = hypot(roiCenter.x - candCx, roiCenter.y - candCy)

                        // 🌟 에러 픽스: 없어진 roiBitmap 대신 safeRect.width 사용
                        if (distToCenter < (safeRect.width / 3.0) && contourArea > maxArea) {
                            maxArea = contourArea
                            bestApprox2f?.release()
                            bestApprox2f = approx
                            continue
                        }
                    }
                    approx.release()
                }

                if (bestApprox2f != null) {
                    resultPoints = bestApprox2f.toArray().map { 
                        ImmutablePoint((it.x + safeRect.x).toFloat(), (it.y + safeRect.y).toFloat()) 
                    }
                    bestApprox2f.release()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                roiGray.release(); roiEdge.release()
                combinedEdge.release(); closed0.release(); closed15.release(); closedMinus15.release()
                kernel0?.release(); kernel15?.release(); kernelMinus15?.release()
                roiContours.forEach { it.release() }
                
                hierarchy.release()
                // 🌟 에러 픽스: OpenCV Java의 CLAHE는 release()가 없으므로 해당 라인 삭제
            }
            return resultPoints
        }

        try {
            // =================================================================================
            // [Phase 1] 가볍고 빠른 순차적 탐색 (정면 비율 4.0, 최대 3% 고정)
            // =================================================================================
            val stepRois = listOf(0.01, 0.02, 0.03)
            for (roi in stepRois) {
                val basicResult = attemptSearch(
                    roiPercent = roi, 
                    targetRatio = 4.0, 
                    claheLim = 2.0, 
                    cannyLow = 0.33, 
                    cannyUp = 1.33, 
                    kernelRatio = 0.08, 
                    useMultiAngle = false
                )
                if (basicResult != null) return basicResult
            }

            // =================================================================================
            // [Phase 2] 정밀 탐색 모드 (1.5초 타임아웃 + 최대 8회 반복)
            // =================================================================================
            val startTime = SystemClock.elapsedRealtime()
            var iteration = 0

            var claheLim = 3.0
            var cannyLow = 0.25
            var cannyUp = 1.50
            var kernelRatio = 0.08
            
            val fixedRoiPercent = 0.03 
            var currentRatio = 3.5 

            // 🌟 ANR 방지: 반복 횟수 및 시스템 시간 제한 결합
            while (iteration < 8 && SystemClock.elapsedRealtime() - startTime < 1500) {
                val desperateResult = attemptSearch(
                    roiPercent = fixedRoiPercent, 
                    targetRatio = currentRatio, 
                    claheLim = claheLim, 
                    cannyLow = cannyLow, 
                    cannyUp = cannyUp, 
                    kernelRatio = kernelRatio, 
                    useMultiAngle = true
                )
                if (desperateResult != null) return desperateResult

                claheLim += 1.5                             
                cannyLow = max(0.05, cannyLow - 0.04)       
                cannyUp = min(2.5, cannyUp + 0.15)           
                
                kernelRatio += 0.03
                if (kernelRatio > 0.16) kernelRatio = 0.06 
                
                currentRatio -= 0.5 
                if (currentRatio < 2.5) { 
                    currentRatio = 3.5 
                }
                
                iteration++
            }

            return null
        } finally {
            // 루프 종료 후 한 번 생성했던 대형 Mat 해제
            fullMat.release()
            fullGray.release()
        }
    }

    private fun createAngledKernel(length: Int, angleDegree: Double): Mat {
        val kernel = Mat.zeros(length, length, CvType.CV_8UC1)
        val radian = Math.toRadians(angleDegree)
        val cx = length / 2.0
        val cy = length / 2.0
        val halfLen = length / 2.0
        val pt1 = Point(cx - halfLen * Math.cos(radian), cy - halfLen * Math.sin(radian))
        val pt2 = Point(cx + halfLen * Math.cos(radian), cy + halfLen * Math.sin(radian))
        Imgproc.line(kernel, pt1, pt2, Scalar(255.0), 2)
        return kernel
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
            val sortedHull = sortPolygonPoints(hullPoints)
            return MatOfPoint2f(*sortedHull.toTypedArray())
        }
        return null
    }

    private fun isValidRescueGeometry(originalContourArea: Double, hullPoints: Array<Point>, roiArea: Double): Boolean {
        val hullMat = MatOfPoint(*hullPoints)
        val hullArea = Imgproc.contourArea(hullMat)
        
        val solidity = originalContourArea / hullArea
        if (solidity < 0.85) { hullMat.release(); return false }

        val normalizedArea = originalContourArea / roiArea
        if (normalizedArea < 0.03 || normalizedArea > 0.80) { hullMat.release(); return false }

        val hullMat2f = MatOfPoint2f(*hullPoints)
        val minAreaRect = Imgproc.minAreaRect(hullMat2f)
        
        val extent = originalContourArea / (Imgproc.boundingRect(hullMat).area().toDouble())
        if (extent < 0.25) { hullMat.release(); hullMat2f.release(); return false }
        
        val rectArea = minAreaRect.size.width * minAreaRect.size.height
        val rectangularity = if (rectArea > 0) originalContourArea / rectArea else 0.0
        if (rectangularity < 0.35) { hullMat.release(); hullMat2f.release(); return false }

        var w = minAreaRect.size.width; var h = minAreaRect.size.height
        if (w < h) { val temp = w; w = h; h = temp }
        if (h < 1e-6 || w / h !in 2.5..6.0) { hullMat.release(); hullMat2f.release(); return false }

        hullMat.release(); hullMat2f.release()
        return true
    }

    private fun sortPolygonPoints(points: List<Point>): List<Point> {
        if (points.size < 3) return points
        val cx = points.map { it.x }.average()
        val cy = points.map { it.y }.average()
        var sorted = points.sortedBy { Math.atan2(it.y - cy, it.x - cx) }
        var area = 0.0
        val n = sorted.size
        for (i in 0 until n) {
            val p1 = sorted[i]; val p2 = sorted[(i + 1) % n]
            area += (p1.x * p2.y - p2.x * p1.y)
        }
        if (area < 0) sorted = sorted.reversed()
        return sorted
    }
}
