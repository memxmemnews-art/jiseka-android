package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

// 💡 주의: ImmutablePoint와 CandidatePolygon data class는 이미 다른 곳에 있으므로 여기서는 제거했습니다.

object PlateDetectionEngine {

    /**
     * 1. 전체 이미지 대상: 1차 다각형 후보군 사전 계산
     */
    fun precalculateGeometryCandidates(fullBitmap: Bitmap): List<CandidatePolygon> {
        val candidates = mutableListOf<CandidatePolygon>()
        val mat = Mat(); val grayMat = Mat(); val bilateralMat = Mat()
        val tophatMat = Mat(); val edgeMat = Mat(); val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        var topHatKernel: Mat? = null; var morphKernel: Mat? = null
        var clahe: org.opencv.imgproc.CLAHE? = null

        try {
            Utils.bitmapToMat(fullBitmap, mat)
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY)
            
            clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(grayMat, grayMat)
            
            Imgproc.bilateralFilter(grayMat, bilateralMat, 5, 20.0, 20.0)
            
            topHatKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 9.0))
            Imgproc.morphologyEx(bilateralMat, tophatMat, Imgproc.MORPH_TOPHAT, topHatKernel)
            
            val meanVal = Core.mean(tophatMat).`val`[0]
            val sigma = 0.33
            val lowerThresh = max(0.0, (1.0 - sigma) * meanVal)
            val upperThresh = min(255.0, (1.0 + sigma) * meanVal)
            Imgproc.Canny(tophatMat, edgeMat, lowerThresh, upperThresh)
            
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 3.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val minPlateHeight = fullBitmap.height * 0.025

            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                if (rect.height < minPlateHeight) continue
                
                val approxPoints = extractRobustQuadrilateral(contour) ?: continue
                
                val pointArray = approxPoints.toArray()
                if (!isValidLicensePlateGeometry(pointArray)) {
                    approxPoints.release()
                    continue
                }

                val refinedPoints = applySubPixelCorrection(grayMat, approxPoints)
                approxPoints.release()

                val immutablePoints = refinedPoints.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
                val xCoords = immutablePoints.map { it.x }
                val yCoords = immutablePoints.map { it.y }
                
                val bounds = RectF(
                    xCoords.minOrNull() ?: 0f,
                    yCoords.minOrNull() ?: 0f,
                    xCoords.maxOrNull() ?: 0f,
                    yCoords.maxOrNull() ?: 0f
                )
                
                candidates.add(CandidatePolygon(immutablePoints, bounds))
            }
            return candidates
        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        } finally {
            mat.release(); grayMat.release(); bilateralMat.release()
            tophatMat.release(); edgeMat.release(); hierarchy.release()
            contours.forEach { it.release() }
            topHatKernel?.release(); morphKernel?.release()
            clahe?.collectGarbage()
        }
    }

    /**
     * 2. 관심 영역(ROI) 대상: 십자선 정밀 추적 및 롱프레스(Dwell) 대응 복원
     */
    fun refineAnchoredPolygon(fullBitmap: Bitmap, anchorPolygon: List<ImmutablePoint>, targetLevel: Int): List<ImmutablePoint>? {
        val padding = if (targetLevel == 1) 40f else 20f 
        
        val minX = (anchorPolygon.minOfOrNull { it.x } ?: 0f) - padding
        val minY = (anchorPolygon.minOfOrNull { it.y } ?: 0f) - padding
        val maxX = (anchorPolygon.maxOfOrNull { it.x } ?: 0f) + padding
        val maxY = (anchorPolygon.maxOfOrNull { it.y } ?: 0f) + padding

        val safeRect = android.graphics.Rect(
            max(0, minX.toInt()), max(0, minY.toInt()),
            min(fullBitmap.width, maxX.toInt()), min(fullBitmap.height, maxY.toInt())
        )
        
        if (safeRect.width() <= 1 || safeRect.height() <= 1) return null

        val roiBitmap = Bitmap.createBitmap(fullBitmap, safeRect.left, safeRect.top, safeRect.width(), safeRect.height())

        val roiMat = Mat(); val roiGray = Mat(); val roiEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>()
        var bestGlobalPoints: List<ImmutablePoint>? = null
        
        try {
            Utils.bitmapToMat(roiBitmap, roiMat)
            Imgproc.cvtColor(roiMat, roiGray, Imgproc.COLOR_RGBA2GRAY)
            
            val clahe = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
            clahe.apply(roiGray, roiGray)
            
            val meanVal = Core.mean(roiGray).`val`[0]
            val sigma = if (targetLevel == 1) 0.5 else 0.33 
            val lowerThresh = max(0.0, (1.0 - sigma) * meanVal)
            val upperThresh = min(255.0, (1.0 + sigma) * meanVal)
            
            Imgproc.Canny(roiGray, roiEdge, lowerThresh, upperThresh)
            
            val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(roiEdge, roiEdge, Imgproc.MORPH_CLOSE, morphKernel)
            
            Imgproc.findContours(roiEdge, roiContours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestApprox2f: MatOfPoint2f? = null
            var maxArea = -1.0

            for (contour in roiContours) {
                val area = Imgproc.contourArea(contour)
                if (area < 500) continue 

                val approx = extractRobustQuadrilateral(contour)
                if (approx != null) {
                    if (isValidLicensePlateGeometry(approx.toArray()) && area > maxArea) {
                        maxArea = area
                        bestApprox2f?.release()
                        bestApprox2f = approx
                    } else {
                        approx.release()
                    }
                }
            }

            if (bestApprox2f == null && targetLevel == 1) {
                val houghPoints = fallbackToHoughIntersections(roiEdge)
                if (houghPoints != null && isValidLicensePlateGeometry(houghPoints.toTypedArray())) {
                    bestApprox2f = MatOfPoint2f(*houghPoints.toTypedArray())
                }
            }

            if (bestApprox2f != null) {
                val refinedLocalPoints = applySubPixelCorrection(roiGray, bestApprox2f!!)
                bestApprox2f!!.release()
                
                bestGlobalPoints = refinedLocalPoints.map { 
                    ImmutablePoint((it.x + safeRect.left).toFloat(), (it.y + safeRect.top).toFloat()) 
                }
            }
            
            morphKernel.release()
            clahe.collectGarbage()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            roiMat.release(); roiGray.release(); roiEdge.release()
            roiContours.forEach { it.release() }
            roiBitmap.recycle()
        }

        return bestGlobalPoints
    }

    // =========================================================================
    // 💡 아래는 모두 헬퍼 함수입니다. 반드시 object 내부에 위치해야 합니다.
    // =========================================================================

    private fun extractRobustQuadrilateral(contour: MatOfPoint): MatOfPoint2f? {
        val contour2f = MatOfPoint2f(*contour.toArray())
        val perimeter = Imgproc.arcLength(contour2f, true)
        
        var minEps = 0.0
        var maxEps = perimeter * 0.1 
        var bestApprox: MatOfPoint2f? = null

        val approx2f = MatOfPoint2f()
        
        for (i in 0 until 10) {
            val eps = (minEps + maxEps) / 2.0
            Imgproc.approxPolyDP(contour2f, approx2f, eps, true)
            
            val pointsCount = approx2f.rows()
            
            if (pointsCount == 4) {
                bestApprox = MatOfPoint2f(*approx2f.toArray())
                break
            } else if (pointsCount > 4) {
                minEps = eps
            } else {
                maxEps = eps
            }
        }
        
        approx2f.release()
        contour2f.release()
        return bestApprox
    }

    private fun isValidLicensePlateGeometry(points: Array<Point>): Boolean {
        if (points.size != 4) return false

        val matOfPoint = MatOfPoint(*points)
        val isConvex = Imgproc.isContourConvex(matOfPoint)
        matOfPoint.release()
        
        if (!isConvex) return false

        var minAngleDeg = 180.0
        var maxAngleDeg = 0.0
        val edgeLengths = DoubleArray(4)

        for (i in 0 until 4) {
            val pt1 = points[i]
            val pt2 = points[(i + 1) % 4]
            val pt0 = points[(i + 3) % 4] 

            val dx = pt1.x - pt2.x
            val dy = pt1.y - pt2.y
            edgeLengths[i] = hypot(dx, dy)

            val v1x = pt0.x - pt1.x
            val v1y = pt0.y - pt1.y
            val v2x = pt2.x - pt1.x
            val v2y = pt2.y - pt1.y

            val dot = v1x * v2x + v1y * v2y
            val norm = hypot(v1x, v1y) * hypot(v2x, v2y)
            
            if (norm > 1e-6) {
                var cosTheta = dot / norm
                cosTheta = cosTheta.coerceIn(-1.0, 1.0) 
                val angle = Math.toDegrees(acos(cosTheta))

                minAngleDeg = min(minAngleDeg, angle)
                maxAngleDeg = max(maxAngleDeg, angle)
            }
        }

        if (minAngleDeg < 45.0 || maxAngleDeg > 135.0) return false

        val widthApprox = (edgeLengths[0] + edgeLengths[2]) / 2.0
        val heightApprox = (edgeLengths[1] + edgeLengths[3]) / 2.0

        if (widthApprox < 1e-6 || heightApprox < 1e-6) return false
        
        val ratio = if (widthApprox > heightApprox) widthApprox / heightApprox else heightApprox / widthApprox
        if (ratio < 1.5 || ratio > 6.0) return false

        return true
    }

    private fun applySubPixelCorrection(grayMat: Mat, approx2f: MatOfPoint2f): List<Point> {
        val points = approx2f.toArray()
        val cornersMat = MatOfPoint2f(*points)
        
        val winSize = Size(5.0, 5.0)
        val zeroZone = Size(-1.0, -1.0)
        val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.001)

        Imgproc.cornerSubPix(grayMat, cornersMat, winSize, zeroZone, criteria)

        val refinedArray = cornersMat.toArray().toList()
        cornersMat.release()
        
        return refinedArray
    }

    private fun fallbackToHoughIntersections(edgeMat: Mat): List<Point>? {
        val lines = Mat()
        Imgproc.HoughLines(edgeMat, lines, 1.0, Math.PI / 180.0, 40)

        if (lines.rows() < 4) {
            lines.release()
            return null 
        }

        val horizontals = mutableListOf<DoubleArray>()
        val verticals = mutableListOf<DoubleArray>()   

        for (i in 0 until lines.rows()) {
            val vec = lines.get(i, 0)
            val angleDeg = Math.toDegrees(vec[1])
            
            if (angleDeg in 60.0..120.0) {
                horizontals.add(vec)
            } else if (angleDeg < 30.0 || angleDeg > 150.0) {
                verticals.add(vec)
            }
        }
        lines.release()

        if (horizontals.size < 2 || verticals.size < 2) return null

        horizontals.sortBy { abs(it[0]) } 
        val topEdge = horizontals.first()
        val bottomEdge = horizontals.last()

        verticals.sortBy { abs(it[0]) }
        val leftEdge = verticals.first()
        val rightEdge = verticals.last()

        val topLeft = calculateIntersection(topEdge, leftEdge) ?: return null
        val topRight = calculateIntersection(topEdge, rightEdge) ?: return null
        val bottomLeft = calculateIntersection(bottomEdge, leftEdge) ?: return null
        val bottomRight = calculateIntersection(bottomEdge, rightEdge) ?: return null

        return listOf(topLeft, topRight, bottomRight, bottomLeft)
    }

    private fun calculateIntersection(line1: DoubleArray, line2: DoubleArray): Point? {
        val a1 = cos(line1[1]); val b1 = sin(line1[1]); val c1 = line1[0]
        val a2 = cos(line2[1]); val b2 = sin(line2[1]); val c2 = line2[0]

        val det = a1 * b2 - a2 * b1
        if (abs(det) < 0.0001) return null

        val x = (c1 * b2 - c2 * b1) / det
        val y = (a1 * c2 - a2 * c1) / det

        return Point(x, y)
    }
}
