package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    private val defaultClahe: org.opencv.imgproc.CLAHE by lazy {
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    }
    
    private val roiClahe: org.opencv.imgproc.CLAHE by lazy {
        Imgproc.createCLAHE(3.0, Size(4.0, 4.0)) 
    }

    fun precalculateGeometryCandidates(fullBitmap: Bitmap): List<CandidatePolygon> {
        val candidates = mutableListOf<CandidatePolygon>()
        val mat = Mat(); val grayMat = Mat(); val bilateralMat = Mat()
        val edgeMat = Mat(); val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        var morphKernel: Mat? = null

        try {
            Utils.bitmapToMat(fullBitmap, mat)
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY)
            
            defaultClahe.apply(grayMat, grayMat)
            Imgproc.GaussianBlur(grayMat, bilateralMat, Size(7.0, 7.0), 0.0)
            
            val meanVal = Core.mean(bilateralMat).`val`[0]
            val lowerThresh = max(0.0, 0.5 * meanVal)
            val upperThresh = min(255.0, 1.5 * meanVal)
            Imgproc.Canny(bilateralMat, edgeMat, lowerThresh, upperThresh)
            
            val kernelWidth = (fullBitmap.width * 0.01).coerceIn(15.0, 50.0)
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelWidth, 2.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            val imageArea = fullBitmap.width * fullBitmap.height.toDouble()
            val imageHeight = fullBitmap.height.toDouble()

            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                if (rect.x <= 2 || rect.y <= 2 || rect.x + rect.width >= fullBitmap.width - 2 || rect.y + rect.height >= fullBitmap.height - 2) continue

                val contourArea = Imgproc.contourArea(contour)
                val approxPoints = extractRobustPolygon(contour) ?: continue
                val pointArray = approxPoints.toArray()
                
                if (!isValidLicensePlateGeometry(contourArea, pointArray, imageArea, imageHeight)) {
                    approxPoints.release()
                    continue
                }

                val immutablePoints = pointArray.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
                approxPoints.release()

                val xCoords = immutablePoints.map { it.x }
                val yCoords = immutablePoints.map { it.y }
                
                val bounds = RectF(
                    xCoords.minOrNull() ?: 0f, yCoords.minOrNull() ?: 0f,
                    xCoords.maxOrNull() ?: 0f, yCoords.maxOrNull() ?: 0f
                )
                
                candidates.add(CandidatePolygon(immutablePoints, bounds))
            }
            return candidates
        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        } finally {
            mat.release(); grayMat.release(); bilateralMat.release()
            edgeMat.release(); hierarchy.release()
            contours.forEach { it.release() }
            morphKernel?.release() 
        }
    }

    fun rescuePlateFromCrosshair(fullBitmap: Bitmap, crosshairX: Float, crosshairY: Float): List<ImmutablePoint>? {
        val imageArea = fullBitmap.width * fullBitmap.height.toDouble()
        val targetRoiArea = imageArea * 0.03
        val roiHeight = Math.sqrt(targetRoiArea / 3.0)
        val roiWidth = roiHeight * 3.0

        var minX = crosshairX - (roiWidth / 2.0).toFloat()
        var minY = crosshairY - (roiHeight / 2.0).toFloat()
        var maxX = crosshairX + (roiWidth / 2.0).toFloat()
        var maxY = crosshairY + (roiHeight / 2.0).toFloat()

        minX = max(0f, minX); minY = max(0f, minY)
        maxX = min(fullBitmap.width.toFloat(), maxX); maxY = min(fullBitmap.height.toFloat(), maxY)

        val safeRect = android.graphics.Rect(minX.toInt(), minY.toInt(), maxX.toInt(), maxY.toInt())
        if (safeRect.width() <= 10 || safeRect.height() <= 10) return null

        val roiBitmap = Bitmap.createBitmap(fullBitmap, safeRect.left, safeRect.top, safeRect.width(), safeRect.height())
        val roiImageArea = roiBitmap.width * roiBitmap.height.toDouble()
        
        val roiMat = Mat(); val roiGray = Mat(); val roiEdge = Mat()
        val closed0 = Mat(); val closed15 = Mat(); val closedMinus15 = Mat(); val combinedEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>()
        
        var bestGlobalPoints: List<ImmutablePoint>? = null
        var kernel0: Mat? = null; var kernel15: Mat? = null; var kernelMinus15: Mat? = null
        
        try {
            Utils.bitmapToMat(roiBitmap, roiMat)
            Imgproc.cvtColor(roiMat, roiGray, Imgproc.COLOR_RGBA2GRAY)
            roiClahe.apply(roiGray, roiGray)
            
            val meanVal = Core.mean(roiGray).`val`[0]
            Imgproc.Canny(roiGray, roiEdge, max(0.0, 0.33 * meanVal), min(255.0, 1.33 * meanVal))
            
            val kernelLen = (safeRect.width() * 0.08).coerceIn(15.0, 50.0).toInt()
            kernel0 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelLen.toDouble(), 2.0))
            kernel15 = createAngledKernel(kernelLen, 15.0)
            kernelMinus15 = createAngledKernel(kernelLen, -15.0)

            Imgproc.morphologyEx(roiEdge, closed0, Imgproc.MORPH_CLOSE, kernel0)
            Imgproc.morphologyEx(roiEdge, closed15, Imgproc.MORPH_CLOSE, kernel15)
            Imgproc.morphologyEx(roiEdge, closedMinus15, Imgproc.MORPH_CLOSE, kernelMinus15)

            Core.bitwise_or(closed0, closed15, combinedEdge)
            Core.bitwise_or(combinedEdge, closedMinus15, combinedEdge)
            
            Imgproc.findContours(combinedEdge, roiContours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestApprox2f: MatOfPoint2f? = null
            var maxArea = -1.0
            val roiCenter = Point(roiBitmap.width / 2.0, roiBitmap.height / 2.0)

            for (contour in roiContours) {
                val contourArea = Imgproc.contourArea(contour)
                val approx = extractRobustPolygon(contour) ?: continue
                val candidateArray = approx.toArray()
                
                if (isValidRescueGeometry(contourArea, candidateArray, roiImageArea)) {
                    val candCx = candidateArray.map { it.x }.average()
                    val candCy = candidateArray.map { it.y }.average()
                    val distToCenter = hypot(roiCenter.x - candCx, roiCenter.y - candCy)
                    
                    if (distToCenter < (roiBitmap.width / 3.0) && contourArea > maxArea) {
                        maxArea = contourArea
                        bestApprox2f?.release()
                        bestApprox2f = approx
                        continue
                    }
                }
                approx.release()
            }

            bestApprox2f?.let { approx ->
                // 🌟 다각형을 강제로 4각 박스로 변환하지 않고 유지합니다.
                val polygonPoints = approx.toArray()

                bestGlobalPoints = polygonPoints.map { 
                    ImmutablePoint((it.x + safeRect.left).toFloat(), (it.y + safeRect.top).toFloat()) 
                }
                approx.release()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            roiMat.release(); roiGray.release(); roiEdge.release()
            closed0.release(); closed15.release(); closedMinus15.release(); combinedEdge.release()
            kernel0?.release(); kernel15?.release(); kernelMinus15?.release()
            roiContours.forEach { it.release() }; roiBitmap.recycle()
        }
        return bestGlobalPoints
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

    private fun isValidLicensePlateGeometry(
        originalContourArea: Double, hullPoints: Array<Point>, 
        referenceImageArea: Double, referenceImageHeight: Double
    ): Boolean {
        if (hullPoints.size !in 4..10) return false
        val hullMat = MatOfPoint(*hullPoints)
        val hullArea = Imgproc.contourArea(hullMat)
        
        val solidity = originalContourArea / hullArea
        if (solidity < 0.80) { hullMat.release(); return false }

        val normalizedArea = originalContourArea / referenceImageArea
        if (normalizedArea < 0.002 || normalizedArea > 0.03) { hullMat.release(); return false }
        
        val hullMat2f = MatOfPoint2f(*hullPoints)
        val minAreaRect = Imgproc.minAreaRect(hullMat2f)
        var w = minAreaRect.size.width; var h = minAreaRect.size.height
        if (w < h) { val temp = w; w = h; h = temp }
        
        if (h < referenceImageHeight * 0.01) { hullMat.release(); hullMat2f.release(); return false }
        
        val extent = originalContourArea / (Imgproc.boundingRect(hullMat).area())
        if (extent < 0.40) { hullMat.release(); hullMat2f.release(); return false }
        
        val rectArea = w * h
        val rectangularity = if (rectArea > 0) originalContourArea / rectArea else 0.0
        if (rectangularity < 0.55) { hullMat.release(); hullMat2f.release(); return false }

        if (h < 1e-6) { hullMat.release(); hullMat2f.release(); return false }
        
        // 🌟 원근 왜곡을 수용하기 위해 하한선 2.5로 완화
        val ratio = w / h
        if (ratio < 2.5 || ratio > 6.0) { hullMat.release(); hullMat2f.release(); return false }

        hullMat.release(); hullMat2f.release()
        return true
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

        // 🌟 원근 왜곡을 수용하기 위해 구조대 모드에서도 하한선 2.5로 완화
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

    fun calculatePolygonScore(points: List<ImmutablePoint>, imageArea: Double): Double {
        if (points.size < 4) return 0.0
        val pts = points.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()
        val matOfPoint = MatOfPoint(*pts)
        val hullIndices = MatOfInt()
        Imgproc.convexHull(matOfPoint, hullIndices)
        
        val hullPoints = mutableListOf<Point>()
        for (index in hullIndices.toArray()) hullPoints.add(pts[index])
        val hullMat = MatOfPoint(*hullPoints.toTypedArray())
        
        val polygonArea = Imgproc.contourArea(matOfPoint)
        val hullArea = Imgproc.contourArea(hullMat)
        val boundingRect = Imgproc.boundingRect(hullMat)
        
        val solidity = if (hullArea > 0) polygonArea / hullArea else 0.0
        val extent = if (boundingRect.area() > 0) polygonArea / boundingRect.area() else 0.0
        
        val minAreaRect = Imgproc.minAreaRect(MatOfPoint2f(*pts))
        var w = minAreaRect.size.width; var h = minAreaRect.size.height
        if (w < h) { val temp = w; w = h; h = temp }
        
        val rectArea = w * h
        val rectangularity = if (rectArea > 0) polygonArea / rectArea else 0.0
        
        val normalizedAreaScore = min((polygonArea / imageArea) / 0.25, 1.0)
        val score = (solidity * 0.30) + (rectangularity * 0.25) + (extent * 0.20) + (normalizedAreaScore * 0.25)
        
        hullIndices.release(); hullMat.release(); matOfPoint.release()
        return score
    }
}
