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
        Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
    }

    private val morphCloseKernelRoi: Mat by lazy {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
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
            
            // 🌟 그릴 떡짐 방지를 위한 가로 커널(Horizontal Kernel) 유지
            val kernelWidth = (fullBitmap.width * 0.01).coerceIn(15.0, 50.0)
            val kernelHeight = 2.0 
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelWidth, kernelHeight))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            val imageArea = fullBitmap.width * fullBitmap.height.toDouble()
            val imageHeight = fullBitmap.height.toDouble()

            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                
                if (rect.x <= 2 || rect.y <= 2 || rect.x + rect.width >= fullBitmap.width - 2 || rect.y + rect.height >= fullBitmap.height - 2) {
                    continue
                }

                val contourArea = Imgproc.contourArea(contour)
                val approxPoints = extractRobustPolygon(contour) ?: continue
                val pointArray = approxPoints.toArray()
                
                if (!isValidLicensePlateGeometry(contourArea, pointArray, imageArea, imageHeight, isRoi = false)) {
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

        val anchorCx = anchorPolygon.map { it.x }.average()
        val anchorCy = anchorPolygon.map { it.y }.average()
        val anchorMatOfPoint = MatOfPoint(*anchorPolygon.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        val anchorArea = Imgproc.contourArea(anchorMatOfPoint)
        val anchorMatOfPointRoi = MatOfPoint(*anchorPolygon.map { Point((it.x - safeRect.left).toDouble(), (it.y - safeRect.top).toDouble()) }.toTypedArray())

        val roiBitmap = Bitmap.createBitmap(fullBitmap, safeRect.left, safeRect.top, safeRect.width(), safeRect.height())
        val roiMat = Mat(); val roiGray = Mat(); val roiEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>()
        var bestGlobalPoints: List<ImmutablePoint>? = null
        
        val roiImageArea = roiBitmap.width * roiBitmap.height.toDouble()
        val roiImageHeight = roiBitmap.height.toDouble()
        
        try {
            Utils.bitmapToMat(roiBitmap, roiMat)
            Imgproc.cvtColor(roiMat, roiGray, Imgproc.COLOR_RGBA2GRAY)
            
            roiClahe.apply(roiGray, roiGray)
            
            val meanVal = Core.mean(roiGray).`val`[0]
            val sigma = if (targetLevel == 1) 0.5 else 0.33 
            val lowerThresh = max(0.0, (1.0 - sigma) * meanVal)
            val upperThresh = min(255.0, (1.0 + sigma) * meanVal)
            
            Imgproc.Canny(roiGray, roiEdge, lowerThresh, upperThresh)
            
            Imgproc.morphologyEx(roiEdge, roiEdge, Imgproc.MORPH_CLOSE, morphCloseKernelRoi)
            Imgproc.findContours(roiEdge, roiContours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            var bestApprox2f: MatOfPoint2f? = null
            var maxArea = -1.0

            for (contour in roiContours) {
                val contourArea = Imgproc.contourArea(contour)
                val approx = extractRobustPolygon(contour)
                
                if (approx != null) {
                    val candidateArray = approx.toArray()
                    val isValid = isValidLicensePlateGeometry(contourArea, candidateArray, roiImageArea, roiImageHeight, isRoi = true)
                    
                    if (isValid) {
                        val candidateMat = MatOfPoint(*candidateArray)
                        
                        val candGlobalCx = candidateArray.map { it.x + safeRect.left }.average()
                        val candGlobalCy = candidateArray.map { it.y + safeRect.top }.average()
                        val distance = hypot(anchorCx - candGlobalCx, anchorCy - candGlobalCy)
                        
                        val intersectPoints = MatOfPoint()
                        val intersectArea = if (Imgproc.intersectConvexConvex(anchorMatOfPointRoi, candidateMat, intersectPoints) > 0f) {
                            Imgproc.contourArea(intersectPoints)
                        } else 0.0
                        
                        val unionArea = anchorArea + contourArea - intersectArea
                        val iou = if (unionArea > 0) intersectArea / unionArea else 0.0
                        
                        intersectPoints.release()
                        candidateMat.release()

                        if (distance <= 50.0 && iou >= 0.4 && contourArea > maxArea) {
                            maxArea = contourArea
                            bestApprox2f?.release()
                            bestApprox2f = approx
                            continue
                        }
                    }
                    approx.release()
                }
            }

            bestApprox2f?.let { approx ->
                val pointsArray = approx.toArray()
                approx.release()
                
                bestGlobalPoints = pointsArray.map { 
                    ImmutablePoint((it.x + safeRect.left).toFloat(), (it.y + safeRect.top).toFloat()) 
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            roiMat.release(); roiGray.release(); roiEdge.release()
            roiContours.forEach { it.release() }; roiBitmap.recycle()
            anchorMatOfPoint.release(); anchorMatOfPointRoi.release()
        }
        return bestGlobalPoints
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
        val indicesArray = hullIndices.toArray()
        
        for (index in indicesArray) {
            hullPoints.add(approxPointsArray[index])
        }

        approx2f.release()
        contour2f.release()
        approxPointMat.release()
        hullIndices.release()

        // 🌟 꼭짓점 개수 10개 커트라인 유지
        if (hullPoints.size in 4..10) {
            val sortedHull = sortPolygonPoints(hullPoints)
            return MatOfPoint2f(*sortedHull.toTypedArray())
        }
        return null
    }

    private fun sortPolygonPoints(points: List<Point>): List<Point> {
        if (points.size < 3) return points
        
        val cx = points.map { it.x }.average()
        val cy = points.map { it.y }.average()

        var sorted = points.sortedBy { Math.atan2(it.y - cy, it.x - cx) }

        var area = 0.0
        val n = sorted.size
        for (i in 0 until n) {
            val p1 = sorted[i]
            val p2 = sorted[(i + 1) % n]
            area += (p1.x * p2.y - p2.x * p1.y)
        }

        if (area < 0) {
            sorted = sorted.reversed()
        }

        return sorted
    }

    fun calculatePolygonScore(points: List<ImmutablePoint>, imageArea: Double): Double {
        if (points.size < 4) return 0.0
        
        val pts = points.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()
        val matOfPoint = MatOfPoint(*pts)
        val matOfPoint2f = MatOfPoint2f(*pts)
        
        val hullIndices = MatOfInt()
        Imgproc.convexHull(matOfPoint, hullIndices)
        
        val hullPoints = mutableListOf<Point>()
        for (index in hullIndices.toArray()) {
            hullPoints.add(pts[index])
        }
        val hullMat = MatOfPoint(*hullPoints.toTypedArray())
        
        val polygonArea = Imgproc.contourArea(matOfPoint)
        val hullArea = Imgproc.contourArea(hullMat)
        val boundingRect = Imgproc.boundingRect(hullMat)
        
        val solidity = if (hullArea > 0) polygonArea / hullArea else 0.0
        val extent = if (boundingRect.area() > 0) hullArea / boundingRect.area() else 0.0
        
        val minAreaRect = Imgproc.minAreaRect(matOfPoint2f)
        var w = minAreaRect.size.width
        var h = minAreaRect.size.height
        if (w < h) { val temp = w; w = h; h = temp }
        val rectArea = w * h
        val rectangularity = if (rectArea > 0) polygonArea / rectArea else 0.0
        
        val normalizedArea = polygonArea / imageArea
        val normalizedAreaScore = min(normalizedArea / 0.25, 1.0)
        
        val score = (solidity * 0.30) + (rectangularity * 0.25) + (extent * 0.20) + (normalizedAreaScore * 0.25)
        
        hullIndices.release()
        hullMat.release()
        matOfPoint.release()
        matOfPoint2f.release()
        
        return score
    }

    private fun isValidLicensePlateGeometry(
        originalContourArea: Double,
        hullPoints: Array<Point>, 
        referenceImageArea: Double,
        referenceImageHeight: Double,
        isRoi: Boolean
    ): Boolean {
        if (hullPoints.size !in 4..10) return false

        val hullMat = MatOfPoint(*hullPoints)
        val hullArea = Imgproc.contourArea(hullMat)
        val boundingRect = Imgproc.boundingRect(hullMat)
        
        val normalizedArea = hullArea / referenceImageArea
        
        if (isRoi) {
            if (normalizedArea < 0.10) {
                hullMat.release()
                return false
            }
        } else {
            // 🌟 [수정] 면적 0.3% ~ 1.0% 로 정밀 타겟팅 설정
            if (normalizedArea < 0.003 || normalizedArea > 0.01) {
                hullMat.release()
                return false
            }
        }

        val hullMat2f = MatOfPoint2f(*hullPoints)
        val minAreaRect = Imgproc.minAreaRect(hullMat2f)
        var w = minAreaRect.size.width
        var h = minAreaRect.size.height
        if (w < h) { val temp = w; w = h; h = temp }
        
        if (!isRoi && h < referenceImageHeight * 0.01) {
            hullMat.release()
            hullMat2f.release()
            return false
        }

        val solidity = originalContourArea / hullArea
        if (solidity < 0.60) {
            hullMat.release()
            hullMat2f.release()
            return false
        }

        val extent = hullArea / (boundingRect.width * boundingRect.height).toDouble()
        if (extent < 0.40) {
            hullMat.release()
            hullMat2f.release()
            return false
        }
        
        val rectArea = w * h
        val rectangularity = if (rectArea > 0) hullArea / rectArea else 0.0
        
        hullMat.release()
        hullMat2f.release()

        // 🌟 직사각형 비율 55% 유지
        if (rectangularity < 0.55) return false

        if (h < 1e-6) return false
        val ratio = w / h
        if (ratio < 1.2 || ratio > 8.0) return false

        return true
    }
}
