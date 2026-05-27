package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    fun precalculateGeometryCandidates(fullBitmap: Bitmap): List<CandidatePolygon> {
        val candidates = mutableListOf<CandidatePolygon>()
        val mat = Mat(); val grayMat = Mat(); val bilateralMat = Mat()
        val edgeMat = Mat(); val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>(); var morphKernel: Mat? = null

        try {
            Utils.bitmapToMat(fullBitmap, mat)
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY)
            
            Imgproc.bilateralFilter(grayMat, bilateralMat, 9, 15.0, 15.0)
            Imgproc.Canny(bilateralMat, edgeMat, 60.0, 180.0)
            
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 5.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val minPlateHeight = fullBitmap.height * 0.025

            for (contour in contours) {
                val contour2f = MatOfPoint2f(*contour.toArray())
                val rotatedRect = Imgproc.minAreaRect(contour2f)
                
                val rectW = max(rotatedRect.size.width, rotatedRect.size.height)
                val rectH = min(rotatedRect.size.width, rotatedRect.size.height)
                if (rectH < minPlateHeight) continue 
                
                val aspectRatio = rectW / rectH
                if (aspectRatio in 2.3..5.7) {
                    val roughPoints = arrayOfNulls<Point>(4)
                    rotatedRect.points(roughPoints)
                    
                    val immutablePoints = roughPoints.filterNotNull().map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
                    val xCoords = immutablePoints.map { it.x }; val yCoords = immutablePoints.map { it.y }
                    val bounds = RectF(xCoords.min(), yCoords.min(), xCoords.max(), yCoords.max())
                    candidates.add(CandidatePolygon(immutablePoints, bounds))
                }
            }
            return candidates
        } catch (e: Exception) {
            return emptyList()
        } finally {
            mat.release(); grayMat.release(); bilateralMat.release()
            edgeMat.release(); hierarchy.release(); contours.forEach { it.release() }; morphKernel?.release()
        }
    }

    fun refineAnchoredPolygon(fullBitmap: Bitmap, anchorPolygon: List<ImmutablePoint>, targetLevel: Int): List<ImmutablePoint>? {
        val padding = if (targetLevel == 1) 30f else 15f 
        
        val minX = anchorPolygon.minOf { it.x } - padding
        val minY = anchorPolygon.minOf { it.y } - padding
        val maxX = anchorPolygon.maxOf { it.x } + padding
        val maxY = anchorPolygon.maxOf { it.y } + padding

        val safeRect = android.graphics.Rect(
            max(0, minX.toInt()), max(0, minY.toInt()),
            min(fullBitmap.width, maxX.toInt()), min(fullBitmap.height, maxY.toInt())
        )
        
        // 🌟 폭발 방지 (크래시 차단): 자르려는 영역이 너무 작거나 0이면 원본 앵커 반환
        if (safeRect.width() <= 1 || safeRect.height() <= 1) return null

        val roiBitmap = Bitmap.createBitmap(fullBitmap, safeRect.left, safeRect.top, safeRect.width(), safeRect.height())

        val anchorMatOfPoint2f = MatOfPoint2f(*anchorPolygon.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        val anchorRotatedRect = Imgproc.minAreaRect(anchorMatOfPoint2f)
        val anchorArea = anchorRotatedRect.size.width * anchorRotatedRect.size.height
        val anchorCenter = anchorRotatedRect.center
        val anchorAngle = anchorRotatedRect.angle

        val roiMat = Mat(); val roiGray = Mat(); val roiEdge = Mat(); val roiContours = ArrayList<MatOfPoint>()
        var bestTightPolygon: List<ImmutablePoint>? = null
        
        try {
            Utils.bitmapToMat(roiBitmap, roiMat)
            Imgproc.cvtColor(roiMat, roiGray, Imgproc.COLOR_RGBA2GRAY)
            
            val cannyThresh1 = if (targetLevel == 1) 40.0 else 20.0
            val cannyThresh2 = if (targetLevel == 1) 120.0 else 60.0
            
            Imgproc.Canny(roiGray, roiEdge, cannyThresh1, cannyThresh2) 
            Imgproc.findContours(roiEdge, roiContours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var minAreaDiff = Double.MAX_VALUE

            for (contour in roiContours) {
                val contour2f = MatOfPoint2f(*contour.toArray())
                val candRect = Imgproc.minAreaRect(contour2f)
                
                val candArea = candRect.size.width * candRect.size.height
                val candGlobalCenter = Point(candRect.center.x + safeRect.left, candRect.center.y + safeRect.top)

                if (candArea / anchorArea !in 0.7..1.3) continue 
                if (Math.hypot(candGlobalCenter.x - anchorCenter.x, candGlobalCenter.y - anchorCenter.y) > 30.0) continue 
                val angleDiff = Math.abs(anchorAngle - candRect.angle)
                if (angleDiff > 10.0 && angleDiff < 80.0) continue 

                val areaDiff = Math.abs(anchorArea - candArea)
                if (areaDiff < minAreaDiff) {
                    minAreaDiff = areaDiff
                    val candPoints = arrayOfNulls<Point>(4)
                    candRect.points(candPoints)
                    bestTightPolygon = candPoints.filterNotNull().map { 
                        ImmutablePoint((it.x + safeRect.left).toFloat(), (it.y + safeRect.top).toFloat()) 
                    }
                }
            }
        } finally {
            roiMat.release(); roiGray.release(); roiEdge.release()
            roiContours.forEach { it.release() }; anchorMatOfPoint2f.release()
            roiBitmap.recycle()
        }

        return bestTightPolygon
    }
}
