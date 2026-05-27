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
            
            // 🌟 1. 글자 연결성은 유지하면서 대각선 윤곽을 보존하는 절충형 커널로 변경
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 3.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val minPlateHeight = fullBitmap.height * 0.025

            for (contour in contours) {
                val contour2f = MatOfPoint2f(*contour.toArray())
                
                // 🌟 핵심: minAreaRect를 통해 회전 정보를 유지한 사각형 추출
                val rotatedRect = Imgproc.minAreaRect(contour2f)
                
                val rectW = max(rotatedRect.size.width, rotatedRect.size.height)
                val rectH = min(rotatedRect.size.width, rotatedRect.size.height)
                if (rectH < minPlateHeight) continue 
                
                val aspectRatio = rectW / rectH
                
                // 🌟 2. 크게 기울어진 번호판도 허용할 수 있도록 비율 범위 확장 (기존 2.3~5.7)
                if (aspectRatio in 1.8..6.0) {
                    val roughPoints = arrayOfNulls<Point>(4)
                    rotatedRect.points(roughPoints)
                    
                    val immutablePoints = roughPoints.filterNotNull().map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
                    val xCoords = immutablePoints.map { it.x }; val yCoords = immutablePoints.map { it.y }
                    val bounds = RectF(xCoords.min(), yCoords.min(), xCoords.max(), yCoords.max())
                    
                    // 회전된 4개의 꼭짓점 포인트와 축 정렬 바운드 박스를 함께 저장
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
                
                // 🌟 3. 각도 차이 허용 범위 확장 (기울어진 상태에서 재검색 시 실패 방지)
                val angleDiff = Math.abs(anchorAngle - candRect.angle)
                if (angleDiff > 15.0 && angleDiff < 75.0) continue 

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
