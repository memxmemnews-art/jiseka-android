package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    data class LineData(val p1: Point, val p2: Point) {
        val angle: Double
            get() = Math.toDegrees(Math.atan2(p2.y - p1.y, p2.x - p1.x))
        val center: Point
            get() = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
        val length: Double
            get() = Math.hypot(p2.x - p1.x, p2.y - p1.y)
    }

    fun findPlateCorners(roiBitmap: Bitmap): List<PointF>? {
        val mat = Mat()
        val grayMat = Mat()
        val bilateralMat = Mat()
        val edgeMat = Mat()
        val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        var morphKernel: Mat? = null

        try {
            Utils.bitmapToMat(roiBitmap, mat)
            
            // 1. 전처리 (CLAHE + Bilateral Filter)
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY)
            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(grayMat, grayMat)
            clahe.clear() 
            Imgproc.bilateralFilter(grayMat, bilateralMat, 11, 17.0, 17.0)

            // 2. 엣지 및 형태학적 덩어리 생성
            Imgproc.Canny(bilateralMat, edgeMat, 50.0, 150.0)
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 5.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)

            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val candidates = mutableListOf<Pair<RotatedRect, MatOfPoint>>()
            
            // 동적 면적 하한선 (가이드 박스 면적의 30%)
            val roiArea = roiBitmap.width.toDouble() * roiBitmap.height.toDouble()
            val minRequiredArea = roiArea * 0.30

            for (contour in contours) {
                // Convex Hull 포장
                val hull = MatOfInt()
                Imgproc.convexHull(contour, hull)
                val contourArray = contour.toArray()
                val hullPoints = hull.toArray().map { contourArray[it] }.toTypedArray()
                val hullContour = MatOfPoint(*hullPoints)
                
                val area = Imgproc.contourArea(hullContour)
                hull.release(); hullContour.release()

                if (area < minRequiredArea) continue

                val contour2f = MatOfPoint2f(*contourArray)
                val rotatedRect = Imgproc.minAreaRect(contour2f)
                
                val rectWidth = max(rotatedRect.size.width, rotatedRect.size.height)
                val rectHeight = min(rotatedRect.size.width, rotatedRect.size.height)
                if (rectHeight <= 0) continue 
                
                val aspectRatio = rectWidth / rectHeight
                // 최신 승용차 번호판 타겟팅 (비율 2.5 ~ 5.5)
                if (aspectRatio in 2.5..5.5) {
                    candidates.add(Pair(rotatedRect, contour))
                }
            }

            candidates.sortByDescending { it.first.size.area() }

            // 3. 평탄화(Warp) 및 OCR 검증
            for ((rotatedRect, contour) in candidates) {
                val roughPoints = arrayOfNulls<Point>(4)
                rotatedRect.points(roughPoints)
                val nonNullPoints = roughPoints.filterNotNull().toTypedArray()

                // 평탄화
                val warpedBitmap = warpForOCR(grayMat, nonNullPoints) ?: continue
                
                if (verifyWithOCR(warpedBitmap)) {
                    val boundingRect = rotatedRect.boundingRect()
                    // 4. Hough 교차점 기반 정밀 복원
                    val refinedCorners = extractCornersWithHough(contour, edgeMat, boundingRect)
                    
                    val finalPoints = if (refinedCorners != null && refinedCorners.size == 4) {
                        refinedCorners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()
                    } else {
                        nonNullPoints
                    }

                    // 5. 서브픽셀 미세 조정
                    return applySubPixelRefinement(grayMat, finalPoints)
                }
            }
            return null 

        } catch (e: Exception) {
            Log.e("CAMERA_DEBUG", "번호판 엔진 분석 에러", e)
            return null
        } finally {
            // 철저한 메모리 해제
            mat.release(); grayMat.release(); bilateralMat.release() 
            edgeMat.release(); hierarchy.release()
            contours.forEach { it.release() }; morphKernel?.release()
        }
    }

    private fun warpForOCR(grayMat: Mat, corners: Array<Point>): Bitmap? {
        if (corners.size != 4) return null
        val orderedCorners = orderCorners(corners)
        val srcPts = MatOfPoint2f(*orderedCorners)

        val targetWidth = 400.0
        val targetHeight = 100.0
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0), Point(targetWidth, 0.0),
            Point(targetWidth, targetHeight), Point(0.0, targetHeight)
        )

        val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val warpedMat = Mat()
        val colorWarped = Mat()
        
        try {
            Imgproc.warpPerspective(grayMat, warpedMat, transform, Size(targetWidth, targetHeight))
            Imgproc.cvtColor(warpedMat, colorWarped, Imgproc.COLOR_GRAY2RGBA)
            val resultBitmap = Bitmap.createBitmap(targetWidth.toInt(), targetHeight.toInt(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(colorWarped, resultBitmap)
            return resultBitmap
        } catch (e: Exception) {
            return null
        } finally {
            srcPts.release(); dstPts.release()
            transform.release(); warpedMat.release(); colorWarped.release()
        }
    }

    private fun orderCorners(pts: Array<Point>): Array<Point> {
        val sortedByX = pts.sortedBy { it.x }
        val leftPoints = sortedByX.take(2).sortedBy { it.y }
        val rightPoints = sortedByX.takeLast(2).sortedBy { it.y }
        return arrayOf(leftPoints[0], rightPoints[0], rightPoints[1], leftPoints[1]) // tl, tr, br, bl
    }

    private fun verifyWithOCR(bitmap: Bitmap): Boolean {
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val result = Tasks.await(recognizer.process(image))
            result.text.count { it.isDigit() } >= 2 // 숫자 2개 이상 통과
        } catch (e: Exception) {
            false
        } finally {
            bitmap.recycle() 
        }
    }

    private fun applySubPixelRefinement(grayMat: Mat, roughPoints: Array<Point>): List<PointF> {
        val cornersMat = MatOfPoint2f(*roughPoints)
        try {
            val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.001)
            Imgproc.cornerSubPix(grayMat, cornersMat, Size(5.0, 5.0), Size(-1.0, -1.0), criteria)
        } catch (e: Exception) {
            Log.w("CAMERA_DEBUG", "Subpixel refinement 실패, 기존 좌표 우회")
        }
        val refinedResult = cornersMat.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        cornersMat.release()
        return refinedResult
    }

    private fun extractCornersWithHough(contour: MatOfPoint, edgeMat: Mat, boundingRect: Rect): List<PointF>? {
        val mask = Mat.zeros(edgeMat.size(), CvType.CV_8UC1)
        val isolatedEdges = Mat()
        val lines = Mat()
        try {
            Imgproc.drawContours(mask, listOf(contour), -1, Scalar(255.0), 2)
            Core.bitwise_and(edgeMat, mask, isolatedEdges)
            Imgproc.HoughLinesP(isolatedEdges, lines, 1.0, Math.PI / 180, 40, 30.0, 10.0)

            val horizontalLines = mutableListOf<LineData>()
            val verticalLines = mutableListOf<LineData>()

            for (i in 0 until lines.rows()) {
                val vec = lines.get(i, 0)
                if (vec != null && vec.size >= 4) {
                    val line = LineData(Point(vec[0], vec[1]), Point(vec[2], vec[3]))
                    val angleAbs = abs(line.angle)
                    if (angleAbs < 30 || angleAbs > 150) horizontalLines.add(line)
                    else if (angleAbs in 60.0..120.0) verticalLines.add(line)
                }
            }
            if (horizontalLines.isEmpty() || verticalLines.isEmpty()) return null

            val centerY = boundingRect.y + boundingRect.height / 2.0
            val centerX = boundingRect.x + boundingRect.width / 2.0

            val top = horizontalLines.filter { it.center.y < centerY }.maxByOrNull { it.length }
            val bottom = horizontalLines.filter { it.center.y > centerY }.maxByOrNull { it.length }
            val left = verticalLines.filter { it.center.x < centerX }.maxByOrNull { it.length }
            val right = verticalLines.filter { it.center.x > centerX }.maxByOrNull { it.length }

            if (top == null || bottom == null || left == null || right == null) return null

            val topLeft = computeIntersection(top, left)
            val topRight = computeIntersection(top, right)
            val bottomRight = computeIntersection(bottom, right)
            val bottomLeft = computeIntersection(bottom, left)

            if (topLeft != null && topRight != null && bottomRight != null && bottomLeft != null) {
                return listOf(topLeft, topRight, bottomRight, bottomLeft)
            }
            return null
        } finally {
            mask.release(); isolatedEdges.release(); lines.release()
        }
    }

    private fun computeIntersection(l1: LineData, l2: LineData): PointF? {
        val x1 = l1.p1.x; val y1 = l1.p1.y
        val x2 = l2.p2.x; val y2 = l2.p2.y
        val x3 = l2.p1.x; val y3 = l2.p1.y
        val x4 = l2.p2.x; val y4 = l2.p2.y

        val denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (denominator == 0.0) return null 

        val intersectX = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        val intersectY = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return PointF(intersectX.toFloat(), intersectY.toFloat())
    }
}
