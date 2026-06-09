package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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

    private fun addDebugHUD(original: Bitmap, title: String, logs: List<String>): Bitmap {
        val bmp = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bmp)
        
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 50f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }
        val bgPaint = Paint().apply { color = Color.parseColor("#99000000") }

        val padding = 30f
        val lineHeight = 60f
        val totalHeight = padding + lineHeight + (logs.size * lineHeight) + padding
        
        canvas.drawRect(0f, 0f, bmp.width.toFloat(), totalHeight, bgPaint)

        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        canvas.drawText(title, padding, padding + 40f, paint)

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        var currentY = padding + lineHeight + 40f
        for (log in logs) {
            canvas.drawText(log, padding, currentY, paint)
            currentY += lineHeight
        }
        return bmp
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
        // 🛠️ [1단계 수정] 안전 영역(Padded ROI) 대폭 다이어트 (2.5 -> 1.5배)
        // =====================================================================
        val paddedSize = lineLen * 1.5 
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
            
            val hudBmp = addDebugHUD(debugBmp, "Step 1: Optimized Padded ROI", listOf(
                "Input Line Length: ${String.format("%.1f", lineLen)} px",
                "Padded ROI Size: ${paddedRect.width} x ${paddedRect.height} (x1.5 instead of x2.5)",
                "Result: Noise (Grille/Floor) heavily reduced"
            ))
            
            it.pauseAndShowStep("1단계: 다이어트된 Padded ROI 확보", hudBmp)
            debugMat.release(); debugBmp.recycle()
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

        var tightTop = 0
        var tightBottom = rotatedPaddedGray.rows()
        var tightLeft = 0
        var tightRight = rotatedPaddedGray.cols()
        
        val sobelX = Mat(); val searchMat = Mat()
        val vProjection = Mat(); val hProjection = Mat()
        
        try {
            Imgproc.Sobel(rotatedPaddedGray, sobelX, CvType.CV_32F, 1, 0, 3)
            Core.convertScaleAbs(sobelX, sobelX)
            Imgproc.threshold(sobelX, sobelX, 50.0, 255.0, Imgproc.THRESH_BINARY)

            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(sobelX.cols(), sobelX.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(sobelX, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 2.5 (1/4): Vertical Edge Extraction", listOf(
                    "Applied Rotation: ${String.format("%.2f", -angle)} deg",
                    "Sobel Thresh: 50.0"
                ))
                it.pauseAndShowStep("2.5단계 (1/4): 수직 엣지 추출", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🛠️ [2.5단계 수정] 세로 탐색: 폭을 30%로 극단적 축소 (다이아몬드 그릴 차단)
            // =====================================================================
            val searchWidth = max(30.0, lineLen * 0.3).toInt() 
            val searchLeft = (roiCx - searchWidth / 2.0).toInt().coerceIn(0, sobelX.cols() - 1)
            val searchRight = (roiCx + searchWidth / 2.0).toInt().coerceIn(1, sobelX.cols())
            
            val vSearchRect = Rect(searchLeft, 0, searchRight - searchLeft, sobelX.rows())
            sobelX.submat(vSearchRect).copyTo(searchMat)

            Core.reduce(searchMat, vProjection, 1, Core.REDUCE_SUM, CvType.CV_32S)
            val vProfile = IntArray(vProjection.rows())
            vProjection.get(0, 0, vProfile)

            val startY = roiCy.toInt().coerceIn(0, vProfile.lastIndex)
            var centerSum = 0L; var count = 0
            for(i in max(0, startY - 3)..min(vProfile.lastIndex, startY + 3)) {
                centerSum += vProfile[i]; count++
            }
            val vCenterDensity = centerSum / count.toDouble()
            val vThreshold = vCenterDensity * 0.25 

            var detectedTop = startY; var detectedBottom = startY
            for (y in startY downTo 0) { if (vProfile[y] < vThreshold) { detectedTop = y; break } }
            for (y in startY..vProfile.lastIndex) { if (vProfile[y] < vThreshold) { detectedBottom = y; break } }

            var dynamicHeight = (detectedBottom - detectedTop).toDouble()
            val expectedGeometricHeight = lineLen / 4.7
            var usedVerticalFallback = false

            // 🛠️ 롤백(Fallback) 발생 시 패딩 넉넉하게 2.5배로 할당 (측면 투시왜곡 방어)
            if (dynamicHeight > expectedGeometricHeight * 2.5 || dynamicHeight < expectedGeometricHeight * 0.4) {
                val fallbackHeight = expectedGeometricHeight * 2.5 
                tightTop = max(0, (roiCy - fallbackHeight / 2.0).toInt())
                tightBottom = min(rotatedPaddedGray.rows(), (roiCy + fallbackHeight / 2.0).toInt())
                usedVerticalFallback = true
            } else {
                val padding = dynamicHeight * 0.2
                tightTop = max(0, (detectedTop - padding).toInt())
                tightBottom = min(rotatedPaddedGray.rows(), (detectedBottom + padding).toInt())
            }

            debugListener?.let {
                val debugMat = rotatedPaddedMat.clone()
                val color = if(usedVerticalFallback) Scalar(255.0, 0.0, 0.0, 255.0) else Scalar(255.0, 255.0, 0.0, 255.0)
                Imgproc.rectangle(debugMat, vSearchRect, Scalar(255.0, 0.0, 255.0, 255.0), 3)
                Imgproc.line(debugMat, Point(0.0, tightTop.toDouble()), Point(debugMat.cols().toDouble(), tightTop.toDouble()), color, 4)
                Imgproc.line(debugMat, Point(0.0, tightBottom.toDouble()), Point(debugMat.cols().toDouble(), tightBottom.toDouble()), color, 4)
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val statusText = if(usedVerticalFallback) "FAIL -> Fallback applied (2.5x)" else "SUCCESS -> Margin added"
                val hudBmp = addDebugHUD(debugBmp, "Step 2.5 (2/4): Vertical Scan (Grille Blocked)", listOf(
                    "Search Width: $searchWidth px (0.3x LineLen - Narrow)",
                    "Center Density: ${String.format("%.1f", vCenterDensity)}",
                    "Status: $statusText"
                ))
                it.pauseAndShowStep("2.5단계 (2/4): 세로 탐색 (그릴 노이즈 원천차단)", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🛠️ [2.6단계 유지] 가로 방향 동적 확장 (빈 공간 함정 해결 버전)
            // =====================================================================
            val bandHeight = tightBottom - tightTop
            var usedHorizontalFallback = false
            var hCenterDensity = 0.0
            var hThreshold = 0.0
            var textWidth = 0.0
            
            if (bandHeight > 10) {
                val hSearchRect = Rect(0, tightTop, sobelX.cols(), bandHeight)
                val textBandSobel = sobelX.submat(hSearchRect)

                Core.reduce(textBandSobel, hProjection, 0, Core.REDUCE_SUM, CvType.CV_32S)
                val hProfile = IntArray(hProjection.cols())
                hProjection.get(0, 0, hProfile)

                val startX = roiCx.toInt().coerceIn(0, hProfile.lastIndex)
                
                val searchSpan = (lineLen * 0.5).toInt()
                val peakLeft = max(0, startX - searchSpan)
                val peakRight = min(hProfile.lastIndex, startX + searchSpan)
                
                var maxDensity = 0
                for(i in peakLeft..peakRight) {
                    if(hProfile[i] > maxDensity) maxDensity = hProfile[i]
                }
                
                hCenterDensity = maxDensity.toDouble()
                hThreshold = hCenterDensity * 0.15 

                var textLeft = startX
                var textRight = startX

                for (x in 0..startX) { if (hProfile[x] > hThreshold) { textLeft = x; break } }
                for (x in hProfile.lastIndex downTo startX) { if (hProfile[x] > hThreshold) { textRight = x; break } }

                textWidth = (textRight - textLeft).toDouble()
                
                if (textWidth < lineLen * 0.4 || textWidth > lineLen * 2.0) {
                    val fallbackWidth = max(100.0, lineLen * 1.1)
                    tightLeft = max(0, (roiCx - fallbackWidth / 2.0).toInt())
                    tightRight = min(rotatedPaddedGray.cols(), (roiCx + fallbackWidth / 2.0).toInt())
                    usedHorizontalFallback = true
                } else {
                    val paddingX = textWidth * 0.25
                    tightLeft = max(0, (textLeft - paddingX).toInt())
                    tightRight = min(rotatedPaddedGray.cols(), (textRight + paddingX).toInt())
                }
                textBandSobel.release()
            } else {
                val fallbackWidth = max(100.0, lineLen * 1.1)
                tightLeft = max(0, (roiCx - fallbackWidth / 2.0).toInt())
                tightRight = min(rotatedPaddedGray.cols(), (roiCx + fallbackWidth / 2.0).toInt())
                usedHorizontalFallback = true
            }

            debugListener?.let {
                val debugMat = rotatedPaddedMat.clone()
                val color = if(usedHorizontalFallback) Scalar(255.0, 0.0, 0.0, 255.0) else Scalar(0.0, 255.0, 0.0, 255.0)
                Imgproc.line(debugMat, Point(tightLeft.toDouble(), tightTop.toDouble()), Point(tightLeft.toDouble(), tightBottom.toDouble()), color, 4)
                Imgproc.line(debugMat, Point(tightRight.toDouble(), tightTop.toDouble()), Point(tightRight.toDouble(), tightBottom.toDouble()), color, 4)
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val statusText = if(usedHorizontalFallback) "FAIL -> Fallback (1.1x LineLen)" else "SUCCESS -> TextWidth + 25% Margin"
                val hudBmp = addDebugHUD(debugBmp, "Step 2.6 (4/4): Horizontal Scan Result", listOf(
                    "Center Text Peak Density: ${String.format("%.1f", hCenterDensity)}",
                    "Detected Text Block Width: ${String.format("%.1f", textWidth)} px",
                    "Status: $statusText"
                ))
                it.pauseAndShowStep("2.6단계 (4/4): 가로 스캔 (빈 공간 함정 해결됨)", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

        } finally {
            sobelX.release(); searchMat.release()
            vProjection.release(); hProjection.release()
        }

        // =====================================================================
        // [3단계] 최종 타이트 ROI 생성
        // =====================================================================
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
            Imgproc.rectangle(debugMat, safeTightRect, Scalar(0.0, 255.0, 0.0, 255.0), 10)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val hudBmp = addDebugHUD(debugBmp, "Step 3: Final Tight ROI Generation", listOf(
                "Final ROI Size: ${safeTightRect.width} x ${safeTightRect.height} px"
            ))
            it.pauseAndShowStep("3단계: 최종 타이트 ROI", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        val roiEdge = Mat(); val combinedEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>(); val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            val clahe = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
            clahe.apply(tightGray, tightGray)
            
            val meanVal = Core.mean(tightGray).`val`[0]
            Imgproc.Canny(tightGray, roiEdge, max(0.0, 0.33 * meanVal), min(255.0, 1.33 * meanVal))
            
            val kernelLen = (safeTightRect.width * 0.08).toInt().coerceIn(10, 60)
            val kernel0 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelLen.toDouble(), 2.0))
            Imgproc.morphologyEx(roiEdge, combinedEdge, Imgproc.MORPH_CLOSE, kernel0)
            kernel0.release()

            Imgproc.findContours(combinedEdge, roiContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            
            val validPolygons = ArrayList<MatOfPoint2f>()
            for (contour in roiContours) {
                val approx = extractRobustPolygon(contour) ?: continue
                validPolygons.add(approx)
            }

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
