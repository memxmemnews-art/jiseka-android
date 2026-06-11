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

    // 🛠️ 가로/세로 어떤 모양이 들어와도 글자를 침범하지 않고 안전하게 최대화하는 반응형 HUD
    private fun addDebugHUD(original: Bitmap, title: String, logs: List<String>, screenRatio: Float): Bitmap {
        val canvasWidth = 1080
        val canvasHeight = (canvasWidth * screenRatio).toInt()
        val combinedBmp = Bitmap.createBitmap(canvasWidth, canvasHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combinedBmp)
        
        // 1. 전체 배경색 (반투명 블랙)
        val bgPaint = Paint().apply { color = Color.parseColor("#E6000000") }
        canvas.drawRect(0f, 0f, canvasWidth.toFloat(), canvasHeight.toFloat(), bgPaint)

        // 2. 텍스트 페인트 설정 (크기 고정으로 절대 잘리지 않음)
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 45f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val padding = 40f
        val lineHeight = 65f
        
        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        canvas.drawText(title, padding, padding + 50f, paint)

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        var currentY = padding + lineHeight + 50f
        for (log in logs) {
            canvas.drawText(log, padding, currentY, paint)
            currentY += lineHeight
        }

        // 3. 남은 공간 계산
        val textBottom = currentY + 30f // 텍스트 아래쪽 마진
        val margin = 50f
        
        val maxImgWidth = canvasWidth - (margin * 2)
        val maxImgHeight = canvasHeight - textBottom - margin // 남은 높이 최대치

        if (maxImgHeight > 0) {
            // 4. 가로 확대 한계치와 세로 확대 한계치 중 '더 작은 값'을 선택 (비율 유지, 화면 이탈 방지)
            val scaleX = maxImgWidth / original.width.toFloat()
            val scaleY = maxImgHeight / original.height.toFloat()
            val safeScaleFactor = kotlin.math.min(scaleX, scaleY)

            val scaledWidth = (original.width * safeScaleFactor).toInt()
            val scaledHeight = (original.height * safeScaleFactor).toInt()
            
            val scaledImg = Bitmap.createScaledBitmap(original, max(1, scaledWidth), max(1, scaledHeight), true)
            
            // 5. 남은 공간의 정확한 정중앙에 배치
            val imgX = (canvasWidth - scaledWidth) / 2f
            val imgY = textBottom + (maxImgHeight - scaledHeight) / 2f

            // 외곽선 (이미지가 검은색일 경우 배경과 구분하기 위해)
            val borderPaint = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 5f }
            canvas.drawRect(imgX - 2f, imgY - 2f, imgX + scaledWidth + 2f, imgY + scaledHeight + 2f, borderPaint)
            
            canvas.drawBitmap(scaledImg, imgX, imgY, null)
            scaledImg.recycle()
        }

        return combinedBmp
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

        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()

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
        // [1단계] 넉넉한 정사각형 임시 ROI 확보 (선 길이의 2배로 넓게)
        // =====================================================================
        val looseSize = lineLen * 2.0 
        val looseLeft = (cx - looseSize / 2.0).toInt().coerceIn(0, fullMat.cols() - 1)
        val looseTop = (cy - looseSize / 2.0).toInt().coerceIn(0, fullMat.rows() - 1)
        val looseRight = (cx + looseSize / 2.0).toInt().coerceIn(1, fullMat.cols())
        val looseBottom = (cy + looseSize / 2.0).toInt().coerceIn(1, fullMat.rows())

        val looseRect = Rect(looseLeft, looseTop, looseRight - looseLeft, looseBottom - looseTop)
        
        val looseMat = Mat(); val looseGray = Mat()
        fullMat.submat(looseRect).copyTo(looseMat)
        fullGray.submat(looseRect).copyTo(looseGray)

        val roiCx = cx - looseLeft
        val roiCy = cy - looseTop

        // =====================================================================
        // 🚀 [2단계] 수평 정렬 후 '실제 선분(P1, P2) 좌표' 기반 타이트 박스 생성
        // =====================================================================
        val angle = Math.toDegrees(Math.atan2(dy, dx))
        val rotMat = Imgproc.getRotationMatrix2D(Point(roiCx, roiCy), -angle, 1.0)

        val rotatedLooseMat = Mat(); val rotatedLooseGray = Mat()
        Imgproc.warpAffine(looseMat, rotatedLooseMat, rotMat, looseMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(looseGray, rotatedLooseGray, rotMat, looseGray.size(), Imgproc.INTER_LINEAR)

        // 1. P1, P2를 looseMat 좌표계로 옮긴 후 회전 변환 적용 (rotP1, rotP2 획득)
        val srcLinePts = MatOfPoint2f(
            Point(p1x - looseLeft, p1y - looseTop),
            Point(p2x - looseLeft, p2y - looseTop)
        )
        val dstLinePts = MatOfPoint2f()
        Core.transform(srcLinePts, dstLinePts, rotMat)

        val rotLinePts = dstLinePts.toArray()
        val rotP1 = rotLinePts[0]
        val rotP2 = rotLinePts[1]

        // 2. 변환된 선분의 X축 시작과 끝을 구함
        val minX = min(rotP1.x, rotP2.x)
        val maxX = max(rotP1.x, rotP2.x)
        val rotLineLen = maxX - minX

        // 3. 선의 실제 위치(X) 기반으로 넉넉하게 너비 할당 (양옆 25% 추가)
        val marginX = rotLineLen * 0.25
        var tightLeft = (minX - marginX).toInt()
        var tightRight = (maxX + marginX).toInt()
        val tightWidth = tightRight - tightLeft

        // 4. 선의 실제 위치(Y) 기반으로 높이 할당 (4.7 대신 3.0 비율로 상하 여백 보장)
        val midY = (rotP1.y + rotP2.y) / 2.0
        val expectedHeight = max(tightWidth / 3.0, 80.0) 
        var tightTop = (midY - expectedHeight / 2.0).toInt()
        var tightBottom = (midY + expectedHeight / 2.0).toInt()

        // 안전 영역 제한
        tightLeft = tightLeft.coerceIn(0, rotatedLooseMat.cols() - 1)
        tightRight = tightRight.coerceIn(1, rotatedLooseMat.cols())
        tightTop = tightTop.coerceIn(0, rotatedLooseMat.rows() - 1)
        tightBottom = tightBottom.coerceIn(1, rotatedLooseMat.rows())

        val tightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)

        val tightMat = Mat(); val tightGray = Mat()
        rotatedLooseMat.submat(tightRect).copyTo(tightMat)
        rotatedLooseGray.submat(tightRect).copyTo(tightGray)

        srcLinePts.release(); dstLinePts.release()

        debugListener?.let {
            val debugMat = fullMat.clone()
            
            // 사용자가 그은 선 표시 (노란색)
            Imgproc.line(debugMat, Point(p1x, p1y), Point(p2x, p2y), Scalar(255.0, 255.0, 0.0, 255.0), 6)
            
            // 실제 변환 좌표(P1, P2) 기반으로 생성된 박스 표시 (하늘색)
            val tightRectPts = arrayOf(
                Point(tightLeft.toDouble(), tightTop.toDouble()),
                Point(tightRight.toDouble(), tightTop.toDouble()),
                Point(tightRight.toDouble(), tightBottom.toDouble()),
                Point(tightLeft.toDouble(), tightBottom.toDouble())
            )
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)
            val srcPts = MatOfPoint2f(*tightRectPts)
            val dstPts = MatOfPoint2f()
            Core.transform(srcPts, dstPts, invRotMat)
            val fullPts = dstPts.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }
            
            for (i in 0..3) {
                Imgproc.line(debugMat, fullPts[i], fullPts[(i + 1) % 4], Scalar(0.0, 200.0, 255.0, 255.0), 8)
            }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)

            val hudBmp = addDebugHUD(debugBmp, "Step 1 & 2: True Endpoint ROI Mapping", listOf(
                "Rotation Angle: ${String.format("%.1f", -angle)} deg",
                "Endpoint-based Width: $tightWidth px",
                "Endpoint-based Height (/3.0): ${expectedHeight.toInt()} px",
                "Status: Bounding box perfectly tied to user line"
            ), screenRatio)
            
            it.pauseAndShowStep("1~2단계: 실제 선분 기반 ROI 확정", hudBmp)
            debugMat.release(); debugBmp.recycle(); invRotMat.release(); srcPts.release(); dstPts.release()
        }

        // =====================================================================
        // [3단계] OpenCV 내장 윤곽선(Contour) 탐지
        // =====================================================================
        val thresh = Mat()
        
        // 박스가 넉넉해졌으므로, 너무 큰 덩어리로 뭉치지 않도록 커널 조절
        val kernelX = max(15.0, tightRect.width * 0.12) 
        val kernelY = max(5.0, tightRect.height * 0.10)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kernelX, kernelY))
        
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            Imgproc.GaussianBlur(tightGray, thresh, Size(5.0, 5.0), 0.0)
            Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 10.0)
            
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3: Morphology Close (Dynamic)", listOf(
                    "Kernel Size: ${kernelX.toInt()} x ${kernelY.toInt()}",
                    "Status: Text blocks fully merged!"
                ), screenRatio)
                it.pauseAndShowStep("3단계: 다이내믹 모폴로지 닫기", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var maxArea = -1.0
            var bestContour: MatOfPoint? = null
            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area > maxArea) {
                    maxArea = area
                    bestContour = contour
                }
            }

            val rectPoints = arrayOf(Point(), Point(), Point(), Point())
            if (bestContour != null && maxArea > tightRect.width * tightRect.height * 0.1) { 
                val contour2f = MatOfPoint2f(*bestContour.toArray())
                val minRect = Imgproc.minAreaRect(contour2f)
                minRect.points(rectPoints)
                contour2f.release()
            } else {
                rectPoints[0] = Point(0.0, tightGray.rows().toDouble())
                rectPoints[1] = Point(0.0, 0.0)
                rectPoints[2] = Point(tightGray.cols().toDouble(), 0.0)
                rectPoints[3] = Point(tightGray.cols().toDouble(), tightGray.rows().toDouble())
            }

            debugListener?.let {
                val debugMat = tightMat.clone()
                val color = Scalar(0.0, 255.0, 0.0, 255.0)
                for (i in 0..3) {
                    Imgproc.line(debugMat, rectPoints[i], rectPoints[(i + 1) % 4], color, 4)
                }
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val hudBmp = addDebugHUD(debugBmp, "Step 4: Final MinAreaRect", listOf(
                    "Detected Plate Area: ${String.format("%.1f", maxArea)} px",
                    "Method: MinAreaRect on massive blob",
                    "Status: Extraction Complete"
                ), screenRatio)
                it.pauseAndShowStep("4단계: 최종 번호판 사각형 추출", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // [4단계] 좌표 원복 
            // =====================================================================
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)

            val pointsInRotatedLoose = rectPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val srcMat = MatOfPoint2f(*pointsInRotatedLoose)
            val dstMat = MatOfPoint2f()

            Core.transform(srcMat, dstMat, invRotMat)

            resultPoints = dstMat.toArray().map { 
                ImmutablePoint((it.x + looseRect.x).toFloat(), (it.y + looseRect.y).toFloat()) 
            }

            invRotMat.release(); srcMat.release(); dstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release(); kernel.release()
            contours.forEach { it.release() }; hierarchy.release()
            
            rotMat.release()
            looseMat.release(); looseGray.release()
            rotatedLooseMat.release(); rotatedLooseGray.release()
            tightMat.release(); tightGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
