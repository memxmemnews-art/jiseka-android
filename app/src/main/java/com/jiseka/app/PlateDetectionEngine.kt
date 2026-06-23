package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(stageName: String, bitmap: Bitmap)
    }

    private fun drawTextWithWrap(canvas: Canvas, text: String, x: Float, y: Float, paint: Paint, maxWidth: Float, lineHeight: Float): Float {
        var currentY = y
        val originalTextSize = paint.textSize
        
        var textWidth = paint.measureText(text)
        if (textWidth > maxWidth) {
            paint.textSize = originalTextSize * 0.85f
            textWidth = paint.measureText(text)
        }

        if (textWidth > maxWidth) {
            val words = text.split(" ")
            var currentLine = ""

            for (word in words) {
                val testLine = if (currentLine.isEmpty()) word else "$currentLine $word"
                if (paint.measureText(testLine) > maxWidth && currentLine.isNotEmpty()) {
                    canvas.drawText(currentLine, x, currentY, paint)
                    currentLine = word
                    currentY += lineHeight
                } else {
                    currentLine = testLine
                }
            }
            if (currentLine.isNotEmpty()) {
                canvas.drawText(currentLine, x, currentY, paint)
                currentY += lineHeight
            }
        } else {
            canvas.drawText(text, x, currentY, paint)
            currentY += lineHeight
        }

        paint.textSize = originalTextSize
        return currentY
    }

    private fun addDebugHUD(original: Bitmap, title: String, logs: List<String>, screenRatio: Float): Bitmap {
        val canvasWidth = 1080
        val canvasHeight = (canvasWidth * screenRatio).toInt()
        val combinedBmp = Bitmap.createBitmap(canvasWidth, canvasHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combinedBmp)
        
        val bgPaint = Paint().apply { color = Color.parseColor("#E6000000") }
        canvas.drawRect(0f, 0f, canvasWidth.toFloat(), canvasHeight.toFloat(), bgPaint)

        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 38f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val paddingX = 80f
        val lineHeight = 55f
        val maxTextWidth = canvasWidth - (paddingX * 2)
        var currentY = 100f 
        
        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        paint.textSize = 45f
        currentY = drawTextWithWrap(canvas, title, paddingX, currentY, paint, maxTextWidth, lineHeight)

        currentY += 20f 

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        paint.textSize = 35f
        for (log in logs) {
            currentY = drawTextWithWrap(canvas, log, paddingX, currentY, paint, maxTextWidth, lineHeight)
        }

        val textBottom = currentY + 30f 
        val margin = 50f
        
        val maxImgWidth = canvasWidth - (margin * 2)
        val maxImgHeight = canvasHeight - textBottom - margin 

        if (maxImgHeight > 0) {
            val scaleX = maxImgWidth / original.width.toFloat()
            val scaleY = maxImgHeight / original.height.toFloat()
            val safeScaleFactor = min(scaleX, scaleY)

            val scaledWidth = (original.width * safeScaleFactor).toInt()
            val scaledHeight = (original.height * safeScaleFactor).toInt()
            
            val scaledImg = Bitmap.createScaledBitmap(original, max(1, scaledWidth), max(1, scaledHeight), true)
            
            val imgX = (canvasWidth - scaledWidth) / 2f
            val imgY = textBottom + (maxImgHeight - scaledHeight) / 2f

            val borderPaint = Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 6f }
            canvas.drawRect(imgX - 3f, imgY - 3f, imgX + scaledWidth + 3f, imgY + scaledHeight + 3f, borderPaint)
            
            canvas.drawBitmap(scaledImg, imgX, imgY, null)
            scaledImg.recycle()
        }

        return combinedBmp
    }

    fun rescuePlateFromPoint(
        fullBitmap: Bitmap, 
        touchX: Float, touchY: Float, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat(); val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        val screenRatio = fullMat.rows().toFloat() / fullMat.cols().toFloat()
        val cx = touchX.toDouble(); val cy = touchY.toDouble()

        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.circle(debugMat, Point(cx, cy), 20, Scalar(255.0, 0.0, 0.0, 255.0), -1)
            Imgproc.circle(debugMat, Point(cx, cy), 25, Scalar(255.0, 255.0, 255.0, 255.0), 4)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 1: User Touch Point", listOf(
                "Action: 터치 좌표 수신 완료",
                "Touch Point X: ${cx.toInt()} px", 
                "Touch Point Y: ${cy.toInt()} px",
                "Resolution: ${fullMat.cols()} x ${fullMat.rows()}"
            ), screenRatio)
            it.pauseAndShowStep("1단계: 터치 좌표 매핑", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // [Step 2] 초기 ROI 설정
        // =====================================================================
        val roiWidth = (fullMat.cols() * 0.22).toInt() 
        val roiHeight = (roiWidth / 2.0).toInt()       

        val looseLeft = (cx - roiWidth / 2.0).toInt().coerceIn(0, fullMat.cols() - 1)
        val looseTop = (cy - roiHeight / 2.0).toInt().coerceIn(0, fullMat.rows() - 1)
        val looseRight = (cx + roiWidth / 2.0).toInt().coerceIn(1, fullMat.cols())
        val looseBottom = (cy + roiHeight / 2.0).toInt().coerceIn(1, fullMat.rows())

        val looseRect = Rect(looseLeft, looseTop, looseRight - looseLeft, looseBottom - looseTop)
        
        val looseMat = Mat(); val looseGray = Mat()
        fullMat.submat(looseRect).copyTo(looseMat)
        fullGray.submat(looseRect).copyTo(looseGray)

        debugListener?.let {
            val debugMat = fullMat.clone()
            Imgproc.rectangle(debugMat, looseRect, Scalar(0.0, 255.0, 0.0, 255.0), 8)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 2: Downsized ROI (Ratio 2:1)", listOf(
                "ROI Width: $roiWidth px",
                "ROI Height: $roiHeight px",
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 컴팩트 탐색 구역 설정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // [Step 3 & 4] 1차 이진화 및 모폴로지
        // =====================================================================
        val thresh = Mat()
        Imgproc.medianBlur(looseGray, looseGray, 3)
        Imgproc.GaussianBlur(looseGray, thresh, Size(5.0, 5.0), 0.0)
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, 7.0)

        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        
        // =====================================================================
        // [Step 5] 바운딩 박스 기반 뼈대 구축 및 군집화
        // =====================================================================
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect)
        val charList = mutableListOf<CharData>()

        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
            
            if (ratio in 1.2..4.2 && area > 120 && rect.height >= 40 && area < looseRect.area() * 0.08) {
                val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
                charList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect))
            }
        }
        tempContours.forEach { it.release() }; tempHierarchy.release(); tempOpen.release(); tempClose.release()

        if (charList.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        val sortedChars = charList.sortedBy { it.center.x }
        val gaps = (1 until sortedChars.size).map { sortedChars[it].center.x - sortedChars[it - 1].center.x }
        val medianGap = if (gaps.isNotEmpty()) gaps.sorted()[gaps.size / 2] else 0.0
        val avgWidth = sortedChars.map { it.width }.average()
        
        val maxGap = max(medianGap * 1.7, avgWidth * 1.8) 

        val clusters = mutableListOf<MutableList<CharData>>()
        var currentCluster = mutableListOf(sortedChars.first())

        for (i in 1 until sortedChars.size) {
            val gap = sortedChars[i].center.x - sortedChars[i - 1].center.x
            if (gap > maxGap) {
                clusters.add(currentCluster) 
                currentCluster = mutableListOf(sortedChars[i]) 
            } else {
                currentCluster.add(sortedChars[i])
            }
        }
        clusters.add(currentCluster)

        val validChars = clusters.maxByOrNull { it.size } ?: sortedChars

        if (validChars.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        val pointsMat = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(pointsMat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        val vx = line.get(0, 0)[0]; val vy = line.get(1, 0)[0]
        
        val topPtsArray = validChars.map { Point(it.center.x, it.rect.y.toDouble()) }.toTypedArray()
        val bottomPtsArray = validChars.map { Point(it.center.x, it.rect.y.toDouble() + it.rect.height) }.toTypedArray()
        
        val topPts = MatOfPoint2f(*topPtsArray); val bottomPts = MatOfPoint2f(*bottomPtsArray)
        val topLineMat = Mat(); val bottomLineMat = Mat()
        
        Imgproc.fitLine(topPts, topLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        Imgproc.fitLine(bottomPts, bottomLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)

        val tvx = topLineMat.get(0, 0)[0]; val tvy = topLineMat.get(1, 0)[0]
        val tx0 = topLineMat.get(2, 0)[0]; val ty0 = topLineMat.get(3, 0)[0]
        
        val bvx = bottomLineMat.get(0, 0)[0]; val bvy = bottomLineMat.get(1, 0)[0]
        val bx0 = bottomLineMat.get(2, 0)[0]; val by0 = bottomLineMat.get(3, 0)[0]

        val firstChar = validChars.first()
        val lastChar = validChars.last()

        val leftTopMid = Point(firstChar.rect.x + firstChar.rect.width / 2.0, firstChar.rect.y.toDouble())
        val leftCenter = firstChar.center
        val leftBottomMid = Point(firstChar.rect.x + firstChar.rect.width / 2.0, firstChar.rect.y + firstChar.rect.height.toDouble())

        val rightTopMid = Point(lastChar.rect.x + lastChar.rect.width / 2.0, lastChar.rect.y.toDouble())
        val rightCenter = lastChar.center
        val rightBottomMid = Point(lastChar.rect.x + lastChar.rect.width / 2.0, lastChar.rect.y + lastChar.rect.height.toDouble())

        val leftPts = MatOfPoint2f(leftTopMid, leftCenter, leftBottomMid)
        val leftLine = Mat()
        Imgproc.fitLine(leftPts, leftLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var lvx = leftLine.get(0, 0)[0]; var lvy = leftLine.get(1, 0)[0]
        if (lvy < 0) { lvx = -lvx; lvy = -lvy }
        val lx0 = firstChar.center.x; val ly0 = firstChar.center.y

        val rightPts = MatOfPoint2f(rightTopMid, rightCenter, rightBottomMid)
        val rightLine = Mat()
        Imgproc.fitLine(rightPts, rightLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var rvx = rightLine.get(0, 0)[0]; var rvy = rightLine.get(1, 0)[0]
        if (rvy < 0) { rvx = -rvx; rvy = -rvy }
        val rx0 = lastChar.center.x; val ry0 = lastChar.center.y

        fun getIntersect(x1: Double, y1: Double, vx1: Double, vy1: Double, x2: Double, y2: Double, vx2: Double, vy2: Double): Point {
            val dx = x2 - x1; val dy = y2 - y1
            val det = vx2 * vy1 - vy2 * vx1
            if (abs(det) < 1e-6) return Point(x1, y1)
            val u = (dy * vx1 - dx * vy1) / det
            return Point(x2 + u * vx2, y2 + u * vy2)
        }

        val initTL = getIntersect(tx0, ty0, tvx, tvy, lx0, ly0, lvx, lvy)
        val initTR = getIntersect(tx0, ty0, tvx, tvy, rx0, ry0, rvx, rvy)
        val initBR = getIntersect(bx0, by0, bvx, bvy, rx0, ry0, rvx, rvy)
        val initBL = getIntersect(bx0, by0, bvx, bvy, lx0, ly0, lvx, lvy)
        
        debugListener?.let {
            val debugMat = looseMat.clone()
            for (charData in charList) {
                val color = if (validChars.contains(charData)) Scalar(0.0, 255.0, 0.0, 255.0) else Scalar(255.0, 0.0, 0.0, 255.0)
                Imgproc.rectangle(debugMat, charData.rect, color, 2)
            }
            for (i in 0..3) {
                val pts = arrayOf(initTL, initTR, initBR, initBL)
                Imgproc.line(debugMat, pts[i], pts[(i+1)%4], Scalar(255.0, 0.0, 255.0, 255.0), 3)
            }

            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 3~5: Base Wireframe", listOf(
                "Method: 바운딩 박스 기반 기초 뼈대 구축 완료",
                "초록 박스: 통과 / 빨간 박스: 탈락",
                "이 뼈대가 가림막 크기 역산의 기준이 됩니다."
            ), screenRatio)
            it.pauseAndShowStep("3~5단계: 노이즈 필터링 및 기초 뼈대 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        pointsMat.release(); line.release(); topPts.release(); bottomPts.release()
        topLineMat.release(); bottomLineMat.release()
        leftPts.release(); rightPts.release(); leftLine.release(); rightLine.release()

        // =====================================================================
        // 🚀 [Step 6] 3D 원근(사다리꼴 왜곡)을 보존하는 신형 번호판 규격 역산
        // =====================================================================
        var resultPoints: List<ImmutablePoint>? = null

        try {
            val avgH = validChars.map { it.height }.average()
            val midX = (initTL.x + initTR.x + initBR.x + initBL.x) / 4.0
            val midY = (initTL.y + initTR.y + initBR.y + initBL.y) / 4.0
            
            // 신형 번호판 규격 비율 및 가로세로 확장 배율(Scale Factor) 계산
            val textOccupancyY = 0.45 // 세로 대비 글자 비율 (약 45%)
            val scaleY = 1.0 / textOccupancyY
            
            val estimatedPlateH = avgH * scaleY
            val estimatedPlateW = estimatedPlateH * (520.0 / 110.0) // 대한민국 신형 번호판 표준 비율 4.727
            
            // 기울어진 텍스트 와이어프레임의 실제 가로 폭 계산 (원근 스케일링용)
            val textW = hypot(initTR.x - initTL.x, initTR.y - initTL.y)
            val scaleX = estimatedPlateW / textW

            // 직교 수직 벡터 계산 (nX, nY)
            val nX = -vy; val nY = vx

            // 각 꼭짓점을 로컬 축 기준으로 독립 확장하여 사다리꼴 왜곡 보존
            val finalPts = listOf(initTL, initTR, initBR, initBL).map { pt ->
                // 중심점으로부터의 거리
                val dx = pt.x - midX
                val dy = pt.y - midY
                
                // 로컬 좌표계 분리
                val localX = dx * vx + dy * vy
                val localY = dx * nX + dy * nY
                
                // 독립 팽창 적용
                val scaledX = localX * scaleX
                val scaledY = localY * scaleY
                
                // 원본 이미지 좌표계로 복원
                Point(
                    midX + scaledX * vx + scaledY * nX + looseRect.x,
                    midY + scaledX * vy + scaledY * nY + looseRect.y
                )
            }

            debugListener?.let {
                val debugMat = fullMat.clone()
                val colors = arrayOf(Scalar(255.0, 0.0, 0.0, 255.0), Scalar(0.0, 255.0, 0.0, 255.0), 
                                     Scalar(0.0, 0.0, 255.0, 255.0), Scalar(255.0, 255.0, 0.0, 255.0))
                val labels = arrayOf("TL", "TR", "BR", "BL")

                for (i in 0..3) {
                    Imgproc.line(debugMat, finalPts[i], finalPts[(i + 1) % 4], Scalar(0.0, 255.0, 0.0, 255.0), 5)
                    Imgproc.circle(debugMat, finalPts[i], 15, colors[i], -1)
                    Imgproc.putText(debugMat, labels[i], Point(finalPts[i].x - 20, finalPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 1.8, colors[i], 4)
                }
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 6: Perspective Plate Estimation", listOf(
                    "Target: 3D 측면 원근 왜곡 대응 가림막 생성",
                    "Formula: Text Height -> Plate Size -> Local Vector Expansion",
                    "Result: 사진의 사다리꼴 기울기와 소실점을 그대로 복사한 가림막 도출 완료"
                ), screenRatio)
                it.pauseAndShowStep("최종 단계: 원근 보존 가림막 좌표 확정", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release()
            looseMat.release(); looseGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
