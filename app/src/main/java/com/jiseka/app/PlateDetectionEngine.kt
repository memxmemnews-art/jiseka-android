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
        // [Step 5] 바운딩 박스 기반 뼈대 구축 및 KOR 마크 동적 필터링
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

        val validChars = (clusters.maxByOrNull { it.size } ?: sortedChars).toMutableList()

        // 💡 파란색 KOR 마크 정밀 타격 및 깃발(Flag) 꽂기
        var hasKorMark = false

        if (validChars.size > 2) {
            val firstChar = validChars.first() 
            val roi = looseMat.submat(firstChar.rect)
            val meanColor = Core.mean(roi)
            roi.release() 
            
            val r = meanColor.`val`[0]
            val g = meanColor.`val`[1]
            val b = meanColor.`val`[2]
            
            // KOR 마크는 파란 바탕이므로 B 채널이 R, G보다 확연히 높음
            if (b > r + 20 && b > g + 10) {
                hasKorMark = true // 깃발 기억
                validChars.removeAt(0) // 뼈대에서 KOR 마크 탈락
            }
        }

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
                "Method: 기초 뼈대 구축 및 KOR 마크 동적 판별 완료",
                "Status: KOR Mark Detected = $hasKorMark",
                "이 뼈대가 가림막 크기 역산의 기준이 됩니다."
            ), screenRatio)
            it.pauseAndShowStep("3~5단계: 노이즈 필터링 및 기초 뼈대 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        pointsMat.release(); line.release(); topPts.release(); bottomPts.release()
        topLineMat.release(); bottomLineMat.release()
        leftPts.release(); rightPts.release(); leftLine.release(); rightLine.release()

        // =====================================================================
        // 🚀 [Step 6] 가로/세로 축 분리 스케일링 기반 3D 원근 보존 핏팅
        // =====================================================================
        var resultPoints: List<ImmutablePoint>? = null

        try {
            val avgH = validChars.map { it.height }.average()
            val textW = hypot(initTR.x - initTL.x, initTR.y - initTL.y)
            
            var midX = (initTL.x + initTR.x + initBR.x + initBL.x) / 4.0
            var midY = (initTL.y + initTR.y + initBR.y + initBL.y) / 4.0

            // 1. 세로축 독립 팽창 계수 (글자 높이의 1.35배)
            val scaleY = 1.35 

            // 2. 가로축 독립 팽창 계수 및 무게중심 이동(Shift) 결정
            var scaleX = 1.0
            var shiftNorm = 0.0

            if (hasKorMark) {
                // [케이스 A] KOR 마크가 뼈대에 포함된 경우 (오분류 혹은 KOR 포함 군집)
                // 이미 텍스트가 번호판 가로의 대부분을 차지하므로 좌우 끝 여백만 살짝 덧댐
                scaleX = 1.08
                shiftNorm = 0.0 // 중심점 이동 없음
            } else {
                // [케이스 B] KOR 마크가 걸러진 순수 숫자 뼈대인 경우 (기본 로직)
                // 숫자는 가로의 약 74%를 차지하므로 1.35배 팽창
                scaleX = 1.35
                // 팽창된 마스크를 KOR 마크가 있는 좌측(-vx 방향)으로 전체 너비의 4.5%만큼 밀어줌
                shiftNorm = -0.045 * (textW * scaleX)
            }

            // 중심점 좌표 보정 (이동된 센터 기준으로 스케일링)
            midX += vx * shiftNorm
            midY += vy * shiftNorm

            val nX = -vy; val nY = vx

            // 로컬 좌표계에서 가로/세로를 독립적으로 곱해 사다리꼴 왜곡을 방어하며 팽창
            val finalPts = listOf(initTL, initTR, initBR, initBL).map { pt ->
                val dx = pt.x - midX
                val dy = pt.y - midY
                
                val localX = dx * vx + dy * vy
                val localY = dx * nX + dy * nY
                
                val scaledX = localX * scaleX
                val scaledY = localY * scaleY
                
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

                // 보정된 핏팅 중심점을 시각적으로 표시
                Imgproc.circle(debugMat, Point(midX + looseRect.x, midY + looseRect.y), 8, Scalar(0.0, 255.0, 255.0, 255.0), -1)

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 6: Decoupled Axis Estimation", listOf(
                    "Mode: 가로/세로 독립 스케일링 방어 체계 작동",
                    "Status: hasKorMark = $hasKorMark (ScaleX: $scaleX)",
                    "Result: 번호판 흰색 철판 규격에 정확히 들어맞는 핏 구현 완료"
                ), screenRatio)
                it.pauseAndShowStep("최종 단계: 독립 제어 가림막 좌표 확정", hudBmp)
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
