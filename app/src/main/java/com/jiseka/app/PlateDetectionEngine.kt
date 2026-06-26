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
            textSize = 35f
            isAntiAlias = true
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        val paddingX = 60f
        val lineHeight = 50f
        val maxTextWidth = canvasWidth - (paddingX * 2)
        var currentY = 80f 
        
        paint.color = Color.YELLOW
        paint.isFakeBoldText = true
        paint.textSize = 42f
        currentY = drawTextWithWrap(canvas, title, paddingX, currentY, paint, maxTextWidth, lineHeight)

        currentY += 20f 

        paint.color = Color.WHITE
        paint.isFakeBoldText = false
        paint.textSize = 32f
        for (log in logs) {
            if (log.startsWith("->") || log.startsWith("[경고]")) {
                paint.color = Color.parseColor("#FF5555")
            } else if (log.startsWith("[진단")) {
                paint.color = Color.parseColor("#55FF55")
            } else {
                paint.color = Color.WHITE
            }
            currentY = drawTextWithWrap(canvas, log, paddingX, currentY, paint, maxTextWidth, lineHeight)
        }

        val textBottom = currentY + 30f 
        val margin = 40f
        
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
                "Touch Point X: ${cx.toInt()} px, Y: ${cy.toInt()} px"
            ), screenRatio)
            it.pauseAndShowStep("1단계: 터치 좌표 매핑", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

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
            val hudBmp = addDebugHUD(debugBmp, "Step 2: Downsized ROI", listOf(
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 컴팩트 탐색 구역 설정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        val thresh = Mat()
        Imgproc.medianBlur(looseGray, looseGray, 3)
        Imgproc.GaussianBlur(looseGray, thresh, Size(5.0, 5.0), 0.0)
        Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, 7.0)

        // 모폴로지 2x2/3x3 유지 (글자 보존력 극대화)
        val tempOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        val tempClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, tempOpen)
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, tempClose)
        
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect)
        val charList = mutableListOf<CharData>()
        val rejectedList = mutableListOf<CharData>() 

        val step3_5_logs = mutableListOf<String>()
        var failReason = ""

        step3_5_logs.add("[진단 1] 모폴로지 및 비율 검증")
        step3_5_logs.add(" -> 발견된 덩어리(Contours): ${tempContours.size}개")

        // =====================================================================
        // 💡 [핵심 수정] 동적 최소 크기 기준 설정 (그릴 등 노이즈 덩어리 원천 차단)
        // =====================================================================
        val minArea = max(150.0, looseRect.area() * 0.001)   // ROI 면적의 최소 0.1% 이상
        val minHeight = max(20.0, looseRect.height * 0.08) // ROI 높이의 최소 8% 이상

        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
            val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
            
            if (area > minArea && area < looseRect.area() * 0.08) {
                if (ratio in 0.9..5.5 && rect.height >= minHeight) {
                    charList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect)) 
                } else if (ratio in 0.25..1.5 && rect.height >= minHeight * 0.75) {
                    rejectedList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect)) 
                }
            }
        }
        
        step3_5_logs.add(" -> 1차 통과 후보: ${charList.size}개")

        var sortedChars = mutableListOf<CharData>()
        var validChars = mutableListOf<CharData>()
        var clusters = mutableListOf<MutableList<CharData>>()

        if (charList.isEmpty()) {
            failReason = "원인 1: 1차 비율 검증에서 글자 전멸 (떡짐/지워짐)"
        } else {
            sortedChars = charList.sortedBy { it.center.x }.toMutableList()
            val rescueCandidates = mutableListOf<CharData>()

            for (i in 0 until sortedChars.size - 1) {
                val leftChar = sortedChars[i]
                val rightChar = sortedChars[i + 1]
                val gapX = rightChar.center.x - leftChar.center.x
                val avgW = (leftChar.width + rightChar.width) / 2.0

                if (gapX > avgW * 1.2) { 
                    val avgH = (leftChar.height + rightChar.height) / 2.0
                    val avgY = (leftChar.center.y + rightChar.center.y) / 2.0
                    
                    val rescuers = rejectedList.filter { r ->
                        r.center.x > leftChar.center.x + leftChar.width * 0.4 && 
                        r.center.x < rightChar.center.x - rightChar.width * 0.4 && 
                        abs(r.center.y - avgY) < avgH * 0.5 
                    }
                    rescueCandidates.addAll(rescuers) 
                }
            }
            
            sortedChars.addAll(rescueCandidates)
            sortedChars = sortedChars.sortedBy { it.center.x }.toMutableList()

            val gaps = (1 until sortedChars.size).map { sortedChars[it].center.x - sortedChars[it - 1].center.x }
            val medianGap = if (gaps.isNotEmpty()) gaps.sorted()[gaps.size / 2] else 0.0
            val avgWidth = sortedChars.map { it.width }.average()
            val maxGap = max(medianGap * 1.7, avgWidth * 1.8) 

            var currentCluster = mutableListOf(sortedChars.first())
            
            for (i in 1 until sortedChars.size) {
                val curr = sortedChars[i]
                val prev = sortedChars[i - 1]
                val gap = curr.center.x - prev.center.x
                val yDiff = abs(curr.center.y - prev.center.y)
                val avgH = (prev.height + curr.height) / 2.0

                val localMaxGap = if (gap > maxGap && gap < avgWidth * 3.5 && yDiff < avgH * 0.45) {
                    avgWidth * 3.5 
                } else {
                    maxGap
                }

                // 1. 기본 검증 (가장 엄격한 45% 단차 허용)
                var isSplit = gap > localMaxGap || yDiff > avgH * 0.45

                // 2. 동적 사선 예외 허용 (Slope Consistency)
                if (isSplit && gap < avgWidth * 1.3 && yDiff <= avgH * 0.85 && currentCluster.size >= 2) {
                    val prev1 = currentCluster.last()
                    val prev2 = currentCluster[currentCluster.size - 2]
                    
                    val slope1 = (prev1.center.y - prev2.center.y) / max(prev1.center.x - prev2.center.x, 1.0)
                    val slope2 = (curr.center.y - prev1.center.y) / max(curr.center.x - prev1.center.x, 1.0)
                    
                    if (abs(slope1 - slope2) < 0.2) {
                        isSplit = false 
                    }
                }

                if (isSplit) {
                    clusters.add(currentCluster) 
                    currentCluster = mutableListOf(curr) 
                } else {
                    currentCluster.add(curr)
                }
            }
            clusters.add(currentCluster)

            validChars = (clusters.maxByOrNull { it.size } ?: sortedChars).toMutableList()
            step3_5_logs.add("[진단 2] 사선 흐름(Slope) 동적 군집화")
            step3_5_logs.add(" -> 군집화 후 최대 그룹 사이즈: ${validChars.size}개")

            if (validChars.size < 2) {
                failReason = "원인 2: Y축 앵글 왜곡 또는 거리가 너무 멀어 뼈대 토막남"
            } else {
                val pointsMatTemp = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
                val lineTemp = Mat()
                Imgproc.fitLine(pointsMatTemp, lineTemp, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
                
                val vxTemp = lineTemp.get(0, 0)[0]; val vyTemp = lineTemp.get(1, 0)[0]
                val x0Temp = lineTemp.get(2, 0)[0]; val y0Temp = lineTemp.get(3, 0)[0]
                pointsMatTemp.release(); lineTemp.release()

                val A = vyTemp; val B = -vxTemp; val C = vxTemp * y0Temp - vyTemp * x0Temp
                val denominator = hypot(A, B)
                val localAvgHeight = validChars.map { it.height }.average()

                val iterator = validChars.iterator()
                while (iterator.hasNext()) {
                    val charData = iterator.next()
                    val dist = abs(A * charData.center.x + B * charData.center.y + C) / denominator
                    if (dist > localAvgHeight * 0.20) {
                        iterator.remove()
                    }
                }
                
                step3_5_logs.add(" -> 직선(fitLine) 검증 후: ${validChars.size}개 생존")

                if (validChars.size < 2) {
                    failReason = "직선 배열 검증 과정에서 글자 삭제됨"
                } else {
                    step3_5_logs.add("[진단 3] 2-Track KOR 마크 검출")
                    
                    val firstChar = validChars.first() 
                    val roi = looseMat.submat(firstChar.rect)
                    val meanColor = Core.mean(roi)
                    roi.release() 
                    
                    val r = meanColor.`val`[0].toInt()
                    val g = meanColor.`val`[1].toInt()
                    val b = meanColor.`val`[2].toInt()
                    step3_5_logs.add(" -> 첫 글자 RGB 스캔: R:$r, G:$g, B:$b")
                    
                    if (b > r + 25 && b > g + 15) {
                        validChars.removeAt(0) 
                        step3_5_logs.add(" -> [완벽 차단] 찐파랑 확인, KOR 마크로 정상 처리됨")
                    } else {
                        val checkW = firstChar.rect.width.toInt() * 2
                        val leftX = max(0, firstChar.rect.x - checkW)
                        val scanW = firstChar.rect.x - leftX
                        if (scanW > 10) {
                            val leftRoi = looseMat.submat(Rect(leftX, firstChar.rect.y, scanW, firstChar.rect.height))
                            val leftMean = Core.mean(leftRoi)
                            leftRoi.release()
                            if (leftMean.`val`[2] > leftMean.`val`[0] + 10 && leftMean.`val`[2] > leftMean.`val`[1] + 5) {
                                step3_5_logs.add(" -> [완벽 차단] 첫 글자는 보호, 사각지대에서 KOR 흔적 발견")
                            }
                        }
                    }

                    if (validChars.size < 2) {
                        failReason = "원인 3: KOR 처리 과정 오류"
                    } else {
                        step3_5_logs.add(" -> 뼈대 유지 성공 (최종 ${validChars.size}개)")
                    }
                }
            }
        }

        debugListener?.let {
            val debugMat = looseMat.clone()
            Imgproc.drawContours(debugMat, tempContours, -1, Scalar(150.0, 150.0, 150.0, 200.0), 1)
            
            for (charData in rejectedList) {
                Imgproc.rectangle(debugMat, charData.rect, Scalar(80.0, 80.0, 80.0, 255.0), 1)
            }
            for (charData in charList) {
                Imgproc.rectangle(debugMat, charData.rect, Scalar(255.0, 0.0, 0.0, 255.0), 2)
            }
            if (validChars.isNotEmpty()) {
                for (i in 0 until validChars.size) {
                    Imgproc.rectangle(debugMat, validChars[i].rect, Scalar(0.0, 255.0, 0.0, 255.0), 3)
                    if (i > 0) {
                        Imgproc.line(debugMat, validChars[i-1].center, validChars[i].center, Scalar(255.0, 0.0, 255.0, 255.0), 2)
                    }
                }
            }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val title = if (failReason.isNotEmpty()) "3~5단계: 분석 중단 ($failReason)" else "3~5단계: 정상 뼈대 구축"
            val hudBmp = addDebugHUD(debugBmp, title, step3_5_logs, screenRatio)
            
            it.pauseAndShowStep("3~5단계: 디버그 분석", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        tempContours.forEach { it.release() }; tempHierarchy.release(); tempOpen.release(); tempClose.release()

        if (failReason.isNotEmpty() || validChars.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        val pointsMat = MatOfPoint2f(*validChars.map { it.center }.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(pointsMat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        
        var vx = line.get(0, 0)[0]; var vy = line.get(1, 0)[0]
        if (vx < 0) { vx = -vx; vy = -vy } 
        
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

        val leftTopEdge = Point(validChars.first().rect.x.toDouble(), validChars.first().rect.y.toDouble())
        val leftMidEdge = Point(validChars.first().rect.x.toDouble(), validChars.first().center.y)
        val leftBottomEdge = Point(validChars.first().rect.x.toDouble(), validChars.first().rect.y + validChars.first().rect.height.toDouble())

        val rightX = validChars.last().rect.x + validChars.last().rect.width.toDouble()
        val rightTopEdge = Point(rightX, validChars.last().rect.y.toDouble())
        val rightMidEdge = Point(rightX, validChars.last().center.y)
        val rightBottomEdge = Point(rightX, validChars.last().rect.y + validChars.last().rect.height.toDouble())

        val leftPts = MatOfPoint2f(leftTopEdge, leftMidEdge, leftBottomEdge)
        val leftLine = Mat()
        Imgproc.fitLine(leftPts, leftLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var lvx = leftLine.get(0, 0)[0]; var lvy = leftLine.get(1, 0)[0]
        if (lvy < 0) { lvx = -lvx; lvy = -lvy }
        val lx0 = validChars.first().rect.x.toDouble()
        val ly0 = validChars.first().center.y

        val rightPts = MatOfPoint2f(rightTopEdge, rightMidEdge, rightBottomEdge)
        val rightLine = Mat()
        Imgproc.fitLine(rightPts, rightLine, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        var rvx = rightLine.get(0, 0)[0]; var rvy = rightLine.get(1, 0)[0]
        if (rvy < 0) { rvx = -rvx; rvy = -rvy }
        val rx0 = rightX
        val ry0 = validChars.last().center.y

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

        pointsMat.release(); line.release(); topPts.release(); bottomPts.release()
        topLineMat.release(); bottomLineMat.release()
        leftPts.release(); rightPts.release(); leftLine.release(); rightLine.release()

        var resultPoints: List<ImmutablePoint>? = null

        try {
            val midX = (initTL.x + initTR.x + initBR.x + initBL.x) / 4.0
            val midY = (initTL.y + initTR.y + initBR.y + initBL.y) / 4.0

            val scaleY = 1.35 
            val scaleX = 1.35 

            val nX = -vy; val nY = vx

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

                Imgproc.circle(debugMat, Point(midX + looseRect.x, midY + looseRect.y), 8, Scalar(0.0, 255.0, 255.0, 255.0), -1)

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 6: Symmetric Scaling", listOf(
                    "Mode: 겉 테두리 기준 대칭 팽창 완료",
                    "Status: Shift 0.0"
                ), screenRatio)
                it.pauseAndShowStep("최종 단계: 대칭 팽창 가림막 좌표 확정", hudBmp)
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
