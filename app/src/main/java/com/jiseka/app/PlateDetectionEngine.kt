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

        val cx = touchX.toDouble()
        val cy = touchY.toDouble()

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
        // 🚀 [Step 2] 초기 ROI 설정
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
                "ROI Width: $roiWidth px (화면 가로폭의 22%)",
                "ROI Height: $roiHeight px (2:1 비율 완벽 적용)",
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 컴팩트 2:1 탐색 구역 설정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 3 & 4] 1차 이진화 및 모폴로지
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
        // 🚀 [Step 5] 문자 중심점 추출 및 실패 원인 분석
        // =====================================================================
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect)
        val charList = mutableListOf<CharData>()
        val allRects = mutableListOf<Rect>() // 실패 시 원인 분석을 위해 모든 바운딩 박스 기록

        for (contour in tempContours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.area()
            val ratio = rect.height.toDouble() / max(rect.width.toDouble(), 1.0)
            
            allRects.add(rect)

            if (ratio in 1.2..4.2 && area > 120 && rect.height >= 40 && area < looseRect.area() * 0.08) {
                val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
                charList.add(CharData(center, rect.width.toDouble(), rect.height.toDouble(), rect))
            }
        }
        tempContours.forEach { it.release() }; tempHierarchy.release(); tempOpen.release(); tempClose.release()

        // 🚨 실패 원인 부검 (Post-Mortem) 디버그 화면
        if (charList.size < 2) {
            debugListener?.let {
                val debugMat = looseMat.clone()
                for (rect in allRects) Imgproc.rectangle(debugMat, rect, Scalar(255.0, 0.0, 0.0, 255.0), 2)
                for (charData in charList) Imgproc.rectangle(debugMat, charData.rect, Scalar(0.0, 255.0, 0.0, 255.0), 5)
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "🚨 FAILED: Text Core Not Found", listOf(
                    "Reason: 뼈대를 세울 유효 문자가 2개 미만(${charList.size}개)입니다.",
                    "Guide: 빨간 박스(탈락 객체)들의 비율과 찌그러짐을 확인하세요.",
                    "Tip: 세차장 조명 난반사, 도장면 물기, 파라미터(Area/Ratio) 점검 요망."
                ), screenRatio)
                it.pauseAndShowStep("❌ 탐색 실패: 문자 코어 확보 미달", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }
            
            thresh.release()
            looseMat.release(); looseGray.release()
            fullMat.release(); fullGray.release()
            return null
        }

        var angle = 0.0
        var tightLeft = 0; var tightRight = looseRect.width
        var tightTop = 0; var tightBottom = looseRect.height

        val pointsMat = MatOfPoint2f(*charList.map { it.center }.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(pointsMat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        val vx = line.get(0, 0)[0]; val vy = line.get(1, 0)[0]
        angle = Math.toDegrees(Math.atan2(vy, vx))
        
        val tempRotMat = Imgproc.getRotationMatrix2D(Point(looseRect.width / 2.0, looseRect.height / 2.0), angle, 1.0)
        val dstCenterPts = MatOfPoint2f()
        Core.transform(pointsMat, dstCenterPts, tempRotMat)
        
        val rotCenters = dstCenterPts.toArray()
        val minX = rotCenters.minOf { it.x }
        val maxX = rotCenters.maxOf { it.x }
        val avgY = rotCenters.map { it.y }.average()
        val avgH = charList.map { it.height }.average()
        val textSpreadWidth = maxX - minX
        
        val expectedHeight = (avgH * 2.8).coerceAtMost(looseRect.height.toDouble()) 
        val marginX = max(textSpreadWidth * 0.15, avgH * 1.5)

        tightLeft = (minX - marginX).toInt()
        tightRight = (maxX + marginX).toInt()
        tightTop = (avgY - expectedHeight / 2.0).toInt()
        tightBottom = (avgY + expectedHeight / 2.0).toInt()
        
        debugListener?.let {
            val debugMat = looseMat.clone()
            val x0 = line.get(2, 0)[0]; val y0 = line.get(3, 0)[0]
            val pt1 = Point(x0 - vx * 1000, y0 - vy * 1000)
            val pt2 = Point(x0 + vx * 1000, y0 + vy * 1000)
            Imgproc.line(debugMat, pt1, pt2, Scalar(0.0, 255.0, 0.0, 255.0), 3)
            for (charData in charList) Imgproc.circle(debugMat, charData.center, 5, Scalar(255.0, 0.0, 0.0, 255.0), -1)
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 3~5: Cleaned Text Fitting", listOf(
                "Valid Characters Found: ${charList.size}",
                "Fitted Line Angle: ${String.format("%.2f", angle)} deg",
                "Status: 텍스트 클러스터 정상 감지"
            ), screenRatio)
            it.pauseAndShowStep("3~5단계: 문자 탐색 및 기울기 계산", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }
        pointsMat.release(); line.release(); tempRotMat.release(); dstCenterPts.release()

        // =====================================================================
        // 🚀 [Step 6 & 7] 회전 및 타이트 ROI 추출
        // =====================================================================
        val rotMat = Imgproc.getRotationMatrix2D(Point(looseRect.width / 2.0, looseRect.height / 2.0), angle, 1.0)
        val rotatedLooseMat = Mat(); val rotatedLooseGray = Mat()
        Imgproc.warpAffine(looseMat, rotatedLooseMat, rotMat, looseMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(looseGray, rotatedLooseGray, rotMat, looseGray.size(), Imgproc.INTER_LINEAR)

        tightLeft = tightLeft.coerceIn(0, rotatedLooseMat.cols() - 1)
        tightRight = tightRight.coerceIn(1, rotatedLooseMat.cols())
        tightTop = tightTop.coerceIn(0, rotatedLooseMat.rows() - 1)
        tightBottom = tightBottom.coerceIn(1, rotatedLooseMat.rows())

        val tightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)
        val tightMat = Mat(); val tightGray = Mat()
        rotatedLooseMat.submat(tightRect).copyTo(tightMat)
        rotatedLooseGray.submat(tightRect).copyTo(tightGray)

        debugListener?.let {
            val debugMat = rotatedLooseMat.clone()
            Imgproc.rectangle(debugMat, tightRect, Scalar(255.0, 165.0, 0.0, 255.0), 6)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 6~7: True Plate Bound ROI", listOf(
                "Image leveled to 0 degrees.",
                "Tight ROI size: ${tightRect.width} x ${tightRect.height}",
                "Ready for Outer Edge Tracking."
            ), screenRatio)
            it.pauseAndShowStep("6~7단계: 수평 회전 및 타이트 탐색 영역 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 8 & 9] 진짜 테두리 찾기 (이중 구조적 모폴로지 최적화)
        // =====================================================================
        val edges = Mat()
        val verticalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, 7.0))
        val horizontalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 1.0))
        
        val mask = Mat()
        val intersectMat = Mat()
        val topLineMat = Mat()
        val bottomLineMat = Mat()
        val invRotMat = Mat()

        var resultPoints: List<ImmutablePoint>? = null

        try {
            Imgproc.medianBlur(tightGray, tightGray, 3)
            Imgproc.Canny(tightGray, edges, 35.0, 100.0) 
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, verticalKernel)
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, horizontalKernel)

            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(edges, debugMat, Imgproc.COLOR_GRAY2RGBA)
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 8~9: Fixed Canny Map", listOf(
                    "Method: medianBlur(3) + Canny(35,100)",
                    "Morphology: Vertical Close(1x7) -> Horizontal Close(3x1)",
                    "Insight: 세로 단절 우선 복구 및 과연결 부작용 제어"
                ), screenRatio)
                it.pauseAndShowStep("8~9단계: 번호판 테두리 이중 봉합 완료", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // 🚀 [Step 10] Text-Anchored Wireframe Snapping (텍스트 뼈대 스내핑)
            // =====================================================================
            val sortedChars = charList.sortedBy { it.center.x }
            val avgCharHeight = charList.map { it.height }.average()
            val finalSpreadWidth = sortedChars.last().center.x - sortedChars.first().center.x

            // 상/하단 가로줄 뼈대 (Trend Line) 추출
            val topPtsArray = sortedChars.map { Point(it.center.x, it.rect.y.toDouble()) }.toTypedArray()
            val bottomPtsArray = sortedChars.map { Point(it.center.x, it.rect.y.toDouble() + it.rect.height) }.toTypedArray()
            
            val topPts = MatOfPoint2f(*topPtsArray)
            val bottomPts = MatOfPoint2f(*bottomPtsArray)
            
            Imgproc.fitLine(topPts, topLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
            Imgproc.fitLine(bottomPts, bottomLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)

            val tvx = topLineMat.get(0, 0)[0]; val tvy = topLineMat.get(1, 0)[0]
            val tx0 = topLineMat.get(2, 0)[0]; val ty0 = topLineMat.get(3, 0)[0]
            
            val bvx = bottomLineMat.get(0, 0)[0]; val bvy = bottomLineMat.get(1, 0)[0]
            val bx0 = bottomLineMat.get(2, 0)[0]; val by0 = bottomLineMat.get(3, 0)[0]

            // 좌/우측 세로줄 뼈대 (순수 수직 팽창)
            val lvx = 0.0; val lvy = 1.0
            val lx0 = sortedChars.first().rect.x.toDouble(); val ly0 = sortedChars.first().center.y
            
            val rvx = 0.0; val rvy = 1.0
            val rx0 = sortedChars.last().rect.x.toDouble() + sortedChars.last().rect.width; val ry0 = sortedChars.last().center.y

            // 🚨 [디버그 10.1] 팽창 전 초기 뼈대(Wireframe) 상태 확인
            debugListener?.let {
                val debugMat = tightMat.clone()
                val lineLen = finalSpreadWidth / 2.0
                val hLineLen = avgCharHeight
                
                Imgproc.line(debugMat, Point(tx0 - tvx * lineLen, ty0 - tvy * lineLen), Point(tx0 + tvx * lineLen, ty0 + tvy * lineLen), Scalar(255.0, 0.0, 255.0, 255.0), 2)
                Imgproc.line(debugMat, Point(bx0 - bvx * lineLen, by0 - bvy * lineLen), Point(bx0 + bvx * lineLen, by0 + bvy * lineLen), Scalar(255.0, 0.0, 255.0, 255.0), 2)
                Imgproc.line(debugMat, Point(lx0 - lvx * hLineLen, ly0 - lvy * hLineLen), Point(lx0 + lvx * hLineLen, ly0 + lvy * hLineLen), Scalar(0.0, 255.0, 255.0, 255.0), 2)
                Imgproc.line(debugMat, Point(rx0 - rvx * hLineLen, ry0 - rvy * hLineLen), Point(rx0 + rvx * hLineLen, ry0 + rvy * hLineLen), Scalar(0.0, 255.0, 255.0, 255.0), 2)

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 10.1: Initial Wireframe", listOf(
                    "Action: 글자 중심점을 관통하는 뼈대 생성",
                    "Check: 이 선들이 번호판 기울기와 평행한가?",
                    "노이즈(그릴 등)가 섞였다면 선이 심하게 틀어집니다."
                ), screenRatio)
                it.pauseAndShowStep("10.1단계: 팽창 전 초기 뼈대 확인", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            mask.create(edges.size(), CvType.CV_8UC1)
            val imgBounds = Rect(0, 0, edges.cols(), edges.rows())

            // 💡 핵심 팽창 함수: 길이를 글자에 맞춰 최적화
            fun snapLine(
                lineVx: Double, lineVy: Double, originX: Double, originY: Double,
                nx: Double, ny: Double, maxSteps: Int, lineSpan: Double
            ): Pair<Double, Double> {
                var snappedX = originX; var snappedY = originY
                val p1 = Point(); val p2 = Point()
                
                var consecutiveHits = 0 
                val requiredHits = 2
                val safeZone = 5

                for (step in safeZone..maxSteps) {
                    val currentX = originX + nx * step
                    val currentY = originY + ny * step
                    
                    p1.x = currentX - lineVx * (lineSpan / 2.0)
                    p1.y = currentY - lineVy * (lineSpan / 2.0)
                    p2.x = currentX + lineVx * (lineSpan / 2.0)
                    p2.y = currentY + lineVy * (lineSpan / 2.0)
                    
                    Imgproc.clipLine(imgBounds, p1, p2)
                    
                    mask.setTo(Scalar(0.0))
                    Imgproc.line(mask, p1, p2, Scalar(255.0), 1)
                    Core.bitwise_and(edges, mask, intersectMat)
                    
                    val hitCount = Core.countNonZero(intersectMat)
                    val lineLength = hypot(p2.x - p1.x, p2.y - p1.y)
                    
                    if (hitCount > lineLength * 0.15) {
                        consecutiveHits++
                        if (consecutiveHits >= requiredHits) {
                            snappedX = currentX; snappedY = currentY
                            break
                        }
                    } else {
                        consecutiveHits = 0
                    }
                    
                    if (step == maxSteps) { snappedX = currentX; snappedY = currentY }
                }
                return Pair(snappedX, snappedY)
            }

            val limit = (avgCharHeight * 1.2).toInt()
            
            var tnx = -tvy; var tny = tvx
            if (tny > 0) { tnx = -tnx; tny = -tny }
            val finalTop = snapLine(tvx, tvy, tx0, ty0, tnx, tny, limit, finalSpreadWidth * 0.9)

            var bnx = -bvy; var bny = bvx
            if (bny < 0) { bnx = -bnx; bny = -bny }
            val finalBottom = snapLine(bvx, bvy, bx0, by0, bnx, bny, limit, finalSpreadWidth * 0.9)

            val finalLeft = snapLine(lvx, lvy, lx0, ly0, -1.0, 0.0, limit, avgCharHeight * 1.5)
            val finalRight = snapLine(rvx, rvy, rx0, ry0, 1.0, 0.0, limit, avgCharHeight * 1.5)

            // 🚨 [디버그 10.2] Canny 위에서 팽창이 멈춘 위치 확인
            debugListener?.let {
                val debugMat = Mat()
                Imgproc.cvtColor(edges, debugMat, Imgproc.COLOR_GRAY2RGBA)
                
                val cX1 = finalLeft.first; val cX2 = finalRight.first
                val cY1 = finalTop.second; val cY2 = finalBottom.second
                
                Imgproc.line(debugMat, Point(cX1 - 500, cY1), Point(cX2 + 500, cY1), Scalar(0.0, 255.0, 0.0, 255.0), 2)
                Imgproc.line(debugMat, Point(cX1 - 500, cY2), Point(cX2 + 500, cY2), Scalar(0.0, 255.0, 0.0, 255.0), 2)
                Imgproc.line(debugMat, Point(cX1, cY1 - 500), Point(cX1, cY2 + 500), Scalar(0.0, 255.0, 0.0, 255.0), 2)
                Imgproc.line(debugMat, Point(cX2, cY1 - 500), Point(cX2, cY2 + 500), Scalar(0.0, 255.0, 0.0, 255.0), 2)

                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 10.2: Snapped Bounds on Canny", listOf(
                    "Action: 뼈대를 밀어내어 Canny 흰색 선에 멈춤",
                    "Check: 초록색 선이 엉뚱한 노이즈에 걸려 멈췄는가?",
                    "제대로 멈췄다면 이 선들의 교차점이 최종 가림막이 됩니다."
                ), screenRatio)
                it.pauseAndShowStep("10.2단계: 에지 충돌 스내핑 결과 확인", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            fun getIntersect(
                p1: Pair<Double, Double>, v1: Pair<Double, Double>,
                p2: Pair<Double, Double>, v2: Pair<Double, Double>
            ): Point {
                val dx = p2.first - p1.first
                val dy = p2.second - p1.second
                val det = v2.first * v1.second - v2.second * v1.first
                if (abs(det) < 1e-6) return Point(p1.first, p1.second) 
                val u = (dy * v1.first - dx * v1.second) / det
                return Point(p2.first + u * v2.first, p2.second + u * v2.second)
            }

            val ptTL = getIntersect(finalTop, Pair(tvx, tvy), finalLeft, Pair(lvx, lvy))
            val ptTR = getIntersect(finalTop, Pair(tvx, tvy), finalRight, Pair(rvx, rvy))
            val ptBR = getIntersect(finalBottom, Pair(bvx, bvy), finalRight, Pair(rvx, rvy))
            val ptBL = getIntersect(finalBottom, Pair(bvx, bvy), finalLeft, Pair(lvx, lvy))

            val orderedPoints = arrayOf(ptTL, ptTR, ptBR, ptBL)
            topPts.release(); bottomPts.release()

            // =====================================================================
            // 🚀 [Step 11] 기하학 정렬: 최종 극단점 원본 이미지 좌표로 역회전 매핑
            // =====================================================================
            Imgproc.invertAffineTransform(rotMat, invRotMat)
            
            val rectRotated = orderedPoints.map { Point(it.x + tightRect.x, it.y + tightRect.y) }.toTypedArray()
            val rectSrcMat = MatOfPoint2f(*rectRotated)
            val rectDstMat = MatOfPoint2f()
            
            Core.transform(rectSrcMat, rectDstMat, invRotMat)
            val finalPts = rectDstMat.toArray().map { Point(it.x + looseRect.x, it.y + looseRect.y) }

            debugListener?.let {
                val debugMat = fullMat.clone()
                val colors = arrayOf(Scalar(255.0,0.0,0.0,255.0), Scalar(0.0,255.0,0.0,255.0), Scalar(0.0,0.0,255.0,255.0), Scalar(255.0,255.0,0.0,255.0))
                val labels = arrayOf("TL", "TR", "BR", "BL")
                
                for (i in 0..3) {
                    Imgproc.line(debugMat, finalPts[i], finalPts[(i + 1) % 4], Scalar(255.0, 255.0, 255.0, 255.0), 4)
                    Imgproc.circle(debugMat, finalPts[i], 15, colors[i], -1)
                    Imgproc.putText(debugMat, labels[i], Point(finalPts[i].x - 20, finalPts[i].y - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, colors[i], 4)
                }
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)
                
                val hudBmp = addDebugHUD(debugBmp, "Step 11: Final Export", listOf(
                    "Mode: Text Core Polygon Expansion",
                    "Result: Perspective Intact & Snapped.",
                    "교차점을 원본 해상도 좌표계로 복원 완료."
                ), screenRatio)
                it.pauseAndShowStep("11단계: 최종 좌표 보정 완료", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
            
            rectSrcMat.release()
            rectDstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            edges.release()
            horizontalKernel.release() 
            verticalKernel.release()
            
            mask.release()
            intersectMat.release()
            topLineMat.release()
            bottomLineMat.release()
            invRotMat.release()
            
            thresh.release()
            rotMat.release(); looseMat.release(); looseGray.release()
            rotatedLooseMat.release(); rotatedLooseGray.release()
            tightMat.release(); tightGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
    }
}
