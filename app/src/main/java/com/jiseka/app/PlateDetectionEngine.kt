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
                "ROI Width: $roiWidth px",
                "ROI Height: $roiHeight px",
                "Rect Bounds: [L:$looseLeft, T:$looseTop, R:$looseRight, B:$looseBottom]"
            ), screenRatio)
            it.pauseAndShowStep("2단계: 컴팩트 탐색 구역 설정", hudBmp)
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
        // 🚀 [Step 5] 바운딩 박스 중심 기반 뼈대 구축
        // =====================================================================
        val tempContours = ArrayList<MatOfPoint>()
        val tempHierarchy = Mat()
        Imgproc.findContours(thresh, tempContours, tempHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        class CharData(val center: Point, val width: Double, val height: Double, val rect: Rect)
        val charList = mutableListOf<CharData>()
        val allRects = mutableListOf<Rect>()

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

        if (charList.size < 2) {
            thresh.release(); looseMat.release(); looseGray.release(); fullMat.release(); fullGray.release()
            return null
        }

        val sortedChars = charList.sortedBy { it.center.x }
        val pointsMat = MatOfPoint2f(*sortedChars.map { it.center }.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(pointsMat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        val vx = line.get(0, 0)[0]; val vy = line.get(1, 0)[0]
        val angle = Math.toDegrees(Math.atan2(vy, vx))
        
        // 1. 가로 뼈대
        val topPtsArray = sortedChars.map { Point(it.center.x, it.rect.y.toDouble()) }.toTypedArray()
        val bottomPtsArray = sortedChars.map { Point(it.center.x, it.rect.y.toDouble() + it.rect.height) }.toTypedArray()
        
        val topPts = MatOfPoint2f(*topPtsArray); val bottomPts = MatOfPoint2f(*bottomPtsArray)
        val topLineMat = Mat(); val bottomLineMat = Mat()
        
        Imgproc.fitLine(topPts, topLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        Imgproc.fitLine(bottomPts, bottomLineMat, Imgproc.DIST_L2, 0.0, 0.01, 0.01)

        val tvx = topLineMat.get(0, 0)[0]; val tvy = topLineMat.get(1, 0)[0]
        val tx0 = topLineMat.get(2, 0)[0]; val ty0 = topLineMat.get(3, 0)[0]
        
        val bvx = bottomLineMat.get(0, 0)[0]; val bvy = bottomLineMat.get(1, 0)[0]
        val bx0 = bottomLineMat.get(2, 0)[0]; val by0 = bottomLineMat.get(3, 0)[0]

        // 💡 2. 유저 요청 로직: 바운딩 박스(Rect)의 상단 중앙, 중심, 하단 중앙 3점 추출 및 피팅
        val firstChar = sortedChars.first()
        val lastChar = sortedChars.last()

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

        // 교차점 계산
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
        val initWireframePts = arrayOf(initTL, initTR, initBR, initBL)
        
        debugListener?.let {
            val debugMat = looseMat.clone()
            
            // 💡 바운딩 박스 확인용: 모든 글자에 초록색 직사각형 렌더링
            for (charData in charList) {
                Imgproc.rectangle(debugMat, charData.rect, Scalar(0.0, 255.0, 0.0, 255.0), 2)
            }

            // 뼈대 보라색 렌더링
            for (i in 0..3) Imgproc.line(debugMat, initWireframePts[i], initWireframePts[(i+1)%4], Scalar(255.0, 0.0, 255.0, 255.0), 3)
            
            // 💡 유저 요청 로직 검증용: 세로선의 기준이 된 3점(윗변 중앙, 중심, 아랫변 중앙) 하늘색 렌더링
            val testPts = arrayOf(leftTopMid, leftCenter, leftBottomMid, rightTopMid, rightCenter, rightBottomMid)
            for (pt in testPts) {
                Imgproc.circle(debugMat, pt, 6, Scalar(0.0, 255.0, 255.0, 255.0), -1)
            }

            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 3~5: Bounding Box Wireframe", listOf(
                "Method: 바운딩 박스 윗변 중앙, 중심, 아랫변 중앙 연결",
                "초록색: OpenCV가 추출한 바운딩 박스 영역",
                "하늘색: 세로선의 기준이 된 추출된 3개의 포인트"
            ), screenRatio)
            it.pauseAndShowStep("3~5단계: 바운딩 박스 기준 뼈대 및 점 확인", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }
        pointsMat.release(); line.release(); topPts.release(); bottomPts.release()
        topLineMat.release(); bottomLineMat.release()
        leftPts.release(); rightPts.release(); leftLine.release(); rightLine.release()

        // =====================================================================
        // 🚀 [Step 6 & 7] 회전 및 타이트 ROI 추출
        // =====================================================================
        val rotMat = Imgproc.getRotationMatrix2D(Point(looseRect.width / 2.0, looseRect.height / 2.0), angle, 1.0)
        val rotatedLooseMat = Mat(); val rotatedLooseGray = Mat()
        Imgproc.warpAffine(looseMat, rotatedLooseMat, rotMat, looseMat.size(), Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(looseGray, rotatedLooseGray, rotMat, looseGray.size(), Imgproc.INTER_LINEAR)

        val srcPtsMat = MatOfPoint2f(*initWireframePts)
        val rotatedPtsMat = MatOfPoint2f()
        Core.transform(srcPtsMat, rotatedPtsMat, rotMat)
        val rotatedWireframePts = rotatedPtsMat.toArray()

        val minX = rotatedWireframePts.minOf { it.x }
        val maxX = rotatedWireframePts.maxOf { it.x }
        val minY = rotatedWireframePts.minOf { it.y }
        val maxY = rotatedWireframePts.maxOf { it.y }
        
        val avgH = charList.map { it.height }.average()
        val marginX = avgH * 2.5
        val marginY = avgH * 1.5

        val tightLeft = (minX - marginX).toInt().coerceIn(0, rotatedLooseMat.cols() - 1)
        val tightRight = (maxX + marginX).toInt().coerceIn(1, rotatedLooseMat.cols())
        val tightTop = (minY - marginY).toInt().coerceIn(0, rotatedLooseMat.rows() - 1)
        val tightBottom = (maxY + marginY).toInt().coerceIn(1, rotatedLooseMat.rows())

        val tightRect = Rect(tightLeft, tightTop, tightRight - tightLeft, tightBottom - tightTop)
        val tightMat = Mat(); val tightGray = Mat()
        rotatedLooseMat.submat(tightRect).copyTo(tightMat)
        rotatedLooseGray.submat(tightRect).copyTo(tightGray)

        val tightWireframePts = rotatedWireframePts.map { Point(it.x - tightRect.x, it.y - tightRect.y) }.toTypedArray()

        debugListener?.let {
            val debugMat = rotatedLooseMat.clone()
            Imgproc.rectangle(debugMat, tightRect, Scalar(255.0, 165.0, 0.0, 255.0), 6)
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            val hudBmp = addDebugHUD(debugBmp, "Step 6~7: True Plate Bound ROI", listOf(
                "Image leveled to 0 degrees.",
                "Action: 뼈대를 감싸는 타이트한 탐색 구역 설정",
                "좌표계 오차(Mismatch) 완벽 해결."
            ), screenRatio)
            it.pauseAndShowStep("6~7단계: 수평 회전 및 타이트 탐색 영역 확정", hudBmp)
            debugMat.release(); debugBmp.recycle()
        }

        // =====================================================================
        // 🚀 [Step 8 & 9] 진짜 테두리 찾기 (이중 모폴로지)
        // =====================================================================
        val edges = Mat()
        val verticalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, 7.0))
        val horizontalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 1.0))
        
        val mask = Mat()
        val intersectMat = Mat()
        val invRotMat = Mat()

        var resultPoints: List<ImmutablePoint>? = null

        try {
            Imgproc.medianBlur(tightGray, tightGray, 3)
            Imgproc.Canny(tightGray, edges, 35.0, 100.0) 
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, verticalKernel)
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, horizontalKernel)

            // =====================================================================
            // 🚀 [Step 10] 바운딩 박스 뼈대 기반 스내핑 (Snapping)
            // =====================================================================
            val tTL = tightWireframePts[0]; val tTR = tightWireframePts[1]
            val tBR = tightWireframePts[2]; val tBL = tightWireframePts[3]

            var pTvx = tTR.x - tTL.x; var pTvy = tTR.y - tTL.y
            val tLen = hypot(pTvx, pTvy); pTvx /= tLen; pTvy /= tLen
            val pTx0 = tTL.x; val pTy0 = tTL.y

            var pBvx = tBR.x - tBL.x; var pBvy = tBR.y - tBL.y
            val bLen = hypot(pBvx, pBvy); pBvx /= bLen; pBvy /= bLen
            val pBx0 = tBL.x; val pBy0 = tBL.y

            var pLvx = tBL.x - tTL.x; var pLvy = tBL.y - tTL.y
            val lLen = hypot(pLvx, pLvy); pLvx /= lLen; pLvy /= lLen
            val pLx0 = tTL.x; val pLy0 = tTL.y

            var pRvx = tBR.x - tTR.x; var pRvy = tBR.y - tTR.y
            val rLen = hypot(pRvx, pRvy); pRvx /= rLen; pRvy /= rLen
            val pRx0 = tTR.x; val pRy0 = tTR.y

            mask.create(edges.size(), CvType.CV_8UC1)
            val imgBounds = Rect(0, 0, edges.cols(), edges.rows())

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
                    
                    p1.x = currentX - lineVx * (lineSpan / 2.0); p1.y = currentY - lineVy * (lineSpan / 2.0)
                    p2.x = currentX + lineVx * (lineSpan / 2.0); p2.y = currentY + lineVy * (lineSpan / 2.0)
                    
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
                    } else { consecutiveHits = 0 }
                    if (step == maxSteps) { snappedX = currentX; snappedY = currentY }
                }
                return Pair(snappedX, snappedY)
            }

            val vLimit = (lLen * 1.2).toInt() 
            val hLimit = (lLen * 2.2).toInt() 
            
            var tnx = -pTvy; var tny = pTvx; if (tny > 0) { tnx = -tnx; tny = -tny }
            var bnx = -pBvy; var bny = pBvx; if (bny < 0) { bnx = -bnx; bny = -bny }
            var lnx = -pLvy; var lny = pLvx; if (lnx > 0) { lnx = -lnx; lny = -lny }
            var rnx = -pRvy; var rny = pRvx; if (rnx < 0) { rnx = -rnx; rny = -rny }

            val finalTop = snapLine(pTvx, pTvy, pTx0, pTy0, tnx, tny, vLimit, tLen * 0.9)
            val finalBottom = snapLine(pBvx, pBvy, pBx0, pBy0, bnx, bny, vLimit, tLen * 0.9)
            val finalLeft = snapLine(pLvx, pLvy, pLx0, pLy0, lnx, lny, hLimit, lLen * 1.5)
            val finalRight = snapLine(pRvx, pRvy, pRx0, pRy0, rnx, rny, hLimit, lLen * 1.5)

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
                val hudBmp = addDebugHUD(debugBmp, "Step 8~10: Edge Snapping Check", listOf(
                    "Method: Canny Map + Wireframe Expansion",
                    "Action: 뼈대를 밀어내어 진짜 테두리에서 멈춤",
                    "무한선 버그 수정으로 노이즈 관통력 상승!"
                ), screenRatio)
                it.pauseAndShowStep("8~10단계: 에지 렌더링 및 스내핑 확인", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            val ptTL = getIntersect(finalTop.first, finalTop.second, pTvx, pTvy, finalLeft.first, finalLeft.second, pLvx, pLvy)
            val ptTR = getIntersect(finalTop.first, finalTop.second, pTvx, pTvy, finalRight.first, finalRight.second, pRvx, pRvy)
            val ptBR = getIntersect(finalBottom.first, finalBottom.second, pBvx, pBvy, finalRight.first, finalRight.second, pRvx, pRvy)
            val ptBL = getIntersect(finalBottom.first, finalBottom.second, pBvx, pBvy, finalLeft.first, finalLeft.second, pLvx, pLvy)

            val orderedPoints = arrayOf(ptTL, ptTR, ptBR, ptBL)

            // =====================================================================
            // 🚀 [Step 11] 기하학 정렬: 최종 극단점 원본 좌표로 역회전
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
                    "Mode: Bounding Box Midpoints Wireframe",
                    "Result: 바운딩 박스 기준 세로 정렬 완료",
                    "교차점을 원본 해상도 좌표계로 복원 완료."
                ), screenRatio)
                it.pauseAndShowStep("11단계: 최종 좌표 보정 완료", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            resultPoints = finalPts.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
            
            srcPtsMat.release(); rotatedPtsMat.release()
            rectSrcMat.release(); rectDstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            edges.release()
            horizontalKernel.release() 
            verticalKernel.release()
            
            mask.release()
            intersectMat.release()
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
