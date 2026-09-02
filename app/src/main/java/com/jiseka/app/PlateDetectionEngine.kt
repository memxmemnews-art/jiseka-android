package com.jiseka.app

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max

object PlateDetectionEngine {

    interface DetectionDebugListener {
        fun pauseAndShowStep(stageName: String, bitmap: Bitmap, title: String, logs: List<String>)
    }

    data class SeedCropResult(
        val offsetX: Int,
        val offsetY: Int,
        val croppedBitmap: Bitmap,
        val roiRect: Rect
    )

    private fun getSafeRect(x: Int, y: Int, w: Int, h: Int, maxW: Int, maxH: Int): Rect {
        val safeX = x.coerceIn(0, maxW - 1)
        val safeY = y.coerceIn(0, maxH - 1)
        val safeW = w.coerceAtMost(maxW - safeX)
        val safeH = h.coerceAtMost(maxH - safeY)
        return Rect(safeX, safeY, safeW, safeH)
    }

    // 1. 사용자가 터치한 곳 주변을 크롭하여 AI에게 넘겨줄 이미지를 준비하는 함수 (해상도 방어 로직 추가)
    fun prepareWideCrop(fullBitmap: Bitmap, touchX: Float, touchY: Float): SeedCropResult {
        // 비율 기반 기본 크기 계산 (가로 25%, 세로 15%)
        var cropW = (fullBitmap.width * 0.25f).toInt()
        var cropH = (fullBitmap.height * 0.15f).toInt()

        // ⭐️ 최소 물리적 픽셀 크기 보장 (번호판이 온전히 담길 수 있는 최소한의 공간)
        val minPhysicalWidth = 400 
        val minPhysicalHeight = 200

        cropW = max(cropW, minPhysicalWidth).coerceAtMost(fullBitmap.width)
        cropH = max(cropH, minPhysicalHeight).coerceAtMost(fullBitmap.height)

        val safeRect = getSafeRect(
            (touchX - cropW / 2).toInt(),
            (touchY - cropH / 2).toInt(),
            cropW,
            cropH,
            fullBitmap.width, fullBitmap.height
        )

        val croppedBitmap = Bitmap.createBitmap(fullBitmap, safeRect.x, safeRect.y, safeRect.width, safeRect.height)
        return SeedCropResult(safeRect.x, safeRect.y, croppedBitmap, safeRect)
    }

    // 2. AI가 찾은 번호판 바운딩 박스를 받아, OpenCV로 정밀한 4개의 꼭지점을 스냅하는 함수
    suspend fun processWithMLKitResult(
        fullBitmap: Bitmap, 
        aiGlobalBox: android.graphics.Rect, 
        debugListener: DetectionDebugListener? = null
    ): List<ImmutablePoint>? {
        
        val fullMat = Mat()
        val fullGray = Mat()
        Utils.bitmapToMat(fullBitmap, fullMat)
        Imgproc.cvtColor(fullMat, fullGray, Imgproc.COLOR_RGBA2GRAY)

        // AI가 찾은 박스를 상하좌우 15% 팽창 (번호판 끝부분 모서리가 잘리지 않도록 안전 공간 확보)
        val expandW = (aiGlobalBox.width() * 0.15f).toInt()
        val expandH = (aiGlobalBox.height() * 0.15f).toInt()
        val safeRoi = getSafeRect(
            aiGlobalBox.left - expandW,
            aiGlobalBox.top - expandH,
            aiGlobalBox.width() + expandW * 2,
            aiGlobalBox.height() + expandH * 2,
            fullMat.cols(), fullMat.rows()
        )

        // 해당 영역만 잘라내어 Canny 엣지 및 모폴로지(노이즈 제거) 연산 수행
        val roiGray = Mat()
        fullGray.submat(safeRoi).copyTo(roiGray)

        val edges = Mat()
        Imgproc.GaussianBlur(roiGray, edges, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(edges, edges, 50.0, 150.0) // 엣지 추출
        
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        // 윤곽선(Contours) 찾기
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        var bestCorners: List<Point>? = null
        var maxArea = 0.0
        
        // 최소한 팽창된 전체 영역의 15% 이상 크기를 가져야 번호판 테두리로 인정
        val minAreaThreshold = (safeRoi.width * safeRoi.height) * 0.15 

        for (contour in contours) {
            val peri = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            val approx = MatOfPoint2f()
            // 다각형 근사화 (0.03 ~ 0.05 사이값이 사각형 모서리를 가장 잘 땀)
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, 0.03 * peri, true)

            // 4개의 꼭지점을 가진 볼록 다각형(사각형)인지 확인
            if (approx.toArray().size == 4 && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                val area = Imgproc.contourArea(approx)
                // 가장 면적이 큰 사각형을 번호판 외곽선으로 채택
                if (area > maxArea && area > minAreaThreshold) {
                    maxArea = area
                    bestCorners = approx.toArray().toList()
                }
            }
            approx.release()
        }

        roiGray.release(); edges.release(); kernel.release(); hierarchy.release()
        contours.forEach { it.release() }

        // 최종 꼭지점 좌표 결정 (로컬 ROI 좌표 -> 글로벌 좌표로 변환)
        val finalCorners = if (bestCorners == null) {
            // 만약 OpenCV가 뚜렷한 사각형 테두리를 찾지 못했다면, 
            // AI가 찾았던 원본 바운딩 박스를 그대로 4개의 꼭지점으로 변환하여 사용 (안전망)
            listOf(
                Point(aiGlobalBox.left.toDouble(), aiGlobalBox.top.toDouble()),
                Point(aiGlobalBox.right.toDouble(), aiGlobalBox.top.toDouble()),
                Point(aiGlobalBox.right.toDouble(), aiGlobalBox.bottom.toDouble()),
                Point(aiGlobalBox.left.toDouble(), aiGlobalBox.bottom.toDouble())
            )
        } else {
            // OpenCV가 찾은 꼭지점을 (좌상, 우상, 우하, 좌하) 순서로 정렬하고 글로벌 좌표로 이동
            sortCorners(bestCorners).map { 
                Point(it.x + safeRoi.x, it.y + safeRoi.y) 
            }
        }

        val globalPts = finalCorners.map {
            ImmutablePoint(it.x.toFloat(), it.y.toFloat())
        }

        // 디버그 리스너: 화면에 결과 그려주기
        debugListener?.let {
            val debugMat = fullMat.clone()
            
            // 1. AI가 최초에 잡아준 바운딩 박스 (노란색 사각형)
            val cvRect = Rect(aiGlobalBox.left, aiGlobalBox.top, aiGlobalBox.width(), aiGlobalBox.height())
            Imgproc.rectangle(debugMat, cvRect, Scalar(0.0, 255.0, 255.0, 255.0), 3)
            
            // 2. OpenCV가 정밀하게 다듬은 최종 4개 꼭지점 (초록색 선, 빨간색 점)
            for (i in 0..3) {
                val pt1 = Point(globalPts[i].x.toDouble(), globalPts[i].y.toDouble())
                val pt2 = Point(globalPts[(i+1)%4].x.toDouble(), globalPts[(i+1)%4].y.toDouble())
                Imgproc.line(debugMat, pt1, pt2, Scalar(0.0, 255.0, 0.0, 255.0), 5)
                Imgproc.circle(debugMat, pt1, 10, Scalar(255.0, 0.0, 0.0, 255.0), -1)
            }
            
            val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(debugMat, debugBmp)
            
            val logMessage = if (bestCorners != null) "-> 정밀 엣지 기반 외곽선 렌더링 완료" else "-> [경고] 정밀 외곽선을 찾지 못해 AI 영역을 그대로 사용합니다"
            it.pauseAndShowStep("디버그: 꼭지점 정밀 스냅", debugBmp, "AI 박스 내부 정밀 모서리 탐색", listOf(logMessage))
            
            debugMat.release(); debugBmp.recycle()
        }

        fullMat.release(); fullGray.release()
        return globalPts
    }

    // 꼭지점이 뒤틀리지 않도록 좌상(TL), 우상(TR), 우하(BR), 좌하(BL) 순서로 정렬하는 유틸리티
    private fun sortCorners(pts: List<Point>): List<Point> {
        val sumSorted = pts.sortedBy { it.x + it.y }
        val tl = sumSorted.first()
        val br = sumSorted.last()

        val remaining = pts.filter { it != tl && it != br }
        val diffSorted = remaining.sortedBy { it.y - it.x }
        val tr = diffSorted.first()
        val bl = diffSorted.last()

        return listOf(tl, tr, br, bl)
    }
}
