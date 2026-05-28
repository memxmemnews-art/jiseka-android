package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

// 스레드 안전성을 보장하는 불변 포인트
data class ImmutablePoint(val x: Float, val y: Float)

// 1차 고속 필터링(바운딩 박스)과 2차 정밀 다각형을 묶어둔 앵커 데이터
data class CandidatePolygon(
    val points: List<ImmutablePoint>,
    val bounds: RectF
)

object PlateDetectionEngine {

    /**
     * 1. 전체 이미지 대상: 1차 다각형 후보군 사전 계산
     */
    fun precalculateGeometryCandidates(fullBitmap: Bitmap): List<CandidatePolygon> {
        val candidates = mutableListOf<CandidatePolygon>()
        val mat = Mat(); val grayMat = Mat(); val bilateralMat = Mat()
        val tophatMat = Mat(); val edgeMat = Mat(); val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        var topHatKernel: Mat? = null; var morphKernel: Mat? = null
        var clahe: org.opencv.imgproc.CLAHE? = null

        try {
            Utils.bitmapToMat(fullBitmap, mat)
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY)
            
            // 국부 조도 보정
            clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(grayMat, grayMat)
            
            // 양방향 필터 (엣지 보존)
            Imgproc.bilateralFilter(grayMat, bilateralMat, 5, 20.0, 20.0)
            
            // Top-Hat 변환 (그림자/빛반사 전역 평활화)
            topHatKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 9.0))
            Imgproc.morphologyEx(bilateralMat, tophatMat, Imgproc.MORPH_TOPHAT, topHatKernel)
            
            // 동적 Canny
            val meanVal = Core.mean(tophatMat).`val`[0]
            val sigma = 0.33
            val lowerThresh = max(0.0, (1.0 - sigma) * meanVal)
            val upperThresh = min(255.0, (1.0 + sigma) * meanVal)
            Imgproc.Canny(tophatMat, edgeMat, lowerThresh, upperThresh)
            
            // 선 닫힘 유도
            morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 3.0))
            Imgproc.morphologyEx(edgeMat, edgeMat, Imgproc.MORPH_CLOSE, morphKernel)
            
            Imgproc.findContours(edgeMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val minPlateHeight = fullBitmap.height * 0.025

            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                if (rect.height < minPlateHeight) continue
                
                // 1차 다각형 추출 (이진 탐색)
                val approxPoints = extractRobustQuadrilateral(contour) ?: continue
                
                // 2차 기하학적 사전 지식 필터링 (볼록성, 내각, 투영 종횡비 검증)
                val pointArray = approxPoints.toArray()
                if (!isValidLicensePlateGeometry(pointArray)) {
                    approxPoints.release()
                    continue
                }

                // 3차 서브픽셀 미세 보정
                val refinedPoints = applySubPixelCorrection(grayMat, approxPoints)
                approxPoints.release()

                val immutablePoints = refinedPoints.map { ImmutablePoint(it.x.toFloat(), it.y.toFloat()) }
                val xCoords = immutablePoints.map { it.x }
                val yCoords = immutablePoints.map { it.y }
                
                val bounds = RectF(
                    xCoords.minOrNull() ?: 0f,
                    yCoords.minOrNull() ?: 0f,
                    xCoords.maxOrNull() ?: 0f,
                    yCoords.maxOrNull() ?: 0f
                )
                
                candidates.add(CandidatePolygon(immutablePoints, bounds))
            }
            return candidates
        } finally {
            mat.release(); grayMat.release(); bilateralMat.release()
            tophatMat.release(); edgeMat.release(); hierarchy.release()
            contours.forEach { it.release() }
            topHatKernel?.release(); morphKernel?.release()
            clahe?.collectGarbage()
        }
    }

    /**
     * 2. 관심 영역(ROI) 대상: 십자선 정밀 추적 및 롱프레스(Dwell) 대응 복원
     */
    fun refineAnchoredPolygon(fullBitmap: Bitmap, anchorPolygon: List<ImmutablePoint>, targetLevel: Int): List<ImmutablePoint>? {
        val padding = if (targetLevel == 1) 40f else 20f 
        
        val minX = (anchorPolygon.minOfOrNull { it.x } ?: 0f) - padding
        val minY = (anchorPolygon.minOfOrNull { it.y } ?: 0f) - padding
        val maxX = (anchorPolygon.maxOfOrNull { it.x } ?: 0f) + padding
        val maxY = (anchorPolygon.maxOfOrNull { it.y } ?: 0f) + padding

        val safeRect = android.graphics.Rect(
            max(0, minX.toInt()), max(0, minY.toInt()),
            min(fullBitmap.width, maxX.toInt()), min(fullBitmap.height, maxY.toInt())
        )
        
        if (safeRect.width() <= 1 || safeRect.height() <= 1) return null

        val roiBitmap = Bitmap.createBitmap(fullBitmap, safeRect.left, safeRect.top, safeRect.width(), safeRect.height())

        val roiMat = Mat(); val roiGray = Mat(); val roiEdge = Mat()
        val roiContours = ArrayList<MatOfPoint>()
        var bestGlobalPoints: List<ImmutablePoint>? = null
        
        try {
            Utils.bitmapToMat(roiBitmap, roiMat)
            Imgproc.cvtColor(roiMat, roiGray, Imgproc.COLOR_RGBA2GRAY)
            
            // ROI 영역에 대해서도 CLAHE와 Top-Hat 적용
            val clahe = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
            clahe.apply(roiGray, roiGray)
            
            // Dwell(targetLevel 1) 시 Canny 하한선을 강제로 낮춰 숨겨진 엣지까지 모두 끌어
