package com.jiseka.app

import android.graphics.RectF

// 스레드 안전성을 보장하는 불변 포인트
data class ImmutablePoint(val x: Float, val y: Float)

// 다각형의 점들과 바운딩 박스를 묶어둔 앵커 데이터 (점의 개수는 4~10개로 유동적)
data class CandidatePolygon(
    val points: List<ImmutablePoint>,
    val bounds: RectF
)
