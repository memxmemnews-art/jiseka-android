package com.jiseka.app

import android.graphics.RectF

// 스레드 안전성을 보장하는 불변 포인트
data class ImmutablePoint(val x: Float, val y: Float)

// 1차 고속 필터링(바운딩 박스)과 2차 정밀 다각형을 묶어둔 앵커 데이터
data class CandidatePolygon(
    val points: List<ImmutablePoint>,
    val bounds: RectF
)
