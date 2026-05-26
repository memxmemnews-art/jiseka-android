package com.jiseka.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class NativeGuideView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var onGuideDropListener: ((String) -> Unit)? = null

    var currentMode: String = "FRONT"
        private set

    private val currentCorners = mutableListOf<PointF>()

    private val path = Path()
    private val strokePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    private val fillPaint = Paint().apply {
        color = Color.parseColor("#3300FF00")
        style = Paint.Style.FILL
    }
    private val cornerPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private var isDragging = false
    private var isMoved = false
    private var lastTouchX = 0f
    private var lastTouchY = 0f

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (w > 0 && h > 0) {
            // 화면 회전 등으로 인해 사이즈가 변경될 때 기존 좌표 비율 유지
            if (currentCorners.size == 4 && oldw > 0 && oldh > 0) {
                recalculateCornersOnResize(w, h, oldw, oldh)
            } else {
                // 뷰가 처음 생성되었거나 GONE에서 VISIBLE로 돌아와 크기가 확정된 경우
                applyModeGeometry()
            }
        }
    }

    fun setMode(mode: String) {
        this.currentMode = mode
        applyModeGeometry()
    }

    // 뷰 크기가 준비되지 않았으면 안전하게 보류 (크래시 방지 및 라이프사이클 독립성 확보)
    private fun applyModeGeometry() {
        val w = width.toFloat()
        val h = height.toFloat()

        if (w <= 0f || h <= 0f) return

        currentCorners.clear()
        
        when (currentMode) {
            "FRONT" -> currentCorners.addAll(listOf(
                PointF(w * 0.15f, h * 0.42f), PointF(w * 0.85f, h * 0.42f),
                PointF(w * 0.85f, h * 0.58f), PointF(w * 0.15f, h * 0.58f)
            ))
            "PASSENGER" -> currentCorners.addAll(listOf(
                PointF(w * 0.15f, h * 0.44f), PointF(w * 0.85f, h * 0.40f),
                PointF(w * 0.85f, h * 0.56f), PointF(w * 0.15f, h * 0.60f)
            ))
            "DRIVER" -> currentCorners.addAll(listOf(
                PointF(w * 0.15f, h * 0.40f), PointF(w * 0.85f, h * 0.44f),
                PointF(w * 0.85f, h * 0.60f), PointF(w * 0.15f, h * 0.56f)
            ))
        }
        invalidate()
    }

    private fun recalculateCornersOnResize(newW: Int, newH: Int, oldW: Int, oldH: Int) {
        if (oldW <= 0 || oldH <= 0 || currentCorners.size != 4) return
        val ratioX = newW.toFloat() / oldW.toFloat()
        val ratioY = newH.toFloat() / oldH.toFloat()
        for (point in currentCorners) {
            point.x *= ratioX
            point.y *= ratioY
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (currentCorners.size != 4) return

        path.reset()
        path.moveTo(currentCorners[0].x, currentCorners[0].y)
        path.lineTo(currentCorners[1].x, currentCorners[1].y)
        path.lineTo(currentCorners[2].x, currentCorners[2].y)
        path.lineTo(currentCorners[3].x, currentCorners[3].y)
        path.close()

        canvas.drawPath(path, fillPaint)
        canvas.drawPath(path, strokePaint)

        for (point in currentCorners) {
            canvas.drawCircle(point.x, point.y, 10f, cornerPaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (currentCorners.isEmpty()) return false

        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                // 터치 필터링 유지: 가이드 박스 영역(+60px 마진) 내부 터치시에만 이벤트를 가로챔
                val minX = currentCorners.minOf { it.x } - 60f
                val maxX = currentCorners.maxOf { it.x } + 60f
                val minY = currentCorners.minOf { it.y } - 60f
                val maxY = currentCorners.maxOf { it.y } + 60f

                if (x in minX..maxX && y in minY..maxY) {
                    isDragging = true
                    isMoved = false
                    lastTouchX = x
                    lastTouchY = y
                    return true
                }
                return false
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDragging) {
                    isMoved = true
                    val dx = x - lastTouchX
                    val dy = y - lastTouchY

                    // UI 레벨에서는 화면 밖으로 넘어가도 부드럽게 드래그 되도록 제한 해제 (자유도 100%)
                    for (point in currentCorners) {
                        point.x += dx
                        point.y += dy
                    }

                    lastTouchX = x
                    lastTouchY = y
                    invalidate()
                    return true
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (isDragging) {
                    isDragging = false
                    if (isMoved) {
                        onGuideDropListener?.invoke(currentMode)
                    }
                    return true
                }
            }
        }
        return super.onTouchEvent(event)
    }

    fun getCorners(): List<PointF> {
        return currentCorners.toList()
    }

    fun setCorners(newCorners: List<PointF>) {
        if (newCorners.size != 4) return
        currentCorners.clear()
        currentCorners.addAll(newCorners)
        invalidate()
    }
}
