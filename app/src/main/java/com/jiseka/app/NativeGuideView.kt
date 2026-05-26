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
            if (currentCorners.size == 4 && oldw > 0 && oldh > 0) {
                recalculateCornersOnResize(w, h, oldw, oldh)
            } else {
                applyModeGeometry()
            }
        }
    }

    fun setMode(mode: String) {
        this.currentMode = mode
        applyModeGeometry()
    }

    private fun applyModeGeometry() {
        val w = width.toFloat()
        val h = height.toFloat()

        if (w <= 0f || h <= 0f) return

        currentCorners.clear()
        
        // 가이드 박스 크기: 화면 너비의 30% 수준
        val boxW = w * 0.3f 
        val boxH = boxW / 4.7f // 실제 자동차 번호판 비율 적용
        val cx = w / 2f
        val cy = h / 2f
        
        when (currentMode) {
            "FRONT" -> currentCorners.addAll(listOf(
                PointF(cx - boxW/2, cy - boxH/2), PointF(cx + boxW/2, cy - boxH/2),
                PointF(cx + boxW/2, cy + boxH/2), PointF(cx - boxW/2, cy + boxH/2)
            ))
            "PASSENGER" -> currentCorners.addAll(listOf(
                PointF(cx - boxW/2, cy - boxH/2 * 0.7f), PointF(cx + boxW/2, cy - boxH/2 * 1.3f),
                PointF(cx + boxW/2, cy + boxH/2 * 1.3f), PointF(cx - boxW/2, cy + boxH/2 * 0.7f)
            ))
            "DRIVER" -> currentCorners.addAll(listOf(
                PointF(cx - boxW/2, cy - boxH/2 * 1.3f), PointF(cx + boxW/2, cy - boxH/2 * 0.7f),
                PointF(cx + boxW/2, cy + boxH/2 * 0.7f), PointF(cx - boxW/2, cy + boxH/2 * 1.3f)
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
