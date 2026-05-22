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

    private var isInitialized = false
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
    private var lastTouchX = 0f
    private var lastTouchY = 0f

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (!isInitialized && w > 0 && h > 0) {
            isInitialized = true
            setMode(currentMode)
        } else if (isInitialized && (w != oldw || h != oldh) && w > 0 && h > 0) {
            recalculateCornersOnResize(w, h, oldw, oldh)
        }
    }

    fun setMode(mode: String) {
        this.currentMode = mode
        if (!isInitialized) return

        val w = width.toFloat()
        val h = height.toFloat()
        currentCorners.clear()
        
        when (mode) {
            "FRONT" -> currentCorners.addAll(listOf(
                PointF(w * 0.1f, h * 0.3f), PointF(w * 0.9f, h * 0.3f),
                PointF(w * 0.9f, h * 0.7f), PointF(w * 0.1f, h * 0.7f)
            ))
            "PASSENGER" -> currentCorners.addAll(listOf(
                PointF(w * 0.2f, h * 0.4f), PointF(w * 0.8f, h * 0.2f),
                PointF(w * 0.8f, h * 0.8f), PointF(w * 0.2f, h * 0.9f)
            ))
            "DRIVER" -> currentCorners.addAll(listOf(
                PointF(w * 0.2f, h * 0.2f), PointF(w * 0.8f, h * 0.4f),
                PointF(w * 0.8f, h * 0.9f), PointF(w * 0.2f, h * 0.8f)
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
        if (!isInitialized || currentCorners.size != 4) return

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
        if (!isInitialized || currentCorners.isEmpty()) return false

        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDragging = true
                lastTouchX = x
                lastTouchY = y
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDragging) {
                    val dx = x - lastTouchX
                    val dy = y - lastTouchY

                    val minX = currentCorners.minOf { it.x }
                    val maxX = currentCorners.maxOf { it.x }
                    val minY = currentCorners.minOf { it.y }
                    val maxY = currentCorners.maxOf { it.y }
                    
                    val centerX = (minX + maxX) / 2f
                    val centerY = (minY + maxY) / 2f

                    val allowedDx = if (dx > 0) minOf(dx, width.toFloat() - centerX) else maxOf(dx, -centerX)
                    val allowedDy = if (dy > 0) minOf(dy, height.toFloat() - centerY) else maxOf(dy, -centerY)

                    for (point in currentCorners) {
                        point.x += allowedDx
                        point.y += allowedDy
                    }

                    lastTouchX = x
                    lastTouchY = y
                    invalidate()
                    return true
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDragging = false
            }
        }
        return super.onTouchEvent(event)
    }

    fun getCorners(): List<PointF> {
        return currentCorners.toList()
    }
}
