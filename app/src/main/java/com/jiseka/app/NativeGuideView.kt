package com.jiseka.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class NativeGuideView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // 선 긋기 완료 리스너 (시작점, 끝점 전달) - 에러 원인이었던 변수입니다.
    var onLineDropListener: ((PointF, PointF) -> Unit)? = null 

    var currentDeviceRotation = 0f
    private val startPoint = PointF()
    private val endPoint = PointF()
    private var isDrawing = false 

    private val lineBackPaint = Paint().apply { color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 12f; strokeCap = Paint.Cap.ROUND; isAntiAlias = true }
    private val lineFrontPaint = Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 6f; strokeCap = Paint.Cap.ROUND; isAntiAlias = true }
    private val pointPaint = Paint().apply { color = Color.GREEN; style = Paint.Style.FILL; isAntiAlias = true }

    fun resetState() {
        isDrawing = false 
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (isDrawing) {
            canvas.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y, lineBackPaint)
            canvas.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y, lineFrontPaint)
            
            canvas.drawCircle(startPoint.x, startPoint.y, 12f, pointPaint)
            canvas.drawCircle(endPoint.x, endPoint.y, 12f, pointPaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        var x = event.x
        var y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDrawing = true
                startPoint.set(x, y)
                endPoint.set(x, y)
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                endPoint.set(x, y)
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (isDrawing) {
                    isDrawing = false
                    val dist = kotlin.math.hypot(endPoint.x - startPoint.x, endPoint.y - startPoint.y)
                    if (dist > 20f) {
                        onLineDropListener?.invoke(PointF(startPoint.x, startPoint.y), PointF(endPoint.x, endPoint.y))
                    } else {
                        resetState() // 선이 너무 짧으면 취소
                    }
                }
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
