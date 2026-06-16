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

    // 터치 좌표 전달 리스너
    var onTouchPointListener: ((PointF) -> Unit)? = null 

    var currentDeviceRotation = 0f
    private val touchPoint = PointF()
    private var isTouched = false 

    // 터치 포인트를 표시할 페인트 (눈에 띄게 CYAN 크기 확대)
    private val pointPaint = Paint().apply { color = Color.CYAN; style = Paint.Style.FILL; isAntiAlias = true }

    fun resetState() {
        isTouched = false 
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (isTouched) {
            canvas.drawCircle(touchPoint.x, touchPoint.y, 40f, pointPaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isTouched = true
                touchPoint.set(x, y)
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (isTouched) {
                    isTouched = false
                    // 손을 떼는 순간 1개의 좌표만 전달
                    onTouchPointListener?.invoke(PointF(touchPoint.x, touchPoint.y))
                }
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
