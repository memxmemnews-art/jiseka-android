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

    var onCrosshairMoveListener: ((PointF) -> Unit)? = null
    var onCrosshairDropListener: ((PointF) -> Unit)? = null 

    var currentDeviceRotation = 0f
    private val crosshairPoint = PointF()
    private var hasCrosshair = true 
    
    private val touchSlop = 15f 
    private var lastMoveX = 0f; private var lastMoveY = 0f

    private val crosshairBackPaint = Paint().apply { color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 9f; isAntiAlias = true }
    private val crosshairFrontPaint = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true }
    private val centerDotPaint = Paint().apply { color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true }

    fun resetState() {
        hasCrosshair = true 
        postInvalidateOnAnimation()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (w > 0 && h > 0) crosshairPoint.set(w / 2f, h / 2f)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (hasCrosshair) {
            val pt = crosshairPoint
            val radius = 40f
            canvas.drawCircle(pt.x, pt.y, radius, crosshairBackPaint); canvas.drawLine(pt.x - radius - 20f, pt.y, pt.x - radius, pt.y, crosshairBackPaint)
            canvas.drawLine(pt.x + radius, pt.y, pt.x + radius + 20f, pt.y, crosshairBackPaint); canvas.drawLine(pt.x, pt.y - radius - 20f, pt.x, pt.y - radius, crosshairBackPaint)
            canvas.drawLine(pt.x, pt.y + radius, pt.x, pt.y + radius + 20f, crosshairBackPaint)
            canvas.drawCircle(pt.x, pt.y, radius, crosshairFrontPaint); canvas.drawLine(pt.x - radius - 20f, pt.y, pt.x - radius, pt.y, crosshairFrontPaint)
            canvas.drawLine(pt.x + radius, pt.y, pt.x + radius + 20f, pt.y, crosshairFrontPaint); canvas.drawLine(pt.x, pt.y - radius - 20f, pt.x, pt.y - radius, crosshairFrontPaint)
            canvas.drawLine(pt.x, pt.y + radius, pt.x, pt.y + radius + 20f, crosshairFrontPaint); canvas.drawCircle(pt.x, pt.y, 8f, centerDotPaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        var x = event.x
        var y = event.y
        val offset = 150f 

        // 기기 회전 방향에 맞춰 사용자의 '시각적 위쪽'으로 십자선 오프셋 적용
        when (currentDeviceRotation) {
            0f -> y -= offset    
            90f -> x += offset   
            180f -> y += offset  
            270f -> x -= offset  
            else -> y -= offset
        }

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                hasCrosshair = true
                crosshairPoint.set(x, y)
                lastMoveX = x
                lastMoveY = y
                onCrosshairMoveListener?.invoke(PointF(x, y))
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                // UI는 매 픽셀마다 부드럽게 갱신
                crosshairPoint.set(x, y)
                postInvalidateOnAnimation()

                // 리스너 호출은 터치 슬롭 이상 움직였을 때만 (스로틀링 방어)
                if (Math.abs(x - lastMoveX) > touchSlop || Math.abs(y - lastMoveY) > touchSlop) {
                    lastMoveX = x
                    lastMoveY = y
                    onCrosshairMoveListener?.invoke(PointF(x, y))
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                onCrosshairDropListener?.invoke(PointF(crosshairPoint.x, crosshairPoint.y))
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
