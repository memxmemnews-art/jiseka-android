package com.jiseka.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class NativeGuideView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var onCrosshairMoveListener: ((PointF) -> Unit)? = null
    var onCrosshairDropListener: (() -> Unit)? = null
    
    // 현재 정밀도 레벨(0 또는 1)을 액티비티로 전달하는 리스너
    var onDwellTriggeredListener: ((Int) -> Unit)? = null

    private var crosshairPoint: PointF? = null
    private val uiPolygonBuffer = Array(4) { PointF() }
    private var hasHoveredPolygon = false

    // 🌟 다단계 Progressive Refinement 제어기
    private val dwellHandler = Handler(Looper.getMainLooper())
    private var isDwelling = false
    private var refinementLevel = 0
    private val MAX_REFINEMENT_LEVEL = 2
    
    private var lastMoveX = 0f
    private var lastMoveY = 0f
    private val touchSlop = 15f 

    // 레벨 0에서는 1.5초 대기, 레벨 1에서는 1.0초 대기
    private fun getCurrentDwellTime() = if (refinementLevel == 0) 1500L else 1000L

    private val dwellRunnable = Runnable {
        if (!isDwelling && refinementLevel < MAX_REFINEMENT_LEVEL) {
            isDwelling = true
            onDwellTriggeredListener?.invoke(refinementLevel)
            postInvalidateOnAnimation()
        }
    }

    // 🌟 엔진 연산 종료 후 다음 단계 타이머 재장전
    fun notifyRefinementCompleted(success: Boolean) {
        if (success && refinementLevel < MAX_REFINEMENT_LEVEL) {
            refinementLevel++
            isDwelling = false 
            
            dwellHandler.removeCallbacks(dwellRunnable)
            if (refinementLevel < MAX_REFINEMENT_LEVEL) {
                dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
            }
        } else {
            isDwelling = false
        }
        postInvalidateOnAnimation()
    }

    private val crosshairBackPaint = Paint().apply { color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 9f; isAntiAlias = true }
    private val crosshairFrontPaint = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true }
    private val centerDotPaint = Paint().apply { color = Color.RED; style = Paint.Style.FILL; isAntiAlias = true }

    private val hoverStrokePaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 8f; isAntiAlias = true }
    private val hoverFillPaint = Paint().apply { color = Color.parseColor("#4400FF00"); style = Paint.Style.FILL }

    private val polygonPath = Path()

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (w > 0 && h > 0 && crosshairPoint == null) crosshairPoint = PointF(w / 2f, h / 2f)
    }

    fun setHoveredPolygon(points: Array<PointF>?) {
        if (points == null) {
            hasHoveredPolygon = false
        } else {
            for (i in 0 until 4) uiPolygonBuffer[i].set(points[i].x, points[i].y)
            hasHoveredPolygon = true
        }
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (hasHoveredPolygon) {
            polygonPath.reset()
            polygonPath.moveTo(uiPolygonBuffer[0].x, uiPolygonBuffer[0].y)
            polygonPath.lineTo(uiPolygonBuffer[1].x, uiPolygonBuffer[1].y)
            polygonPath.lineTo(uiPolygonBuffer[2].x, uiPolygonBuffer[2].y)
            polygonPath.lineTo(uiPolygonBuffer[3].x, uiPolygonBuffer[3].y)
            polygonPath.close()

            // 🌟 시각적 피드백: 레벨 0(초록) -> 레벨 1(시안) -> 레벨 2(노랑)
            hoverStrokePaint.color = when (refinementLevel) {
                0 -> Color.GREEN
                1 -> Color.CYAN
                else -> Color.YELLOW
            }
            if (isDwelling) hoverStrokePaint.alpha = 128 else hoverStrokePaint.alpha = 255

            canvas.drawPath(polygonPath, hoverFillPaint)
            canvas.drawPath(polygonPath, hoverStrokePaint)
        }

        crosshairPoint?.let { pt ->
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
        val x = event.x; val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                crosshairPoint?.set(x, y); lastMoveX = x; lastMoveY = y
                refinementLevel = 0; isDwelling = false
                
                dwellHandler.removeCallbacks(dwellRunnable)
                dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
                
                onCrosshairMoveListener?.invoke(PointF(x, y))
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                crosshairPoint?.set(x, y)
                
                if (Math.abs(x - lastMoveX) > touchSlop || Math.abs(y - lastMoveY) > touchSlop) {
                    refinementLevel = 0; isDwelling = false
                    lastMoveX = x; lastMoveY = y
                    dwellHandler.removeCallbacks(dwellRunnable)
                    dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
                }

                onCrosshairMoveListener?.invoke(PointF(x, y))
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                dwellHandler.removeCallbacks(dwellRunnable)
                isDwelling = false
                onCrosshairDropListener?.invoke()
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
