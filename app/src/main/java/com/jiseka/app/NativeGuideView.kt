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
    
    // 🌟 시그니처 일치: 정밀도 레벨을 전달하는 리스너
    var onDwellTriggeredListener: ((Int) -> Unit)? = null

    private var crosshairPoint: PointF? = null
    private val uiPolygonBuffer = Array(4) { PointF() }
    private var hasHoveredPolygon = false

    // 🌟 내부 독점 상태 머신 (State Machine)
    private val dwellHandler = Handler(Looper.getMainLooper())
    private var isDwelling = false
    private var refinementLevel = 0
    private val MAX_REFINEMENT_LEVEL = 2
    
    private var lastMoveX = 0f
    private var lastMoveY = 0f
    private val touchSlop = 15f 

    private fun getCurrentDwellTime() = if (refinementLevel == 0) 1500L else 1000L

    private val dwellRunnable = Runnable {
        if (!isDwelling && refinementLevel < MAX_REFINEMENT_LEVEL) {
            isDwelling = true
            onDwellTriggeredListener?.invoke(refinementLevel)
            postInvalidateOnAnimation()
        }
    }

    // 🌟 구조 불일치 해결: 성공 여부에 따라 다단계 타이머를 관리하는 코어 함수
    fun notifyRefinementCompleted(success: Boolean) {
        if (success && refinementLevel < MAX_REFINEMENT_LEVEL) {
            refinementLevel++
        }
        
        isDwelling = false 
        
        if (refinementLevel < MAX_REFINEMENT_LEVEL) {
            dwellHandler.removeCallbacks(dwellRunnable)
            dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
        }
        postInvalidateOnAnimation()
    }

    // 🌟 생명주기 동기화: 액티비티 초기화 시 뷰 내부 상태도 리셋
    fun resetState() {
        dwellHandler.removeCallbacks(dwellRunnable)
        isDwelling = false
        refinementLevel = 0
        hasHoveredPolygon = false
        crosshairPoint = null
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
