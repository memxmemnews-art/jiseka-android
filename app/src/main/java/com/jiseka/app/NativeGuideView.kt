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
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var onCrosshairMoveListener: ((PointF) -> Unit)? = null
    var onCrosshairDropListener: (() -> Unit)? = null
    var onDwellTriggeredListener: ((Int) -> Unit)? = null

    private val crosshairPoint = PointF()
    // 💡 수정됨: 터치 전에도 처음부터 십자선이 보이도록 강제 활성화
    private var hasCrosshair = true 
    
    private var uiPolygonBuffer = emptyArray<PointF>()
    private var hasHoveredPolygon = false

    private val dwellHandler = Handler(Looper.getMainLooper())
    private var isDwelling = false
    private var refinementLevel = 0
    private val MAX_REFINEMENT_LEVEL = 2
    
    private var lastMoveX = 0f; private var lastMoveY = 0f
    private val touchSlop = 15f 

    private fun getCurrentDwellTime() = if (refinementLevel == 0) 1500L else 1000L

    private val dwellRunnable = Runnable {
        if (!isDwelling && refinementLevel < MAX_REFINEMENT_LEVEL) {
            isDwelling = true; onDwellTriggeredListener?.invoke(refinementLevel); postInvalidateOnAnimation()
        }
    }

    fun notifyRefinementCompleted(success: Boolean) {
        if (success && refinementLevel < MAX_REFINEMENT_LEVEL) refinementLevel++
        isDwelling = false 
        if (refinementLevel < MAX_REFINEMENT_LEVEL) {
            dwellHandler.removeCallbacks(dwellRunnable)
            dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
        }
        postInvalidateOnAnimation()
    }

    fun resetState() {
        dwellHandler.removeCallbacksAndMessages(null)
        isDwelling = false; refinementLevel = 0; hasHoveredPolygon = false
        // 💡 수정됨: 라이브 모드로 돌아갈 때 십자선을 다시 켜둠
        hasCrosshair = true 
        uiPolygonBuffer = emptyArray()
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
        // 💡 수정됨: 화면 크기가 결정되면 즉시 화면 정중앙에 십자선을 배치
        if (w > 0 && h > 0) {
            crosshairPoint.set(w / 2f, h / 2f)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (hasHoveredPolygon && uiPolygonBuffer.isNotEmpty()) {
            polygonPath.reset()
            polygonPath.moveTo(uiPolygonBuffer[0].x, uiPolygonBuffer[0].y)
            for (i in 1 until uiPolygonBuffer.size) {
                polygonPath.lineTo(uiPolygonBuffer[i].x, uiPolygonBuffer[i].y)
            }
            polygonPath.close()

            hoverStrokePaint.color = when (refinementLevel) { 0 -> Color.GREEN; 1 -> Color.CYAN; else -> Color.YELLOW }
            hoverStrokePaint.alpha = if (isDwelling) 128 else 255

            canvas.drawPath(polygonPath, hoverFillPaint)
            canvas.drawPath(polygonPath, hoverStrokePaint)
        }

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
        val x = event.x
        val fatFingerOffsetY = -150f 
        val y = event.y + fatFingerOffsetY

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                hasCrosshair = true; crosshairPoint.set(x, y)
                lastMoveX = x; lastMoveY = y; refinementLevel = 0; isDwelling = false
                dwellHandler.removeCallbacks(dwellRunnable); dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
                onCrosshairMoveListener?.invoke(PointF(x, y)); postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                crosshairPoint.set(x, y)
                if (Math.abs(x - lastMoveX) > touchSlop || Math.abs(y - lastMoveY) > touchSlop) {
                    refinementLevel = 0; isDwelling = false; lastMoveX = x; lastMoveY = y
                    dwellHandler.removeCallbacks(dwellRunnable); dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
                }
                onCrosshairMoveListener?.invoke(PointF(x, y)); postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                dwellHandler.removeCallbacks(dwellRunnable); isDwelling = false
                onCrosshairDropListener?.invoke()
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
