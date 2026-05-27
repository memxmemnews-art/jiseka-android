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
    var onDwellTriggeredListener: ((Int) -> Unit)? = null

    // 🌟 핵심 개선: nullable 제거, 객체 재사용(Object Pooling) 방식 채택
    private val crosshairPoint = PointF()
    
    // 🌟 핵심 추가: 렌더링 플래그로 뷰의 가시성 상태 관리
    private var hasCrosshair = false
    
    private val uiPolygonBuffer = Array(4) { PointF() }
    private var hasHoveredPolygon = false

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

    // 🌟 핵심 개선: 생명주기 안정화 (유령 호출 방지 및 상태 초기화)
    fun resetState() {
        dwellHandler.removeCallbacksAndMessages(null) // 모든 예약된 스레드 제거 보장
        isDwelling = false
        refinementLevel = 0
        hasHoveredPolygon = false
        hasCrosshair = false // 객체를 null로 만들지 않고 플래그만 Off
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
        // onSizeChanged에서는 화면의 중앙 좌표만 초기 세팅해둡니다.
        if (w > 0 && h > 0 && !hasCrosshair) {
            crosshairPoint.set(w / 2f, h / 2f)
        }
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

        // 🌟 플래그를 통한 렌더링 분기
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
        
        // 팻핑거 오프셋 (테스트 후 최적의 값을 찾으셨길 바랍니다!)
        val fatFingerOffsetY = -150f 
        val y = event.y + fatFingerOffsetY

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                // 🌟 객체 생성 체크 없이 안전하게 상태 변경 및 좌표 갱신
                hasCrosshair = true 
                crosshairPoint.set(x, y)
                
                lastMoveX = x; lastMoveY = y
                refinementLevel = 0; isDwelling = false
                
                dwellHandler.removeCallbacks(dwellRunnable)
                dwellHandler.postDelayed(dwellRunnable, getCurrentDwellTime())
                
                onCrosshairMoveListener?.invoke(PointF(x, y))
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                crosshairPoint.set(x, y)
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
