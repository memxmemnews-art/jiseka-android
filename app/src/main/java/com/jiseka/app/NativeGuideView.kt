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
    var onDwellTriggeredListener: ((PointF) -> Unit)? = null // 십자선 좌표를 직접 전달

    // MainActivity로부터 기기의 현재 회전 상태(0, 90, 180, 270)를 전달받음
    var currentDeviceRotation = 0f

    private val crosshairPoint = PointF()
    private var hasCrosshair = true 
    
    private var uiPolygonBuffer = emptyArray<PointF>()
    private var hasHoveredPolygon = false

    private val dwellHandler = Handler(Looper.getMainLooper())
    private var isDwelling = false
    
    private var lastMoveX = 0f; private var lastMoveY = 0f
    private val touchSlop = 15f 

    // 단일 1.5초 타이머 고정
    private val DWELL_TIME = 1500L 

    private val dwellRunnable = Runnable {
        if (!isDwelling) {
            isDwelling = true
            onDwellTriggeredListener?.invoke(PointF(crosshairPoint.x, crosshairPoint.y))
            postInvalidateOnAnimation()
        }
    }

    fun notifyRefinementCompleted(success: Boolean) {
        isDwelling = false 
        postInvalidateOnAnimation()
    }

    fun resetState() {
        dwellHandler.removeCallbacksAndMessages(null)
        isDwelling = false
        hasHoveredPolygon = false
        hasCrosshair = true 
        uiPolygonBuffer = emptyArray()
        postInvalidateOnAnimation()
    }

    fun setHoveredPolygon(polygon: Array<PointF>?) {
        if (polygon != null && polygon.isNotEmpty()) {
            uiPolygonBuffer = polygon
            hasHoveredPolygon = true
        } else {
            uiPolygonBuffer = emptyArray()
            hasHoveredPolygon = false
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

            hoverStrokePaint.color = Color.GREEN 
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
        var x = event.x
        var y = event.y
        val offset = 150f 

        // 🌟 기기 회전 방향에 맞춰 사용자의 '시각적 위쪽'으로 십자선 오프셋 적용 (팻핑거 방지)
        when (currentDeviceRotation) {
            0f -> y -= offset    // 세로 정방향
            90f -> x += offset   // 가로 (상단이 왼쪽)
            180f -> y += offset  // 세로 (뒤집힘)
            270f -> x -= offset  // 가로 (상단이 오른쪽)
            else -> y -= offset
        }

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                hasCrosshair = true; crosshairPoint.set(x, y)
                lastMoveX = x; lastMoveY = y; isDwelling = false
                dwellHandler.removeCallbacks(dwellRunnable)
                dwellHandler.postDelayed(dwellRunnable, DWELL_TIME)
                onCrosshairMoveListener?.invoke(PointF(x, y))
                postInvalidateOnAnimation()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                crosshairPoint.set(x, y)
                if (Math.abs(x - lastMoveX) > touchSlop || Math.abs(y - lastMoveY) > touchSlop) {
                    isDwelling = false; lastMoveX = x; lastMoveY = y
                    dwellHandler.removeCallbacks(dwellRunnable)
                    dwellHandler.postDelayed(dwellRunnable, DWELL_TIME)
                }
                onCrosshairMoveListener?.invoke(PointF(x, y))
                postInvalidateOnAnimation()
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
