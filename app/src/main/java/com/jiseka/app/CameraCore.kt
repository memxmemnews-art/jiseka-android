package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental

data class CaptureTransformData(
    val sensorToBufferMatrix: Matrix,
    val rotationDegrees: Int,
    val bufferWidth: Int,
    val bufferHeight: Int
)

fun ImageProxy.toUprightBitmap(): Bitmap {
    val bitmap = toBitmap()
    val rotation = imageInfo.rotationDegrees

    if (rotation == 0) return bitmap

    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

    if (rotated !== bitmap && !bitmap.isRecycled) {
        bitmap.recycle()
    }
    return rotated
}

@TransformExperimental
object CameraCoordinateConverter {
    fun mapUiToExactBitmap(
        previewView: PreviewView,
        transformData: CaptureTransformData,
        uiPoints: List<PointF>,
        uprightBitmapWidth: Int,
        uprightBitmapHeight: Int
    ): List<PointF> {
        if (uiPoints.isEmpty()) return emptyList()

        val sensorToUiMatrix = previewView.outputTransform?.matrix 
            ?: throw IllegalStateException("PreviewView OutputTransform is null")
        
        val uiToSensorMatrix = Matrix()
        sensorToUiMatrix.invert(uiToSensorMatrix)

        val uiToBufferMatrix = Matrix()
        uiToBufferMatrix.setConcat(transformData.sensorToBufferMatrix, uiToSensorMatrix)

        val bufferToUprightMatrix = Matrix()
        if (transformData.rotationDegrees != 0) {
            bufferToUprightMatrix.postRotate(transformData.rotationDegrees.toFloat())
            val bufferRect = RectF(0f, 0f, transformData.bufferWidth.toFloat(), transformData.bufferHeight.toFloat())
            val uprightRect = RectF()
            bufferToUprightMatrix.mapRect(uprightRect, bufferRect)
            bufferToUprightMatrix.postTranslate(-uprightRect.left, -uprightRect.top)
        }

        val finalTransformMatrix = Matrix()
        finalTransformMatrix.setConcat(bufferToUprightMatrix, uiToBufferMatrix)

        val srcPoints = FloatArray(uiPoints.size * 2)
        val dstPoints = FloatArray(uiPoints.size * 2)

        for (i in uiPoints.indices) {
            srcPoints[i * 2] = uiPoints[i].x
            srcPoints[i * 2 + 1] = uiPoints[i].y
        }

        finalTransformMatrix.mapPoints(dstPoints, srcPoints)

        return uiPoints.indices.map { i ->
            val rawX = dstPoints[i * 2]
            val rawY = dstPoints[i * 2 + 1]
            PointF(
                rawX.coerceIn(0f, uprightBitmapWidth.toFloat() - 1f),
                rawY.coerceIn(0f, uprightBitmapHeight.toFloat() - 1f)
            )
        }
    }
}
