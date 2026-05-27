package com.jiseka.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * ImageProxy 확장 함수
 * CameraX에서 전달받은 ImageProxy(JPEG 또는 YUV_420_888)를
 * 화면 회전 값에 맞게 정방향(Upright)의 ARGB_8888 Bitmap으로 변환합니다.
 */
fun ImageProxy.toUprightBitmap(): Bitmap {
    val bitmap = if (format == ImageFormat.YUV_420_888) {
        yuv420ToBitmap(this)
    } else {
        // ImageCapture의 기본 포맷인 JPEG 처리
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    val matrix = Matrix().apply {
        postRotate(imageInfo.rotationDegrees.toFloat())
    }

    val rotatedBitmap = Bitmap.createBitmap(
        bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
    )

    if (rotatedBitmap != bitmap) {
        bitmap.recycle()
    }

    // OpenCV 호환성을 위해 ARGB_8888로 최종 복제
    val finalBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)
    if (finalBitmap != rotatedBitmap) {
        rotatedBitmap.recycle()
    }

    return finalBitmap
}

/**
 * YUV_420_888 포맷을 Bitmap으로 변환하는 내부 헬퍼 함수
 */
private fun yuv420ToBitmap(image: ImageProxy): Bitmap {
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
    
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}
