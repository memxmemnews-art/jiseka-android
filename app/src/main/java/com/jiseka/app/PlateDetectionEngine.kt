// =====================================================================
        // 🚀 [3단계] OpenCV 내장 윤곽선(Contour) 탐지 알고리즘 적용
        // =====================================================================
        val thresh = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 7.0))
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var resultPoints: List<ImmutablePoint>? = null

        try {
            // ---------------------------------------------------------
            // 3-1. 가우시안 블러 (노이즈 제거)
            // ---------------------------------------------------------
            Imgproc.GaussianBlur(rotatedPaddedGray, thresh, Size(5.0, 5.0), 0.0)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-1: Gaussian Blur", listOf(
                    "Kernel Size: 5x5",
                    "Status: High-frequency noise reduced"
                ))
                it.pauseAndShowStep("3-1단계: 가우시안 블러", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            // ---------------------------------------------------------
            // 3-2. 적응형 이진화 (흑백 대비 극대화)
            // ---------------------------------------------------------
            Imgproc.adaptiveThreshold(thresh, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 10.0)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-2: Adaptive Threshold", listOf(
                    "Block Size: 15, C: 10.0",
                    "Status: Foreground (Text/Borders) extracted"
                ))
                it.pauseAndShowStep("3-2단계: 적응형 이진화", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            // ---------------------------------------------------------
            // 3-3. 모폴로지 닫기 (글자와 테두리 뭉치기)
            // ---------------------------------------------------------
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)
            
            debugListener?.let {
                val debugBmp = Bitmap.createBitmap(thresh.cols(), thresh.rows(), Bitmap.Config.ARGB_8888)
                val tempRgb = Mat()
                Imgproc.cvtColor(thresh, tempRgb, Imgproc.COLOR_GRAY2RGBA)
                Utils.matToBitmap(tempRgb, debugBmp)
                val hudBmp = addDebugHUD(debugBmp, "Step 3-3: Morphology Close", listOf(
                    "Kernel Size: 25x7 (Horizontal emphasis)",
                    "Status: Text and boundaries merged into blocks"
                ))
                it.pauseAndShowStep("3-3단계: 모폴로지 닫기", hudBmp)
                tempRgb.release(); debugBmp.recycle()
            }

            // ---------------------------------------------------------
            // 3-4. 윤곽선 탐지 및 최종 사각형(minAreaRect) 도출
            // ---------------------------------------------------------
            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            var maxArea = -1.0
            var bestContour: MatOfPoint? = null
            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area > maxArea) {
                    maxArea = area
                    bestContour = contour
                }
            }

            val rectPoints = arrayOfNulls<Point>(4)
            if (bestContour != null && maxArea > 1000.0) { 
                val contour2f = MatOfPoint2f(*bestContour.toArray())
                val minRect = Imgproc.minAreaRect(contour2f)
                minRect.points(rectPoints)
                contour2f.release()
            } else {
                rectPoints[0] = Point(0.0, rotatedPaddedGray.rows().toDouble())
                rectPoints[1] = Point(0.0, 0.0)
                rectPoints[2] = Point(rotatedPaddedGray.cols().toDouble(), 0.0)
                rectPoints[3] = Point(rotatedPaddedGray.cols().toDouble(), rotatedPaddedGray.rows().toDouble())
            }

            debugListener?.let {
                val debugMat = rotatedPaddedMat.clone()
                val color = Scalar(0.0, 255.0, 0.0, 255.0)
                for (i in 0..3) {
                    Imgproc.line(debugMat, rectPoints[i], rectPoints[(i + 1) % 4], color, 4)
                }
                
                val debugBmp = Bitmap.createBitmap(debugMat.cols(), debugMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(debugMat, debugBmp)

                val hudBmp = addDebugHUD(debugBmp, "Step 3-4: Final Rectangle Extraction", listOf(
                    "Detected Plate Area: ${String.format("%.1f", maxArea)} px",
                    "Method: MinAreaRect on largest contour",
                    "Status: Ready to un-warp and mask"
                ))
                it.pauseAndShowStep("3-4단계: 최종 번호판 사각형 추출", hudBmp)
                debugMat.release(); debugBmp.recycle()
            }

            // =====================================================================
            // [4단계] 좌표 원복 (역회전 및 오프셋 더하기)
            // =====================================================================
            val invRotMat = Mat()
            Imgproc.invertAffineTransform(rotMat, invRotMat)

            val srcMat = MatOfPoint2f(*rectPoints.filterNotNull().toTypedArray())
            val dstMat = MatOfPoint2f()

            Core.transform(srcMat, dstMat, invRotMat)

            resultPoints = dstMat.toArray().map { 
                ImmutablePoint((it.x + paddedRect.x).toFloat(), (it.y + paddedRect.y).toFloat()) 
            }

            invRotMat.release(); srcMat.release(); dstMat.release()

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            thresh.release(); kernel.release()
            contours.forEach { it.release() }; hierarchy.release()
            
            rotMat.release()
            paddedMat.release(); paddedGray.release()
            rotatedPaddedMat.release(); rotatedPaddedGray.release()
            fullMat.release(); fullGray.release()
        }

        return resultPoints
