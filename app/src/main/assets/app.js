let isPhotoCapturing = false;
let isAnalyzing = false;
let isSaving = false;
let captureTimeout = null;

let guideWrapper;
let mainGuideBox;
window.normalizedCornersPayload = null;
let currentPerspectiveMode = 'FRONT';
let guideInitialized = false;

const MODE_POLYGONS = {
    FRONT: [{ x: 0.0, y: 0.0 }, { x: 1.0, y: 0.0 }, { x: 1.0, y: 1.0 }, { x: 0.0, y: 1.0 }],
    PASSENGER: [{ x: 0.0, y: 0.0 }, { x: 0.90, y: 0.15 }, { x: 0.90, y: 0.85 }, { x: 0.0, y: 1.0 }],
    DRIVER: [{ x: 0.10, y: 0.15 }, { x: 1.0, y: 0.0 }, { x: 1.0, y: 1.0 }, { x: 0.10, y: 0.85 }]
};

window.addEventListener('load', () => {
    requestAnimationFrame(() => {
        guideWrapper = document.getElementById('guideWrapper');
        mainGuideBox = document.getElementById('mainGuideBox');
        if (guideWrapper) {
            initializeGuideBox();
            setPerspectiveMode('FRONT');
            const panel = document.getElementById('perspectiveModePanel');
            if (panel) panel.classList.remove('visible');
        }
    });
});

let isDragging = false;
let startX, startY, initialLeft, initialTop;

function resetGuideBoxPosition() {
    if (!guideWrapper) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    const boxW = guideWrapper.offsetWidth || (w * 0.45);
    const boxH = guideWrapper.offsetHeight || (boxW / 3);
    guideWrapper.style.left = ((w - boxW) / 2) + 'px';
    guideWrapper.style.top = ((h - boxH) / 2) + 'px';
}

function getClipPathString(points) {
    return `polygon(${points.map(p => `${(p.x * 100).toFixed(1)}% ${(p.y * 100).toFixed(1)}%`).join(', ')})`;
}

function setPerspectiveMode(mode) {
    if (!MODE_POLYGONS[mode]) return;
    currentPerspectiveMode = mode;

    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.innerText.includes(mode === 'FRONT' ? '정면' : mode === 'PASSENGER' ? '조수석' : '운전석')) {
            btn.classList.add('active');
        }
    });

    if (mainGuideBox) {
        mainGuideBox.classList.remove('mode-front', 'mode-passenger', 'mode-driver');
        mainGuideBox.classList.add(`mode-${mode.toLowerCase()}`);

        const points = MODE_POLYGONS[mode];
        const clipPathValue = getClipPathString(points);
        mainGuideBox.style.clipPath = clipPathValue;
        mainGuideBox.style.webkitClipPath = clipPathValue;
    }
}

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(el => {
        el.classList.remove('active');
    });
    
    const target = document.getElementById(screenId);
    if (target) {
        target.classList.add('active');
    }

    const panel = document.getElementById('perspectiveModePanel');
    if (panel) {
        if (screenId === 'adjustUI') panel.classList.add('visible');
        else panel.classList.remove('visible');
    }
}

function initializeGuideBox() {
    resetGuideBoxPosition();
    if (guideInitialized) return;
    guideInitialized = true;

    guideWrapper.addEventListener('pointerdown', (e) => {
        if (isPhotoCapturing || isAnalyzing) return;
        isDragging = true; guideWrapper.setPointerCapture(e.pointerId);
        startX = e.clientX; startY = e.clientY;
        initialLeft = parseInt(guideWrapper.style.left || 0, 10);
        initialTop = parseInt(guideWrapper.style.top || 0, 10);
    });
    guideWrapper.addEventListener('pointermove', (e) => {
        if (!isDragging || isPhotoCapturing || isAnalyzing) return;
        const boxW = guideWrapper.offsetWidth; const boxH = guideWrapper.offsetHeight;
        let dx = e.clientX - startX; let dy = e.clientY - startY;
        let newLeft = Math.max(0, Math.min(initialLeft + dx, window.innerWidth - boxW));
        let newTop = Math.max(0, Math.min(initialTop + dy, window.innerHeight - boxH));
        guideWrapper.style.left = newLeft + 'px'; guideWrapper.style.top = newTop + 'px';
    });

    guideWrapper.addEventListener('pointerup', (e) => {
        isDragging = false; try { guideWrapper.releasePointerCapture(e.pointerId); } catch (err) { }
        analyzePlate();
    });
    guideWrapper.addEventListener('pointercancel', () => { isDragging = false; });
}

function triggerCapture() {
    if (isPhotoCapturing) return;
    isPhotoCapturing = true;

    const btn = document.querySelector('#liveUI .shutter-btn');
    if (btn) { btn.innerText = "사진 캡처 중..."; btn.disabled = true; }

    captureTimeout = setTimeout(() => {
        if (isPhotoCapturing) {
            showErrorToast("촬영 응답 초과");
            resetUIState();
        }
    }, 10000);

    if (window.AndroidBridge && window.AndroidBridge.takePhoto) {
        window.AndroidBridge.takePhoto();
    } else {
        showErrorToast("앱 환경 전용");
        resetUIState();
    }
}

window.onNativePhotoCaptured = function (imageUri) {
    showScreen('adjustUI');
    
    const previewImg = document.getElementById('previewImg');
    if (previewImg && imageUri) {
        previewImg.src = imageUri;
    }

    if (guideWrapper) {
        resetGuideBoxPosition();
        guideWrapper.classList.add('visible');
    }
    resetUIState();
};

function analyzePlate() {
    if (isAnalyzing) return;
    isAnalyzing = true;

    if (mainGuideBox) mainGuideBox.style.filter = "drop-shadow(0px 0px 4px #FF0000)";

    const boxW = guideWrapper.offsetWidth;
    const boxH = guideWrapper.offsetHeight; 
    const wrapperLeft = guideWrapper.offsetLeft;
    const wrapperTop = guideWrapper.offsetTop;

    const screenW = window.innerWidth;
    const screenH = window.innerHeight;

    const pts = MODE_POLYGONS[currentPerspectiveMode] || MODE_POLYGONS.FRONT;

    const payloadCorners = pts.map(pt => {
        const absPixelX = wrapperLeft + (pt.x * boxW);
        const absPixelY = wrapperTop + (pt.y * boxH);
        return {
            x: Math.max(0, Math.min(1, absPixelX / screenW)),
            y: Math.max(0, Math.min(1, absPixelY / screenH))
        };
    });

    const payloadStr = JSON.stringify({ corners: payloadCorners });

    if (window.AndroidBridge && window.AndroidBridge.analyzePlateWithMode) {
        window.AndroidBridge.analyzePlateWithMode(payloadStr, currentPerspectiveMode);
    } else {
        showErrorToast("분석 모듈 연결 실패");
        resetUIState();
    }
}

window.onNativeSuccess = function (payloadStr) {
    // 🚨 [깜빡임 원천 차단]: 상단에서 즉시 카메라를 숨기던 코드를 제거하여, 
    // 브라우저가 새 이미지를 디코딩하는 동안 네이티브 정지 화면(Freeze Frame)이 공백을 완벽히 메우도록 보장
    try {
        const payload = JSON.parse(payloadStr);

        showScreen('resultUI');
        if (guideWrapper) guideWrapper.classList.remove('visible');

        const capturedImg = document.getElementById('capturedImg');
        const container = document.getElementById('resultContainer');

        if (!capturedImg) return;

        capturedImg.onload = () => {
            if (!payload.corners) {
                window.normalizedCornersPayload = JSON.stringify({ corners: null });
                showErrorToast("번호판 영역 감지 미달. 원본을 유지합니다.");
                resetUIState();
                return;
            }

            const naturalW = capturedImg.naturalWidth;
            const naturalH = capturedImg.naturalHeight;
            const screenW = window.innerWidth;
            const screenH = window.innerHeight;
            
            const imgRatio = naturalW / naturalH;
            const screenRatio = screenW / screenH;

            let renderW, renderH;
            if (imgRatio > screenRatio) {
                renderW = screenW;
                renderH = screenW / imgRatio;
            } else {
                renderH = screenH;
                renderW = screenH * imgRatio;
            }

            container.style.width = renderW + 'px';
            container.style.height = renderH + 'px';

            window.normalizedCornersPayload = JSON.stringify({ corners: payload.corners });
            
            window.applyPlateMask(payload.corners, renderW, renderH);

            // 🚨 [동기화 타이밍 최적화]: 결과 이미지가 브라우저 레이어에 100% 출력 완료된 이 시점에
            // 최종적으로 네이티브 카메라 가시성 차단을 요청하여 단 1 프레임의 깜빡임이나 검은 화면도 허용하지 않음
            if (window.AndroidBridge && window.AndroidBridge.setCameraVisibility) {
                window.AndroidBridge.setCameraVisibility(false);
            }
        };

        if (payload.preview) {
            capturedImg.src = payload.preview;
        }

    } catch (e) {
        showErrorToast("프리뷰 로딩 에러");
        if (window.AndroidBridge && window.AndroidBridge.setCameraVisibility) {
            window.AndroidBridge.setCameraVisibility(false);
        }
    } finally {
        resetUIState();
    }
};

window.onNativeError = function (errorMsg) {
    showErrorToast(errorMsg);
    resetUIState();
};

function showErrorToast(msg) {
    if (window.AndroidBridge && window.AndroidBridge.showToast) window.AndroidBridge.showToast(msg);
    else alert("시스템 알림: " + msg);
}

function retry() {
    if (window.AndroidBridge && window.AndroidBridge.setCameraVisibility) {
        window.AndroidBridge.setCameraVisibility(true);
    }
    showScreen('liveUI');
    window.normalizedCornersPayload = null;

    const capturedImg = document.getElementById('capturedImg');
    if (capturedImg) capturedImg.src = "";

    if (guideWrapper) {
        guideWrapper.classList.remove('visible');
        setPerspectiveMode('FRONT');
    }
    resetUIState();
}

function resetUIState() {
    isPhotoCapturing = false;
    isAnalyzing = false;
    if (captureTimeout) { clearTimeout(captureTimeout); captureTimeout = null; }
    if (mainGuideBox) mainGuideBox.style.filter = "drop-shadow(0px 0px 2px #00FF00)";
    const btnLive = document.querySelector('#liveUI .shutter-btn');
    if (btnLive) { btnLive.innerText = "📸 사진 촬영"; btnLive.disabled = false; }
}

function downloadImage() {
    if (isSaving) return;
    isSaving = true;
    const btn = document.querySelector('.download-btn');
    const originalText = btn.innerText;
    btn.innerText = "최종 갤러리 저장 중..."; btn.disabled = true;

    if (window.AndroidBridge && window.AndroidBridge.saveImageWithNativeOverlay && window.normalizedCornersPayload) {
        window.AndroidBridge.saveImageWithNativeOverlay(window.normalizedCornersPayload);
    } else {
        showErrorToast("합성 모듈 연결 실패");
        btn.innerText = originalText; btn.disabled = false; isSaving = false;
    }
}

window.onNativeSaveComplete = function () {
    const btn = document.querySelector('.download-btn');
    if (btn) {
        btn.innerText = "저장 완료!";
        setTimeout(() => { btn.innerText = "💾 이대로 저장"; btn.disabled = false; }, 2000);
    }
    isSaving = false;
};

window.setCanvasSize = function (w, h) { };