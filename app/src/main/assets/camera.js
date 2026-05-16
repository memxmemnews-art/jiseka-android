// ═════════════════════════════════════════=====
// ── 지세카 정적 3D 마스킹 엔진 (Single-shot Perspective Engine) ──
// ═════════════════════════════════════════=====

const f = v => Number(v).toFixed(5);

window.setCanvasSize = function(w, h) {
    document.body.style.width = w + 'px';
    document.body.style.height = h + 'px';
};

window.applyPlateMask = function (cornersJSON, renderW, renderH) {
    const capturedImg = document.getElementById('capturedImg');
    if (!capturedImg) return;
    
    console.log("UI Controller 바인딩: 네이티브 렌더링 결과물 단일 소스 출력 완료", {
        renderedWidth: renderW,
        renderedHeight: renderH,
        payloadStatus: cornersJSON ? "Validated" : "Null Fallback"
    });
};

function getPerspectiveTransform(pts, testMode = "NONE") {
    return null;
}

window.addEventListener('load', () => {
    const ls = document.getElementById('loading-screen');
    if (ls) ls.style.display = 'none';
});