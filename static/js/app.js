document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".score-ring[data-score]").forEach((ring) => {
        const value = Number.parseFloat(ring.dataset.score || "0");
        const score = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
        ring.style.setProperty("--score", String(score));
    });

    document.querySelectorAll(".meter-fill[data-value]").forEach((fill) => {
        const value = Number.parseFloat(fill.dataset.value || "0");
        const progress = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
        fill.style.width = `${progress}%`;
    });
});
