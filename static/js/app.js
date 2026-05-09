document.addEventListener("DOMContentLoaded", () => {
    // Initialize score circles
    document.querySelectorAll(".score-circle[data-score]").forEach((circle) => {
        const value = Number.parseFloat(circle.dataset.score || "0");
        const score = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
        circle.style.setProperty("--score", String(score));
    });

    // Initialize meter fills
    document.querySelectorAll(".meter-fill[data-value]").forEach((fill) => {
        const value = Number.parseFloat(fill.dataset.value || "0");
        const progress = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
        fill.style.width = `${progress}%`;
    });

    // Trigger animation on page load
    setTimeout(() => {
        document.querySelectorAll(".meter-fill").forEach((fill) => {
            if (!fill.dataset.value) return;
            const value = Number.parseFloat(fill.dataset.value);
            const progress = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
            fill.style.width = `${progress}%`;
        });
    }, 100);
});
