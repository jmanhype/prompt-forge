// Prompt Forge — Canvas controller (image display + bbox overlay + heatmap)

export class ForgeCanvas {
    constructor(canvasEl) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.image = null;
        this.elements = [];
        this.heatmap = null;
        this.showBboxes = true;
        this.showHeatmap = false;
    }

    loadImage(src) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                this.image = img;
                this.canvas.width = img.width;
                this.canvas.height = img.height;
                this.render();
                resolve();
            };
            img.src = src;
        });
    }

    setElements(elements) {
        this.elements = elements || [];
        this.render();
    }

    setHeatmap(heatmapData) {
        this.heatmap = heatmapData;
        this.showHeatmap = true;
        this.render();
    }

    clearHeatmap() {
        this.heatmap = null;
        this.showHeatmap = false;
        this.render();
    }

    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Draw image
        if (this.image) {
            ctx.drawImage(this.image, 0, 0);
        }

        // Draw heatmap overlay
        if (this.showHeatmap && this.heatmap && this.heatmap.regions) {
            for (const region of this.heatmap.regions) {
                const elem = this.elements.find(e => e.id === region.id);
                if (!elem) continue;

                const [x1, y1, x2, y2] = elem.bbox;
                const rx = x1 * w, ry = y1 * h;
                const rw = (x2 - x1) * w, rh = (y2 - y1) * h;

                ctx.fillStyle = region.color + Math.round(region.opacity * 255).toString(16).padStart(2, '0');
                ctx.fillRect(rx, ry, rw, rh);

                // Border
                ctx.strokeStyle = region.color;
                ctx.lineWidth = 2;
                ctx.strokeRect(rx, ry, rw, rh);

                // Label
                ctx.fillStyle = 'white';
                ctx.font = '12px JetBrains Mono';
                ctx.fillText(`${region.label} (${region.score})`, rx + 4, ry + 14);
            }
        }

        // Draw element bboxes
        if (this.showBboxes && !this.showHeatmap) {
            for (const elem of this.elements) {
                if (!elem.bbox || elem.bbox.length !== 4) continue;
                const [x1, y1, x2, y2] = elem.bbox;
                const rx = x1 * w, ry = y1 * h;
                const rw = (x2 - x1) * w, rh = (y2 - y1) * h;

                ctx.strokeStyle = elem.type === 'text' ? '#8b5cf6' : '#f97316';
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(rx, ry, rw, rh);
                ctx.setLineDash([]);

                // Label
                ctx.fillStyle = 'rgba(0,0,0,0.7)';
                ctx.fillRect(rx, ry - 16, ctx.measureText(elem.label).width + 12, 16);
                ctx.fillStyle = 'white';
                ctx.font = '11px JetBrains Mono';
                ctx.fillText(elem.label, rx + 4, ry - 4);
            }
        }
    }

    clear() {
        this.image = null;
        this.elements = [];
        this.heatmap = null;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

window.ForgeCanvas = ForgeCanvas;
