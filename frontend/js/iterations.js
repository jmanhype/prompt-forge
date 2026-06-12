// Prompt Forge — Iteration timeline controller

export class IterationTimeline {
    constructor(containerEl) {
        this.container = containerEl;
        this.iterations = [];
        this.activeIndex = -1;
        this.onSelect = null; // callback(index)
    }

    addIteration(data) {
        this.iterations.push(data);
        this.render();
        this.setActive(this.iterations.length - 1);
    }

    setActive(index) {
        this.activeIndex = index;
        this.render();
        if (this.onSelect) this.onSelect(index);
    }

    clear() {
        this.iterations = [];
        this.activeIndex = -1;
        this.container.innerHTML = '';
    }

    render() {
        this.container.innerHTML = '';
        for (let i = 0; i < this.iterations.length; i++) {
            const iter = this.iterations[i];
            const el = document.createElement('div');
            el.className = `timeline-item${i === this.activeIndex ? ' active' : ''}`;

            if (iter.images && iter.images.length > 0) {
                const img = document.createElement('img');
                img.src = iter.images[0];
                img.alt = `Iteration ${i + 1}`;
                el.appendChild(img);
            }

            const num = document.createElement('span');
            num.className = 'iter-num';
            num.textContent = i + 1;
            el.appendChild(num);

            el.addEventListener('click', () => this.setActive(i));
            this.container.appendChild(el);
        }
    }

    getActive() {
        return this.activeIndex >= 0 ? this.iterations[this.activeIndex] : null;
    }
}

window.IterationTimeline = IterationTimeline;
