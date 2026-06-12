// Prompt Forge — Composition library browser

export class Library {
    constructor(modalEl) {
        this.modal = modalEl;
        this.grid = modalEl.querySelector('#library-grid');
        this.searchInput = modalEl.querySelector('#library-search');
        this.closeBtn = modalEl.querySelector('#close-library');

        this.closeBtn.addEventListener('click', () => this.hide());
        this.searchInput.addEventListener('input', this.debounce(() => this.search(), 300));
    }

    show() {
        this.modal.classList.remove('hidden');
        this.load();
    }

    hide() {
        this.modal.classList.add('hidden');
    }

    async load() {
        const data = await API.getLibrary();
        this.renderGrid(data.results || []);
    }

    async search() {
        const q = this.searchInput.value.trim();
        const data = await API.getLibrary(q);
        this.renderGrid(data.results || []);
    }

    renderGrid(compositions) {
        this.grid.innerHTML = '';
        if (compositions.length === 0) {
            this.grid.innerHTML = '<p style="color: var(--text-muted); grid-column: 1/-1;">No compositions yet. Run a forge to build your library.</p>';
            return;
        }
        for (const comp of compositions) {
            const card = document.createElement('div');
            card.className = 'library-card';
            card.innerHTML = `
                ${comp.image_path ? `<img src="/api/image/${comp.image_path.split('/').pop()}" alt="">` : ''}
                <div class="card-info">
                    <div>${(comp.description || '').substring(0, 50)}</div>
                    <div>Score: ${(comp.final_score || 0).toFixed(2)} | ${comp.iteration_count || 0} iters</div>
                </div>
            `;
            this.grid.appendChild(card);
        }
    }

    debounce(fn, ms) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), ms);
        };
    }
}

window.Library = Library;
