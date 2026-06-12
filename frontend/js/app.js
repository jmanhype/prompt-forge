// Prompt Forge — Main application controller
import { API } from './api.js';
import { ForgeCanvas } from './canvas.js';
import { IterationTimeline } from './iterations.js';
import { Library } from './library.js';

class App {
    constructor() {
        // Canvas
        this.canvas = new ForgeCanvas(document.getElementById('forge-canvas'));
        
        // Timeline
        this.timeline = new IterationTimeline(document.getElementById('timeline-items'));
        this.timeline.onSelect = (i) => this.selectIteration(i);
        
        // Library
        this.library = new Library(document.getElementById('library-modal'));
        
        // State
        this.currentAnalysis = null;
        this.currentFile = null;
        this.forgeRunning = false;
        this.ws = null;
        this.forgeStartTime = null;
        
        this.bindEvents();
        this.checkComfyUI();
    }

    bindEvents() {
        // File input
        const fileInput = document.getElementById('file-input');
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) this.handleFile(e.target.files[0]);
        });

        // Drop zone
        const dropZone = document.getElementById('drop-zone');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files[0]) this.handleFile(e.dataTransfer.files[0]);
        });

        // Text input
        const textInput = document.getElementById('text-input');
        textInput.addEventListener('input', () => {
            document.getElementById('btn-forge').disabled = !textInput.value.trim();
        });

        // Forge button
        document.getElementById('btn-forge').addEventListener('click', () => this.startForge());

        // Nav buttons
        document.getElementById('btn-library').addEventListener('click', () => this.library.show());
        document.getElementById('btn-new').addEventListener('click', () => this.reset());
    }

    async checkComfyUI() {
        const caps = await API.getCapabilities();
        const dot = document.getElementById('comfyui-status');
        const label = document.getElementById('comfyui-label');
        
        if (caps.comfyui_connected) {
            dot.className = 'status-dot online';
            label.textContent = `ComfyUI: ${caps.strategy_description}`;
            document.getElementById('strategy-label').textContent = `Strategy: ${caps.strategy}`;
        } else {
            dot.className = 'status-dot offline';
            label.textContent = 'ComfyUI: offline';
        }
    }

    async handleFile(file) {
        this.currentFile = file;
        this.setStatus('Analyzing image...');
        
        try {
            const result = await API.analyze(file);
            this.currentAnalysis = result;
            
            // Display analysis
            this.showElements(result.elements || []);
            this.showStyle(result.json?.style_description || {});
            
            // Show image on canvas with bboxes
            const url = URL.createObjectURL(file);
            await this.canvas.loadImage(url);
            this.canvas.setElements(result.elements || []);
            
            // Enable forge button
            document.getElementById('btn-forge').disabled = false;
            this.setStatus(`Analyzed: ${result.elements?.length || 0} elements detected`);
            
            // Check LoRA match
            const loras = await API.getLoras();
            if (loras.loras && loras.loras.length > 0) {
                this.showLoraPanel(loras.loras);
            }
        } catch (err) {
            this.setStatus(`Analysis failed: ${err.message}`);
        }
    }

    showElements(elements) {
        const panel = document.getElementById('elements-panel');
        const list = document.getElementById('elements-list');
        panel.classList.remove('hidden');
        
        list.innerHTML = elements.map(e => `
            <div class="region-item">
                <span class="region-dot" style="background: ${e.type === 'text' ? '#8b5cf6' : '#f97316'}"></span>
                <span class="region-label">${e.label}</span>
                <span class="region-score">${(e.confidence * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }

    showStyle(style) {
        if (!style || Object.keys(style).length === 0) return;
        const panel = document.getElementById('style-panel');
        const fields = document.getElementById('style-fields');
        panel.classList.remove('hidden');
        
        fields.innerHTML = Object.entries(style).map(([k, v]) => `
            <div class="region-item">
                <span class="region-label">${k}</span>
                <span class="region-score">${v}</span>
            </div>
        `).join('');
    }

    showLoraPanel(loras) {
        const panel = document.getElementById('lora-panel');
        const info = document.getElementById('lora-info');
        panel.classList.remove('hidden');
        info.innerHTML = loras.slice(0, 5).map(l => `
            <div class="region-item">
                <span class="region-label">${l.filename}</span>
                <span class="region-score">${l.trigger_words?.join(', ') || '—'}</span>
            </div>
        `).join('');
    }

    async startForge() {
        if (this.forgeRunning) return;
        this.forgeRunning = true;
        
        const btn = document.getElementById('btn-forge');
        btn.classList.add('running');
        btn.textContent = '⚒ FORGING...';
        btn.disabled = true;
        
        // Clear previous
        this.timeline.clear();
        this.canvas.clearHeatmap();
        this.clearScores();
        
        this.forgeStartTime = Date.now();
        this.startTimer();
        
        const description = document.getElementById('text-input').value.trim();
        
        try {
            // Start forge session
            const session = await API.startForge(description, this.currentFile);
            this.setStatus(`Forge session ${session.session_id} — connecting...`);
            
            // Connect WebSocket
            this.ws = API.connectForgeWS(session.session_id);
            this.ws.onopen = () => this.setStatus('WebSocket connected — forge running...');
            this.ws.onmessage = (e) => this.handleWSEvent(JSON.parse(e.data));
            this.ws.onclose = () => this.forgeComplete();
            this.ws.onerror = () => {
                // WebSocket failed — fall back to HTTP polling
                this.setStatus('WebSocket unavailable — polling for updates...');
                this.pollForgeStatus(session.session_id);
            };
        } catch (err) {
            this.setStatus(`Forge failed: ${err.message}`);
            this.forgeComplete();
        }
    }

    handleWSEvent(msg) {
        const { type, data } = msg;
        this._lastEventCount = (this._lastEventCount || 0) + 1;
        
        switch (type) {
            case 'connected':
                this.setStatus(`Connected to forge ${data.session_id}`);
                break;
            
            case 'analyzing':
                this.setStatus(`Analyzing: ${data.message}`);
                break;
            
            case 'generating':
                this.setStatus(`Generating iteration ${data.iteration || '?'}/${data.max_iterations || '?'}...`);
                break;
            
            case 'scoring':
                this.setStatus('Scoring output...');
                break;
            
            case 'mutating':
                this.setStatus(`Mutating: ${data.message || 'applying fixes...'}`);
                break;
            
            case 'iteration':
                this.handleIteration(data);
                break;
            
            case 'converged':
                this.handleConverged(data);
                break;
            
            case 'error':
                this.setStatus(`Error: ${data.message}`);
                this.forgeComplete();
                break;
        }
    }

    handleIteration(data) {
        // Add to timeline
        this.timeline.addIteration({
            number: data.number,
            images: data.images?.map(img => API.imageUrl(img)) || [],
            score: data.score,
            diagnosis: data.diagnosis,
            heatmap: data.heatmap,
            mutations: data.mutations || [],
        });
        
        // Update scores panel
        this.updateScores(data.score);
        
        // Show heatmap on canvas
        if (data.heatmap && this.currentAnalysis?.elements) {
            this.canvas.setHeatmap(data.heatmap);
        }
        
        // Update status
        const score = data.score?.overall || 0;
        this.setStatus(`Iteration ${data.number}: score ${(score * 100).toFixed(0)}%${data.score?.converged ? ' — CONVERGED' : ''}`);
    }

    handleConverged(data) {
        this.setStatus(
            data.converged
                ? `Converged in ${data.iterations} iterations (${(data.final_score * 100).toFixed(0)}%) — ${data.total_duration_ms / 1000}s`
                : `Finished ${data.iterations} iterations (${(data.final_score * 100).toFixed(0)}%) — threshold not reached`
        );
        this.forgeComplete();
    }

    async pollForgeStatus(sessionId) {
        this._polling = true;
        this._seenEvents = 0;
        
        while (this._polling && this.forgeRunning) {
            try {
                const resp = await fetch(`${API.baseUrl}/api/forge/${sessionId}`);
                if (!resp.ok) {
                    await new Promise(r => setTimeout(r, 5000));
                    continue;
                }
                
                const data = await resp.json();
                const events = data.events || [];
                
                // Process new events
                for (let i = this._seenEvents; i < events.length; i++) {
                    this.handleWSEvent(events[i]);
                }
                this._seenEvents = events.length;
                
                // Stop polling if done
                if (data.status === 'converged' || data.status === 'error') {
                    this.forgeComplete();
                    return;
                }
            } catch (err) {
                console.error('Poll error:', err);
            }
            
            await new Promise(r => setTimeout(r, 5000));
        }
    }

    forgeComplete() {
        if (!this.forgeRunning) return; // guard against double-call
        this.forgeRunning = false;
        this._polling = false;
        if (this.ws) {
            try { this.ws.close(); } catch(e) {}
            this.ws = null;
        }
        const btn = document.getElementById('btn-forge');
        btn.classList.remove('running');
        btn.textContent = '⚒ FORGE';
        btn.disabled = false;
        this.stopTimer();
    }

    selectIteration(index) {
        const iter = this.timeline.iterations[index];
        if (!iter) return;
        
        // Load image
        if (iter.images && iter.images.length > 0) {
            this.canvas.loadImage(iter.images[0]).then(() => {
                if (iter.heatmap && this.currentAnalysis?.elements) {
                    this.canvas.setElements(this.currentAnalysis.elements);
                    this.canvas.setHeatmap(iter.heatmap);
                }
            });
        }
        
        // Update scores
        if (iter.score) this.updateScores(iter.score);
        
        // Show mutations
        if (iter.mutations && iter.mutations.length > 0) {
            const panel = document.getElementById('mutations-panel');
            const list = document.getElementById('mutations-list');
            panel.classList.remove('hidden');
            list.innerHTML = iter.mutations.map(m => `<div class="mutation-item">${m}</div>`).join('');
        }
    }

    updateScores(score) {
        if (!score) return;
        
        const pct = (v) => `${Math.round(v * 100)}%`;
        
        document.getElementById('overall-bar').style.width = pct(score.overall || 0);
        document.getElementById('overall-value').textContent = pct(score.overall || 0);
        
        document.getElementById('comp-bar').style.width = pct(score.composition || 0);
        document.getElementById('comp-value').textContent = pct(score.composition || 0);
        
        document.getElementById('style-bar').style.width = pct(score.style || 0);
        document.getElementById('style-value').textContent = pct(score.style || 0);
        
        document.getElementById('subject-bar').style.width = pct(score.subject || 0);
        document.getElementById('subject-value').textContent = pct(score.subject || 0);
        
        // Region details
        if (score.regions) {
            const regionList = document.getElementById('region-list');
            regionList.innerHTML = score.regions.map(r => `
                <div class="region-item">
                    <span class="region-dot" style="background: ${r.composite >= 0.8 ? '#22c55e' : r.composite >= 0.6 ? '#eab308' : '#ef4444'}"></span>
                    <span class="region-label">${r.label}</span>
                    <span class="region-score">${(r.composite * 100).toFixed(0)}%</span>
                </div>
            `).join('');
        }
        
        // Diagnosis
        if (score.diagnosis) {
            const diagList = document.getElementById('diagnosis-list');
            diagList.innerHTML = score.diagnosis.map(d => `<div class="diagnosis-item">${d}</div>`).join('');
        }
    }

    clearScores() {
        ['overall', 'comp', 'style', 'subject'].forEach(id => {
            document.getElementById(`${id}-bar`).style.width = '0%';
            document.getElementById(`${id}-value`).textContent = '0%';
        });
        document.getElementById('region-list').innerHTML = '';
        document.getElementById('diagnosis-list').innerHTML = '';
        document.getElementById('mutations-panel').classList.add('hidden');
    }

    setStatus(text) {
        document.getElementById('status-text').textContent = text;
    }

    startTimer() {
        this._timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - this.forgeStartTime) / 1000).toFixed(0);
            document.getElementById('forge-timer').textContent = `${elapsed}s`;
        }, 1000);
    }

    stopTimer() {
        if (this._timerInterval) clearInterval(this._timerInterval);
    }

    reset() {
        this.canvas.clear();
        this.timeline.clear();
        this.clearScores();
        this.currentAnalysis = null;
        this.currentFile = null;
        this.forgeRunning = false;
        document.getElementById('elements-panel').classList.add('hidden');
        document.getElementById('style-panel').classList.add('hidden');
        document.getElementById('lora-panel').classList.add('hidden');
        document.getElementById('btn-forge').disabled = true;
        document.getElementById('text-input').value = '';
        this.setStatus('Ready — drop an image or type a description');
    }
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
