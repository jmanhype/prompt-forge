// Prompt Forge — API client (REST + WebSocket)

export const API = {
    baseUrl: window.location.origin,

    async analyze(file) {
        const form = new FormData();
        if (file instanceof File) {
            form.append('file', file);
        } else {
            form.append('text', file);
        }
        const resp = await fetch(`${this.baseUrl}/api/analyze`, { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`Analyze failed: ${resp.status}`);
        return resp.json();
    },

    async startForge(description, file, maxIterations, threshold) {
        const form = new FormData();
        if (description) form.append('description', description);
        if (file) form.append('file', file);
        if (maxIterations) form.append('max_iterations', maxIterations);
        if (threshold) form.append('threshold', threshold);

        const resp = await fetch(`${this.baseUrl}/api/forge`, { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`Forge start failed: ${resp.status}`);
        return resp.json();
    },

    async getCapabilities() {
        const resp = await fetch(`${this.baseUrl}/api/capabilities`);
        if (!resp.ok) return { comfyui_connected: false };
        return resp.json();
    },

    async getLoras() {
        const resp = await fetch(`${this.baseUrl}/api/loras`);
        if (!resp.ok) return { loras: [] };
        return resp.json();
    },

    async rescanLoras() {
        const resp = await fetch(`${this.baseUrl}/api/rescan-loras`, { method: 'POST' });
        return resp.json();
    },

    async getLibrary(query = '', limit = 50) {
        const params = query ? `?q=${encodeURIComponent(query)}&limit=${limit}` : `?limit=${limit}`;
        const resp = await fetch(`${this.baseUrl}/api/library${params}`);
        return resp.json();
    },

    async getLibraryStats() {
        const resp = await fetch(`${this.baseUrl}/api/library/stats`);
        return resp.json();
    },

    connectForgeWS(sessionId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws/forge/${sessionId}`);
        return ws;
    },

    imageUrl(filename) {
        return `${this.baseUrl}/api/image/${filename.split('/').pop()}`;
    }
};

// Make available globally for other modules
window.API = API;
