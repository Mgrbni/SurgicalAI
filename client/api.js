// SurgicalAI Professional Dashboard API Client
// Clean, modern fetch-based API with streaming support

class SurgicalAI {
    constructor() {
        const DEFAULT_API = `${window.location.protocol}//${window.location.hostname}:7860`;
        this.baseUrl = window.SURGICALAI_API_BASE || DEFAULT_API;
        this.defaultHeaders = {
            'Accept': 'application/json'
        };
    }

    // Health check
    async getHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`, {
                cache: 'no-store'
            });
            
            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                ok: data.ok || false,
                provider: data.provider || 'Unknown',
                model: data.model || 'Unknown',
                fallback: data.fallback || false,
                time: new Date().toISOString()
            };
        } catch (error) {
            console.warn('Health check failed:', error);
            return {
                ok: false,
                provider: 'Offline',
                model: 'N/A',
                fallback: false,
                time: new Date().toISOString()
            };
        }
    }

    // Main analysis endpoint
    async postAnalyze({file, site, suspected, flags = {}}) {
        if (!file || !site) {
            throw new Error('File and site are required');
        }

        const formData = new FormData();
        formData.append('file', file);

        const payload = {
            subunit: site,
            sex: flags.sex || 'Male',
            ...flags
        };

        if (suspected) {
            payload.diagnosis_override = suspected;
        }

        formData.append('payload', JSON.stringify(payload));

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const message = errorData.error?.message || `Server error: ${response.status}`;
            const requestId = response.headers.get('x-request-id') || 'unknown';
            
            throw {
                status: response.status,
                message,
                requestId,
                raw: errorData
            };
        }

        const data = await response.json();
        
        return {
            json: data,
            artifacts: this.processArtifacts(data.artifacts_list || [])
        };
    }

    // Streaming analysis with SSE
    async openStream({file, site, suspected, flags = {}}) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('stream', '1');

        const payload = {
            subunit: site,
            sex: flags.sex || 'Male',
            ...flags
        };

        if (suspected) {
            payload.diagnosis_override = suspected;
        }

        formData.append('payload', JSON.stringify(payload));

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error?.message || `Server error: ${response.status}`);
        }

        return new StreamingAnalysis(response);
    }

    // Process artifacts list into a usable format
    processArtifacts(artifactsList) {
        const artifacts = {};
        
        artifactsList.forEach(artifact => {
            artifacts[artifact.name] = {
                url: `/api/artifact/${artifact.path}`,
                path: artifact.path,
                name: artifact.name
            };
        });

        return artifacts;
    }

    // Get usage logs
    async getLastUsage(limit = 10) {
        try {
            const response = await fetch(`${this.baseUrl}/api/last-usage?limit=${limit}`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('Failed to load usage logs:', error);
        }
        return [];
    }

    // Helper methods
    formatProb = (p) => {
        return Math.round((p || 0) * 100) + '%';
    };

    barWidth = (p) => {
        return Math.round((p || 0) * 100) + '%';
    };

    getArtifactUrl = (path) => {
        return `${this.baseUrl}/api/artifact/${path}`;
    };

    validateImageFile = (file) => {
        if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
            throw new Error('Please upload a valid JPEG or PNG image');
        }
        
        const MAX_SIZE = 10 * 1024 * 1024; // 10MB
        if (file.size > MAX_SIZE) {
            throw new Error('File size must be less than 10MB');
        }
        
        return true;
    };

    formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    normalizeError = (error, response) => {
        return {
            status: response?.status || 500,
            message: error.message || 'Unknown error occurred',
            requestId: response?.headers?.get('x-request-id') || 'unknown',
            details: error
        };
    };
}

// Streaming analysis handler
class StreamingAnalysis {
    constructor(response) {
        this.response = response;
        this.reader = response.body.getReader();
        this.decoder = new TextDecoder();
        this.buffer = '';
        this.chunks = [];
        this.result = null;
    }

    async *stream() {
        try {
            while (true) {
                const { done, value } = await this.reader.read();
                if (done) break;

                this.buffer += this.decoder.decode(value, { stream: true });
                
                // Process complete lines
                const lines = this.buffer.split('\n');
                this.buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        
                        if (data === '[DONE]') {
                            if (this.result) {
                                return this.result;
                            } else {
                                throw new Error('Stream completed without result');
                            }
                        }

                        try {
                            const eventData = JSON.parse(data);
                            
                            if (eventData.type === 'chunk') {
                                this.chunks.push(eventData.content);
                                yield {
                                    type: 'chunk',
                                    content: eventData.content,
                                    accumulated: this.chunks.join('')
                                };
                            } else if (eventData.type === 'complete') {
                                this.result = eventData.result;
                                yield {
                                    type: 'complete',
                                    result: eventData.result
                                };
                            } else if (eventData.type === 'error') {
                                throw new Error(eventData.error);
                            }
                        } catch (e) {
                            console.warn('Failed to parse streaming data:', data);
                        }
                    }
                }
            }
        } finally {
            this.reader.releaseLock();
        }

        throw new Error('Stream ended unexpectedly');
    }

    async collectResult() {
        for await (const event of this.stream()) {
            if (event.type === 'complete') {
                return event.result;
            }
        }
        throw new Error('Stream completed without result');
    }
}

// Cache management
class ResultCache {
    constructor() {
        this.key = 'surgicalai_last_result';
    }

    save(result) {
        try {
            sessionStorage.setItem(this.key, JSON.stringify({
                timestamp: Date.now(),
                data: result
            }));
        } catch (error) {
            console.warn('Failed to cache result:', error);
        }
    }

    load() {
        try {
            const cached = sessionStorage.getItem(this.key);
            if (cached) {
                const { timestamp, data } = JSON.parse(cached);
                // Cache valid for 1 hour
                if (Date.now() - timestamp < 3600000) {
                    return data;
                }
            }
        } catch (error) {
            console.warn('Failed to load cached result:', error);
        }
        return null;
    }

    clear() {
        try {
            sessionStorage.removeItem(this.key);
        } catch (error) {
            console.warn('Failed to clear cache:', error);
        }
    }
}

// Global instance
const surgicalAI = new SurgicalAI();
const resultCache = new ResultCache();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SurgicalAI, StreamingAnalysis, ResultCache };
}
