/**
 * SurgicalAI API Client - Clean, Modern Implementation
 * Implements the new structured API contract
 */

// Auto-detect API base URL
function getApiBase() {
    const { protocol, hostname, port } = window.location;
    
    // If served from the same server as the API, use same origin
    if (port === '8000' || port === '7860') {
        return `${protocol}//${hostname}:${port}`;
    }
    
    // Default to local development server
    return `${protocol}//${hostname}:8000`;
}

const API_BASE = window.SURGICALAI_API_BASE || getApiBase();

class SurgicalAI {
    constructor() {
        this.baseUrl = API_BASE;
        this.defaultHeaders = {
            'Accept': 'application/json',
            'Cache-Control': 'no-cache'
        };
    }

    /**
     * Analyze a lesion image using the new /api/analyze endpoint
     */
    async analyze(imageFile, roi, options = {}) {
        const formData = new FormData();
        formData.append('file', imageFile);
        
        const payload = {
            roi: roi,
            site: options.site || null,
            risk_factors: {
                age: options.age || null,
                sex: options.sex || null,
                h_zone: options.hZone || false,
                ill_defined_borders: options.illDefinedBorders || false,
                recurrent_tumor: options.recurrentTumor || false,
                prior_histology: options.priorHistology || false
            },
            offline: options.offline || false
        };
        
        formData.append('payload', JSON.stringify(payload));
        if (options.offline) {
            formData.append('offline', 'true');
        }

        const response = await fetch(`${this.baseUrl}/api/analyze`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const error = await this.handleError(response);
            throw error;
        }

        const result = await response.json();
        const requestId = response.headers.get('X-Request-ID');
        
        return {
            ...result,
            requestId
        };
    }

    /**
     * Legacy compatibility method for existing UI
     */
    async postAnalyze(options) {
        const { file, site, suspected, flags = {} } = options;
        
        if (!file || !site) {
            throw new Error('File and site are required');
        }

        // Convert to new API format
        const roi = { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }; // Default center ROI
        
        const analysisOptions = {
            site: site,
            age: flags.age,
            sex: flags.sex,
            hZone: flags.h_zone,
            illDefinedBorders: flags.ill_defined_borders,
            recurrentTumor: flags.recurrent,
            priorHistology: flags.prior_histology,
            offline: flags.offline || false
        };

        const result = await this.analyze(file, roi, analysisOptions);
        
        // Convert back to legacy format for compatibility
        return {
            json: result,
            artifacts: this.processArtifacts(result.artifacts || {})
        };
    }

    /**
     * Generate PDF report from analysis results
     */
    async generateReport(analysisPayload, doctorName = null) {
        const formData = new FormData();
        formData.append('analysis_payload', JSON.stringify(analysisPayload));
        if (doctorName) {
            formData.append('doctor_name', doctorName);
        }

        const response = await fetch(`${this.baseUrl}/api/report`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await this.handleError(response);
            throw error;
        }

        // Return blob for PDF download
        const blob = await response.blob();
        const requestId = response.headers.get('X-Request-ID');
        
        return { blob, requestId };
    }

    /**
     * Health check endpoint
     */
    async checkHealth() {
        const response = await fetch(`${this.baseUrl}/healthz`, {
            method: 'GET',
            headers: this.defaultHeaders
        });

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.status}`);
        }

        const data = await response.json();
        return {
            ok: data.status === 'ok' || data.ok === true,
            provider: 'SurgicalAI',
            model: 'GPT-4O',
            fallback: false,
            time: new Date().toISOString()
        };
    }

    /**
     * Legacy getHealth method for compatibility
     */
    async getHealth() {
        try {
            return await this.checkHealth();
        } catch (error) {
            return {
                ok: false,
                provider: 'Offline',
                model: 'Unknown',
                fallback: false,
                time: new Date().toISOString(),
                error: error.message
            };
        }
    }

    /**
     * Get version information
     */
    async getVersion() {
        const response = await fetch(`${this.baseUrl}/version`, {
            headers: this.defaultHeaders
        });

        if (!response.ok) {
            throw new Error(`Version check failed: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get facial subunits (legacy compatibility)
     */
    async getFacialSubunits() {
        const response = await fetch(`${this.baseUrl}/api/facial-subunits`, {
            headers: this.defaultHeaders
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch facial subunits: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get usage logs (legacy compatibility)
     */
    async getLastUsage(limit = 10) {
        // Return empty array for now - could be implemented later
        return [];
    }

    /**
     * Handle API errors consistently
     */
    async handleError(response) {
        const requestId = response.headers.get('X-Request-ID') || 'unknown';
        
        try {
            const errorData = await response.json();
            const error = new Error(errorData.detail || `HTTP ${response.status}`);
            error.status = response.status;
            error.requestId = requestId;
            error.details = errorData;
            return error;
        } catch {
            const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
            error.status = response.status;
            error.requestId = requestId;
            return error;
        }
    }

    /**
     * Process artifacts from response for legacy compatibility
     */
    processArtifacts(artifacts) {
        const processed = {};
        
        if (artifacts.overlay_png_base64) {
            processed.overlay = {
                url: `data:image/png;base64,${artifacts.overlay_png_base64}`,
                path: 'overlay.png'
            };
        }
        
        if (artifacts.heatmap_png_base64) {
            processed.heatmap = {
                url: `data:image/png;base64,${artifacts.heatmap_png_base64}`,
                path: 'heatmap.png'
            };
            processed.gradcam = processed.heatmap; // Alias
        }
        
        return processed;
    }

    /**
     * Download artifact by path
     */
    getArtifactUrl(path) {
        return `${this.baseUrl}/api/artifact/${path}`;
    }

    /**
     * Streaming analysis (legacy compatibility - returns promise)
     */
    async openStream(options) {
        const result = await this.postAnalyze(options);
        
        return {
            async* stream() {
                yield { type: 'chunk', content: 'Starting analysis...' };
                yield { type: 'chunk', content: 'Processing image...' };
                yield { type: 'chunk', content: 'Running AI model...' };
                yield { type: 'chunk', content: 'Generating report...' };
                yield { type: 'complete', result: result.json };
            }
        };
    }

    /**
     * Utility: Format file size
     */
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Legacy utility methods for compatibility
    formatFileSize(bytes) { return SurgicalAI.formatFileSize(bytes); }
    formatProb(prob) { return Math.round((prob || 0) * 100); }
    barWidth(prob) { return Math.round((prob || 0) * 100); }

    /**
     * Utility: Validate image file
     */
    static validateImage(file) {
        const maxSize = 6 * 1024 * 1024; // 6MB
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

        if (!allowedTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please upload JPEG, PNG, or WebP images.');
        }

        if (file.size > maxSize) {
            throw new Error('File too large. Maximum size is 6MB. Try compressing the image or using WebP format.');
        }

        return true;
    }

    /**
     * Utility: Create ROI from two click points
     */
    static createROI(point1, point2, imageWidth, imageHeight) {
        const x1 = Math.min(point1.x, point2.x);
        const y1 = Math.min(point1.y, point2.y);
        const x2 = Math.max(point1.x, point2.x);
        const y2 = Math.max(point1.y, point2.y);

        return {
            x: x1 / imageWidth,
            y: y1 / imageHeight,
            width: (x2 - x1) / imageWidth,
            height: (y2 - y1) / imageHeight
        };
    }
}

// Export for module systems and make globally available
window.SurgicalAI = SurgicalAI;
window.API_BASE = API_BASE;

// Create default instance
const surgicalAI = new SurgicalAI();
window.surgicalAI = surgicalAI;

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SurgicalAI };
}
