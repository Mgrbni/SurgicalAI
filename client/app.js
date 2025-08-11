// SurgicalAI Client Application
document.addEventListener('DOMContentLoaded', function() {
  const imageUpload = document.getElementById('imageUpload');
  const imagePreview = document.getElementById('imagePreview');
  const dropArea = document.getElementById('dropArea');
  const siteDropdown = document.getElementById('site');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const progressSteps = document.getElementById('progressSteps');
  const resultsPanel = document.getElementById('resultsPanel');
  const probabilitiesTable = document.getElementById('probabilitiesTable');
  const gateDecision = document.getElementById('gateDecision');
  const flapSuggestions = document.getElementById('flapSuggestions');
  const riskNotes = document.getElementById('riskNotes');
  const artifacts = document.getElementById('artifacts');

  let currentImageFile = null;
  let imageBase64 = null;

  // Load facial subunits from API
  async function loadMetadata() {
    try {
      const response = await fetch('/api/metadata');
      const data = await response.json();
      
      // Populate site dropdown
      if (data.facial_subunits) {
        siteDropdown.innerHTML = '';
        data.facial_subunits.forEach(subunit => {
          const option = document.createElement('option');
          option.value = subunit;
          option.textContent = subunit;
          siteDropdown.appendChild(option);
        });
      }
    } catch (error) {
      console.error('Failed to load metadata:', error);
      progressSteps.innerHTML = '<p class="error">Failed to load facial subunits. Please check if the server is running.</p>';
    }
  }

  // Set up file upload
  function setupFileUpload() {
    // File input change
    imageUpload.addEventListener('change', function() {
      if (this.files && this.files[0]) {
        previewImage(this.files[0]);
      }
    });

    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
      dropArea.classList.add('highlight');
    }

    function unhighlight() {
      dropArea.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length) {
        imageUpload.files = files;
        previewImage(files[0]);
      }
    }
    
    // Click to select file
    dropArea.addEventListener('click', function() {
      imageUpload.click();
    });
  }

  // Convert file to base64
  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  }

  function previewImage(file) {
    if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
      alert('Please upload a JPEG or PNG image.');
      return;
    }

    currentImageFile = file;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      imagePreview.src = e.target.result;
      imageBase64 = e.target.result; // Store base64 data
      imagePreview.style.display = 'block';
      dropArea.classList.add('has-image');
    };
    reader.readAsDataURL(file);
  }

  // Analyze button
  async function setupAnalyzeButton() {
    analyzeBtn.addEventListener('click', async function() {
      if (!currentImageFile) {
        alert('Please upload an image first.');
        return;
      }

      // Show progress
      progressSteps.style.display = 'block';
      resultsPanel.style.display = 'none';
      progressSteps.innerHTML = '<p>Analyzing image...</p>';

      try {
        // Create request payload with base64 image
        const payload = {
          image: {
            filename: currentImageFile.name,
            data: imageBase64
          },
          site: siteDropdown.value,
          age: document.getElementById('age').value ? parseInt(document.getElementById('age').value) : null,
          sex: document.getElementById('sex').value,
          prior_histology: document.getElementById('priorHistology').value,
          ill_defined_borders: document.getElementById('illDefinedBorders').checked,
          recurrent: document.getElementById('recurrent').checked
        };

        // Send to API
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}`);
        }

        const result = await response.json();
        
        if (result.error) {
          progressSteps.innerHTML = `<p class="error">Error: ${result.error.message}</p>`;
          return;
        }

        // Display results
        displayResults(result);
      } catch (error) {
        progressSteps.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        console.error('Analysis failed:', error);
      }
    });
  }

  function displayResults(data) {
    // Hide progress, show results
    progressSteps.style.display = 'none';
    resultsPanel.style.display = 'block';

    // Probabilities table - handle both legacy and new format
    let probsHTML = '<tr><th>Class</th><th>Probability</th></tr>';
    const probs = data.probs || data.class_probs || data.lesion_probs || {};
    
    // Convert to array and sort by probability
    const probArray = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    
    probArray.forEach(([className, prob]) => {
      const displayName = className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      const percentage = typeof prob === 'number' ? (prob * 100).toFixed(1) : 'N/A';
      const barWidth = typeof prob === 'number' ? (prob * 100) : 0;
      
      probsHTML += `
        <tr>
          <td>${displayName}</td>
          <td>
            <div class="prob-bar-container">
              <div class="prob-bar" style="width: ${barWidth}%"></div>
              <span class="prob-text">${percentage}%</span>
            </div>
          </td>
        </tr>`;
    });
    probabilitiesTable.innerHTML = probsHTML;

    // Enhanced gate decision
    const gateStatus = data.gate?.status || (data.gate?.proceed ? 'Proceed' : 'Defer');
    const gateNotes = data.gate?.notes || data.gate?.reason || '';
    const allowFlap = data.gate?.allow_flap !== undefined ? data.gate.allow_flap : data.gate?.proceed;
    
    let gateColor = 'gray';
    if (gateStatus.includes('RED') || gateStatus.includes('urgent')) gateColor = 'red';
    else if (gateStatus.includes('ORANGE') || gateStatus.includes('likely')) gateColor = 'orange';
    else if (gateStatus.includes('YELLOW') || gateStatus.includes('possible')) gateColor = 'yellow';
    else if (gateStatus.includes('GREEN') || allowFlap) gateColor = 'green';
    else if (gateStatus.includes('UNCERTAIN')) gateColor = 'purple';
    
    gateDecision.innerHTML = `
      <div class="gate-decision">
        <span class="pill ${gateColor}">${gateStatus}</span>
        ${gateNotes ? `<p class="gate-notes">${gateNotes}</p>` : ''}
      </div>`;

    // Display VLM Observer information if available
    if (data.observer) {
      displayObserverInfo(data.observer);
    }

    // Display fusion information if available
    if (data.fusion) {
      displayFusionInfo(data.fusion);
    }

    // Display top-3 predictions if available
    if (data.top3 && data.top3.length > 0) {
      displayTop3Predictions(data.top3);
    }

    // Flap suggestions - handle multiple formats
    flapSuggestions.innerHTML = '';
    const flaps = data.flap_suggestions || data.plan?.candidates || [];
    if (flaps && flaps.length) {
      flaps.forEach(flap => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = typeof flap === 'string' ? flap : flap.flap || flap.name || 'Unknown';
        flapSuggestions.appendChild(chip);
      });
    } else {
      flapSuggestions.textContent = 'No flap suggestions available.';
    }

    // Risk notes
    riskNotes.innerHTML = '';
    const notes = data.risk_notes || data.plan?.danger || [];
    if (notes && notes.length) {
      const ul = document.createElement('ul');
      notes.forEach(note => {
        const li = document.createElement('li');
        li.textContent = typeof note === 'string' ? note : note.message || note;
        ul.appendChild(li);
      });
      riskNotes.appendChild(ul);
    } else {
      riskNotes.textContent = 'No risk notes available.';
    }

    // Enhanced artifacts display
    displayArtifacts(data.artifacts);
  }

  function displayObserverInfo(observer) {
    // Create observer info section if it doesn't exist
    let observerSection = document.getElementById('observerInfo');
    if (!observerSection) {
      observerSection = document.createElement('div');
      observerSection.id = 'observerInfo';
      observerSection.className = 'info-section';
      resultsPanel.insertBefore(observerSection, artifacts);
    }

    const primaryPattern = observer.primary_pattern ? 
      observer.primary_pattern.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Unknown';
    
    const descriptors = observer.descriptors || [];
    const recommendation = observer.recommendation || 'None';
    const abcd = observer.abcd_estimate || {};

    observerSection.innerHTML = `
      <h3>🔍 Vision-LLM Observer</h3>
      <div class="observer-content">
        <div class="observer-primary">
          <strong>Primary Pattern:</strong> ${primaryPattern}
        </div>
        <div class="observer-descriptors">
          <strong>Key Descriptors:</strong>
          ${descriptors.length > 0 ? 
            descriptors.map(d => `<span class="descriptor-tag">${d}</span>`).join(' ') :
            '<span class="no-data">None identified</span>'
          }
        </div>
        <div class="observer-recommendation">
          <strong>Recommendation:</strong> <span class="pill ${getRecommendationColor(recommendation)}">${recommendation}</span>
        </div>
        ${abcd.asymmetry ? `
          <div class="abcd-summary">
            <strong>ABCD Assessment:</strong>
            A=${abcd.asymmetry}, B=${abcd.border}, C=${abcd.color}, D=${abcd.diameter_mm || '?'}mm
          </div>
        ` : ''}
      </div>`;
  }

  function displayFusionInfo(fusion) {
    // Create fusion info section if it doesn't exist
    let fusionSection = document.getElementById('fusionInfo');
    if (!fusionSection) {
      fusionSection = document.createElement('div');
      fusionSection.id = 'fusionInfo';
      fusionSection.className = 'info-section';
      resultsPanel.insertBefore(fusionSection, artifacts);
    }

    const notes = fusion.notes || '';
    const descriptors = fusion.vlm_descriptors || [];
    const cnnWeight = fusion.cnn_weight || 0.7;
    const vlmWeight = fusion.vlm_weight || 0.3;

    fusionSection.innerHTML = `
      <h3>🧠 Neuro-Symbolic Fusion</h3>
      <div class="fusion-content">
        <div class="fusion-weights">
          <span class="weight-indicator">CNN: ${(cnnWeight * 100).toFixed(0)}%</span>
          <span class="weight-indicator">VLM: ${(vlmWeight * 100).toFixed(0)}%</span>
        </div>
        ${notes ? `<div class="fusion-notes">${notes}</div>` : ''}
        ${descriptors.length > 0 ? `
          <div class="fusion-descriptors">
            <strong>Contributing Descriptors:</strong>
            ${descriptors.map(d => `<span class="descriptor-tag">${d}</span>`).join(' ')}
          </div>
        ` : ''}
      </div>`;
  }

  function displayTop3Predictions(top3) {
    // Find or create top3 section
    let top3Section = document.getElementById('top3Predictions');
    if (!top3Section) {
      top3Section = document.createElement('div');
      top3Section.id = 'top3Predictions';
      top3Section.className = 'info-section';
      resultsPanel.insertBefore(top3Section, probabilitiesTable.parentElement);
    }

    const top3HTML = top3.map((item, index) => {
      const [className, prob] = Array.isArray(item) ? item : [item.class || item.name, item.probability || item.prob];
      const displayName = className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      const percentage = (prob * 100).toFixed(1);
      const medal = ['🥇', '🥈', '🥉'][index] || '';
      
      return `
        <div class="top3-item rank-${index + 1}">
          <span class="medal">${medal}</span>
          <span class="class-name">${displayName}</span>
          <span class="probability">${percentage}%</span>
        </div>`;
    }).join('');

    top3Section.innerHTML = `
      <h3>🏆 Top 3 Predictions</h3>
      <div class="top3-container">
        ${top3HTML}
      </div>`;
  }

  function displayArtifacts(artifactsData) {
    if (!artifactsData) {
      artifacts.innerHTML = '<p>No artifacts available.</p>';
      return;
    }

    // Prioritize zoom overlay, then full overlay, then regular overlay
    const overlayUrl = artifactsData.overlay_zoom || artifactsData.overlay_full || 
                      artifactsData.overlay || artifactsData.overlay_png;
    
    const heatmapUrl = artifactsData.heatmap || artifactsData.heatmap_png;
  const reportUrl = artifactsData.pdf || artifactsData.report_pdf || artifactsData.report;

    let artifactsHTML = '<div class="artifacts-grid">';
    
    if (overlayUrl) {
      artifactsHTML += `
        <div class="artifact-item">
          <h4>🎯 Lesion Analysis</h4>
          <img src="${overlayUrl}" alt="Lesion Overlay" class="artifact-image" 
               onclick="openImageModal('${overlayUrl}', 'Lesion Analysis Overlay')">
          <div class="artifact-controls">
            <button onclick="downloadArtifact('${overlayUrl}', 'lesion-overlay.png')" class="btn-small">
              📥 Download
            </button>
          </div>
        </div>`;
    }

    if (heatmapUrl) {
      artifactsHTML += `
        <div class="artifact-item">
          <h4>🔥 Activation Heatmap</h4>
          <img src="${heatmapUrl}" alt="Activation Heatmap" class="artifact-image"
               onclick="openImageModal('${heatmapUrl}', 'Grad-CAM Activation Heatmap')">
          <div class="artifact-controls">
            <button onclick="downloadArtifact('${heatmapUrl}', 'heatmap.png')" class="btn-small">
              📥 Download
            </button>
          </div>
        </div>`;
    }

    if (reportUrl) {
      artifactsHTML += `
        <div class="artifact-item">
          <h4>📄 Clinical Report</h4>
          <div class="pdf-preview">
            <div class="pdf-icon">📋</div>
            <p>Comprehensive analysis report</p>
          </div>
          <div class="artifact-controls">
            <button type="button" onclick="window.open('${reportUrl}', '_blank')" class="btn-small">
              👁️ View
            </button>
            <button type="button" onclick="downloadArtifact('${reportUrl}', 'report.pdf')" class="btn-small">
              📥 Download
            </button>
          </div>
        </div>`;
    }

    // Add toggle for full vs zoom overlay if both available
    if (artifactsData.overlay_full && artifactsData.overlay_zoom) {
      artifactsHTML += `
        <div class="overlay-toggle">
          <button onclick="toggleOverlayView('${artifactsData.overlay_full}', '${artifactsData.overlay_zoom}')" 
                  class="btn-toggle" id="overlayToggleBtn">
            🔄 Switch to Full View
          </button>
        </div>`;
    }

    artifactsHTML += '</div>';
    artifacts.innerHTML = artifactsHTML;
  }

  function getRecommendationColor(recommendation) {
    if (!recommendation) return 'gray';
    const rec = recommendation.toLowerCase();
    if (rec.includes('observe')) return 'green';
    if (rec.includes('dermoscopy')) return 'yellow';
    if (rec.includes('biopsy')) return 'orange';
    if (rec.includes('mohs') || rec.includes('wle')) return 'red';
    return 'gray';
  }

  // Utility functions for enhanced features
  window.openImageModal = function(imageUrl, title) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>${title}</h3>
          <button class="modal-close" onclick="this.closest('.image-modal').remove()">×</button>
        </div>
        <div class="modal-body">
          <img src="${imageUrl}" alt="${title}" class="modal-image">
        </div>
      </div>`;
    
    document.body.appendChild(modal);
    
    // Close on background click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.remove();
      }
    });
  };

  window.downloadArtifact = function(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  window.toggleOverlayView = function(fullUrl, zoomUrl) {
    const currentOverlay = document.querySelector('.artifact-image[src*="overlay"]');
    const toggleBtn = document.getElementById('overlayToggleBtn');
    
    if (currentOverlay && toggleBtn) {
      if (currentOverlay.src.includes('full')) {
        currentOverlay.src = zoomUrl;
        toggleBtn.textContent = '🔄 Switch to Full View';
      } else {
        currentOverlay.src = fullUrl;
        toggleBtn.textContent = '🔄 Switch to Zoom View';
      }
    }
  };

  // Initialize the application
  loadMetadata();
  setupFileUpload();
  setupAnalyzeButton();
});
