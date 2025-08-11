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

    // Probabilities table
    let probsHTML = '<tr><th>Class</th><th>Probability</th></tr>';
    for (const [className, prob] of Object.entries(data.class_probs)) {
      probsHTML += `<tr><td>${className}</td><td>${(prob * 100).toFixed(1)}%</td></tr>`;
    }
    probabilitiesTable.innerHTML = probsHTML;

    // Gate decision
    const gateStatus = data.gate.proceed ? 'Proceed' : 'Defer';
    const gateColor = data.gate.proceed ? 'green' : 'red';
    gateDecision.innerHTML = `<span class="pill ${gateColor}">${gateStatus}</span> ${data.gate.reason || ''}`;

    // Flap suggestions
    flapSuggestions.innerHTML = '';
    if (data.flap_suggestions && data.flap_suggestions.length) {
      data.flap_suggestions.forEach(flap => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = flap;
        flapSuggestions.appendChild(chip);
      });
    } else {
      flapSuggestions.textContent = 'No flap suggestions available.';
    }

    // Risk notes
    riskNotes.innerHTML = '';
    if (data.risk_notes && data.risk_notes.length) {
      const ul = document.createElement('ul');
      data.risk_notes.forEach(note => {
        const li = document.createElement('li');
        li.textContent = note;
        ul.appendChild(li);
      });
      riskNotes.appendChild(ul);
    } else {
      riskNotes.textContent = 'No risk notes available.';
    }

    // Artifacts
    if (data.artifacts) {
      artifacts.innerHTML = `
        <div class="thumbnails">
          <img src="/api/artifact/${data.artifacts.overlay_png.split('/').pop()}" alt="Overlay" class="thumbnail" />
          <img src="/api/artifact/${data.artifacts.heatmap_png.split('/').pop()}" alt="Heatmap" class="thumbnail" />
          <img src="/api/artifact/${data.artifacts.guideline_card_png.split('/').pop()}" alt="Guidelines" class="thumbnail" />
        </div>
        <div class="downloads">
          <a href="/api/artifact/${data.artifacts.report_pdf.split('/').pop()}" class="button" download>Download PDF</a>
          <a href="/api/artifact/${data.artifacts.report_json.split('/').pop()}" class="button" download>Download JSON</a>
        </div>
      `;
    }

    // Citations
    if (data.citations_expanded) {
      const citationsEl = document.getElementById('citations');
      citationsEl.innerHTML = '<h3>Citations</h3><ul>';
      
      data.citations_expanded.forEach(citation => {
        citationsEl.innerHTML += `<li>${citation}</li>`;
      });
      
      citationsEl.innerHTML += '</ul>';
    }
  }

  // Initialize
  loadMetadata();
  setupFileUpload();
  setupAnalyzeButton();
});
