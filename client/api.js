// SurgicalAI Client Application
document.addEventListener('DOMContentLoaded', function() {
  const imageUpload = document.getElementById('imageUpload');
  const imagePreview = document.getElementById('imagePreview');
  const dropArea = document.getElementById('dropArea');
  const siteDropdown = document.getElementById('site');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const progressSteps = document.getElementById('progressSteps');
  const resultsPanel = document.getElementById('resultsPanel');
  const probabilitiesContainer = document.getElementById('probabilitiesContainer');
  const gateDecision = document.getElementById('gateDecision');
  const artifacts = document.getElementById('artifacts');
  const toast = document.getElementById('toast');

  let currentImageFile = null;
  let formState = {
    image: null,
    subunit: '',
    age: null,
    sex: 'Male',
    prior_histology: false,
    ill_defined_borders: false,
    recurrent: false
  };

  // Load form state from localStorage
  function loadFormState() {
    const saved = localStorage.getItem('surgicalai_form');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        formState = { ...formState, ...parsed };
        
        // Restore form values
        if (formState.age) document.getElementById('age').value = formState.age;
        document.getElementById('sex').value = formState.sex;
        document.getElementById('priorHistology').checked = formState.prior_histology;
        document.getElementById('illDefinedBorders').checked = formState.ill_defined_borders;
        document.getElementById('recurrent').checked = formState.recurrent;
      } catch (e) {
        console.warn('Failed to load saved form state:', e);
      }
    }
  }

  // Save form state to localStorage
  function saveFormState() {
    localStorage.setItem('surgicalai_form', JSON.stringify(formState));
  }

  // Show toast message
  function showToast(message, duration = 5000) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, duration);
  }

  // Update form state and validate
  function updateFormState() {
    formState.age = document.getElementById('age').value ? parseInt(document.getElementById('age').value) : null;
    formState.sex = document.getElementById('sex').value;
    formState.subunit = siteDropdown.value;
    formState.prior_histology = document.getElementById('priorHistology').checked;
    formState.ill_defined_borders = document.getElementById('illDefinedBorders').checked;
    formState.recurrent = document.getElementById('recurrent').checked;
    
    saveFormState();
    
    // Enable/disable analyze button
    const hasImage = currentImageFile !== null;
    const hasSubunit = formState.subunit !== '';
    analyzeBtn.disabled = !(hasImage && hasSubunit);
  }

  // Load facial subunits from API
  async function loadFacialSubunits() {
    // Provide immediate visual feedback while loading
    siteDropdown.innerHTML = '<option value="" disabled selected>Loading facial subunits...</option>';
    siteDropdown.disabled = true;
    const fallback = [
      { value: 'forehead', label: 'Forehead' },
      { value: 'temple', label: 'Temple' },
      { value: 'cheek_medial', label: 'Medial Cheek' },
      { value: 'cheek_lateral', label: 'Lateral Cheek' },
      { value: 'nose_dorsum', label: 'Nasal Dorsum' },
      { value: 'nose_tip', label: 'Nasal Tip' },
      { value: 'nose_ala', label: 'Nasal Ala' },
      { value: 'lip_upper', label: 'Upper Lip' },
      { value: 'lip_lower', label: 'Lower Lip' },
      { value: 'chin', label: 'Chin' }
    ];
    try {
      const response = await fetch('/api/facial-subunits', { cache: 'no-store' });
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const data = await response.json();

      const items = Array.isArray(data?.subunits) && data.subunits.length ? data.subunits : fallback;
      siteDropdown.innerHTML = '<option value="" disabled selected>Select facial subunit...</option>';
      items.forEach(item => {
        const option = document.createElement('option');
        option.value = item.value;
        option.textContent = item.label;
        siteDropdown.appendChild(option);
      });
      if (formState.subunit && items.some(i => i.value === formState.subunit)) {
        siteDropdown.value = formState.subunit;
      }
      siteDropdown.disabled = false;
    } catch (error) {
      console.error('Failed to load facial subunits from API, using fallback:', error);
      // Fallback population
      siteDropdown.innerHTML = '<option value="" disabled selected>Select facial subunit...</option>';
      fallback.forEach(item => {
        const option = document.createElement('option');
        option.value = item.value;
        option.textContent = item.label;
        siteDropdown.appendChild(option);
      });
      siteDropdown.disabled = false;
      showToast('Loaded fallback facial subunits (offline mode).');
    }
    // Ensure state + button refresh after population
    updateFormState();
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

  function previewImage(file) {
    if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
      showToast('Please upload a JPEG or PNG image.');
      return;
    }

    currentImageFile = file;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      dropArea.classList.add('has-image');
      updateFormState();
    };
    reader.readAsDataURL(file);
  }

  // Set up form listeners
  function setupFormListeners() {
    ['age', 'sex'].forEach(id => {
      document.getElementById(id).addEventListener('change', updateFormState);
    });
    
    ['priorHistology', 'illDefinedBorders', 'recurrent'].forEach(id => {
      document.getElementById(id).addEventListener('change', updateFormState);
    });
    
    siteDropdown.addEventListener('change', updateFormState);
  }

  // Analyze button
  async function setupAnalyzeButton() {
    analyzeBtn.addEventListener('click', async function() {
      if (!currentImageFile || !formState.subunit) {
        showToast('Please upload an image and select a facial subunit.');
        return;
      }

      // Show progress
      progressSteps.style.display = 'block';
      resultsPanel.style.display = 'none';
      progressSteps.innerHTML = '<p>Analyzing image...</p>';

      try {
        // Create FormData for multipart/form-data
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('payload', JSON.stringify(formState));

        // Send to API
        const response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Server responded with ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        
        if (!result.ok) {
          throw new Error(result.error || 'Analysis failed');
        }

        // Display results
        displayResults(result);
      } catch (error) {
        progressSteps.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        console.error('Analysis failed:', error);
        showToast(`Analysis failed: ${error.message}`);
      }
    });
  }

  function displayResults(data) {
    // Hide progress, show results
    progressSteps.style.display = 'none';
    resultsPanel.style.display = 'block';

    // Probabilities with bars
    probabilitiesContainer.innerHTML = '';
    if (data.lesion_probs) {
      Object.entries(data.lesion_probs).forEach(([className, prob]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        const percentage = Math.round(prob * 100);
        item.innerHTML = `
          <span class="probability-label">${className.replace('_', ' ')}</span>
          <div class="probability-bar">
            <div class="probability-fill" style="width: ${percentage}%"></div>
          </div>
          <span class="probability-value">${percentage}%</span>
        `;
        
        probabilitiesContainer.appendChild(item);
      });
    }

    // Gate decision
    if (data.gate) {
      const status = data.gate.allow_flap ? 'Proceed' : 'Defer';
      const color = data.gate.allow_flap ? 'proceed' : 'defer';
      gateDecision.innerHTML = `
        <span class="pill ${color}">${status}</span> 
        ${data.gate.reason || ''}<br>
        <small>${data.gate.guidance || ''}</small>
      `;
    }

    // Artifacts
    if (data.artifacts) {
      artifacts.innerHTML = `
        <div class="thumbnails">
          <img src="${data.artifacts.overlay_png}" alt="Overlay" class="thumbnail" />
          <img src="${data.artifacts.heatmap_png}" alt="Heatmap" class="thumbnail" />
        </div>
        <div class="downloads">
          <a href="${data.artifacts.report_pdf}" download>Download PDF</a>
        </div>
      `;
    }
  }

  // Initialize
  loadFormState();
  loadFacialSubunits();
  setupFileUpload();
  setupFormListeners();
  setupAnalyzeButton();
  updateFormState();
});
