const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');

let currentFile = null;

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#6366f1';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#94a3b8';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#94a3b8';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.style.display = 'none';
        previewArea.style.display = 'block';
        analyzeBtn.disabled = false;

        // Show results section with placeholders
        resultsSection.style.display = 'block';
        const hybridResult = document.getElementById('hybrid-result');
        const hybridConfidence = document.getElementById('hybrid-confidence');
        if (hybridResult) {
            hybridResult.textContent = '--';
            hybridResult.className = 'result-value';
        }
        if (hybridConfidence) {
            hybridConfidence.textContent = '--';
        }
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    dropZone.style.display = 'block';
    previewArea.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
});

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Analysis failed'));
        }
    } catch (error) {

        alert('An error occurred during analysis.');
    } finally {
        analyzeBtn.textContent = 'Analyze Image';
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {

    resultsSection.style.display = 'block';

    const hybridResult = document.getElementById('hybrid-result');
    const hybridConfidence = document.getElementById('hybrid-confidence');

    // Hybrid
    if (data.hybrid) {
        if (hybridResult) {
            hybridResult.textContent = data.hybrid.prediction;
            hybridResult.className = 'result-value ' + data.hybrid.prediction.toLowerCase();
        }
        if (hybridConfidence) {
            hybridConfidence.textContent = (data.hybrid.confidence * 100).toFixed(1);
        }
    } else {

    }
}
