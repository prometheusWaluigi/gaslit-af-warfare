/**
 * GASLIT-AF WARSTACK - Main JavaScript
 * A modular simulation-and-exposure engine
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('GASLIT-AF WARSTACK frontend initialized');
    
    // Initialize tooltips
    initTooltips();
    
    // Initialize file upload handling
    initFileUploads();
    
    // Initialize form validation
    initFormValidation();
    
    // Initialize auto-dismissing alerts
    initAlertDismissal();
    
    // Initialize any charts if present
    initCharts();
});

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize file upload handling with preview
 */
function initFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        const uploadArea = input.closest('.upload-area');
        if (!uploadArea) return;
        
        // File selection change
        input.addEventListener('change', function(e) {
            const fileList = this.files;
            if (fileList.length > 0) {
                updateFilePreview(this, fileList[0]);
            }
        });
        
        // Drag and drop functionality
        if (uploadArea) {
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
                
                if (e.dataTransfer.files.length > 0) {
                    input.files = e.dataTransfer.files;
                    updateFilePreview(input, e.dataTransfer.files[0]);
                    
                    // Trigger change event
                    const event = new Event('change', { bubbles: true });
                    input.dispatchEvent(event);
                }
            });
        }
    });
}

/**
 * Update file preview after selection
 */
function updateFilePreview(inputElement, file) {
    const previewElement = document.querySelector(`#${inputElement.id}-preview`);
    if (!previewElement) return;
    
    const fileSize = formatFileSize(file.size);
    const fileName = file.name;
    const fileType = file.type || 'Unknown';
    
    let previewContent = `
        <div class="alert alert-info">
            <strong>Selected File:</strong> ${fileName}<br>
            <strong>Type:</strong> ${fileType}<br>
            <strong>Size:</strong> ${fileSize}
        </div>
    `;
    
    // If it's an image, show preview
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewContent += `
                <div class="mt-2">
                    <img src="${e.target.result}" class="img-fluid img-thumbnail" alt="Preview">
                </div>
            `;
            previewElement.innerHTML = previewContent;
        };
        reader.readAsDataURL(file);
    } else {
        previewElement.innerHTML = previewContent;
    }
}

/**
 * Format file size in human-readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Initialize form validation
 */
function initFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * Initialize auto-dismissing alerts
 */
function initAlertDismissal() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000); // Auto-dismiss after 5 seconds
    });
}

/**
 * Initialize charts if Chart.js is available
 */
function initCharts() {
    if (typeof Chart === 'undefined') return;
    
    // Example: Initialize dashboard charts
    initDashboardCharts();
}

/**
 * Initialize dashboard charts
 */
function initDashboardCharts() {
    const evidenceChartEl = document.getElementById('evidenceChart');
    if (evidenceChartEl) {
        new Chart(evidenceChartEl, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    label: 'Evidence Accumulation',
                    data: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                    borderColor: 'rgb(220, 53, 69)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Evidence Accumulation Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    const denialChartEl = document.getElementById('denialChart');
    if (denialChartEl) {
        new Chart(denialChartEl, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    label: 'Institutional Denial',
                    data: [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
                    borderColor: 'rgb(13, 110, 253)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Institutional Denial Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
}

/**
 * Format a date string in a human-readable format
 */
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

/**
 * Toggle visibility of an element
 */
function toggleVisibility(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        if (element.style.display === 'none') {
            element.style.display = '';
        } else {
            element.style.display = 'none';
        }
    }
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text, buttonElement) {
    navigator.clipboard.writeText(text).then(() => {
        // Update button text temporarily
        const originalText = buttonElement.innerHTML;
        buttonElement.innerHTML = '<i class="fas fa-check"></i> Copied!';
        
        setTimeout(() => {
            buttonElement.innerHTML = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

/**
 * Download data as JSON file
 */
function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'data.json';
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
}
