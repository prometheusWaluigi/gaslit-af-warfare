/**
 * GASLIT-AF WARSTACK - Main JavaScript
 * This file contains common functionality for the GASLIT-AF WARSTACK frontend.
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('GASLIT-AF WARSTACK frontend initialized');
    
    // Initialize all components
    initializeTooltips();
    initializePopovers();
    initializeDropdowns();
    initializeAlerts();
    initializeCharts();
    setupEventListeners();
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize Bootstrap popovers
 */
function initializePopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Initialize Bootstrap dropdowns
 */
function initializeDropdowns() {
    const dropdownElementList = [].slice.call(document.querySelectorAll('.dropdown-toggle'));
    dropdownElementList.map(function(dropdownToggleEl) {
        return new bootstrap.Dropdown(dropdownToggleEl);
    });
}

/**
 * Initialize dismissible alerts
 */
function initializeAlerts() {
    const alertList = document.querySelectorAll('.alert');
    alertList.forEach(function(alert) {
        if (!alert.classList.contains('alert-permanent')) {
            // Auto-dismiss alerts after 5 seconds
            setTimeout(function() {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }, 5000);
        }
    });
}

/**
 * Initialize Chart.js charts if they exist on the page
 */
function initializeCharts() {
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded. Charts will not be initialized.');
        return;
    }
    
    // Set default Chart.js options
    Chart.defaults.font.family = "'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif";
    Chart.defaults.color = '#666';
    Chart.defaults.responsive = true;
    
    // Custom chart initialization can be added here
    // This will be called by page-specific scripts
}

/**
 * Set up global event listeners
 */
function setupEventListeners() {
    // Back to top button
    const backToTopBtn = document.getElementById('back-to-top');
    if (backToTopBtn) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                backToTopBtn.classList.add('show');
            } else {
                backToTopBtn.classList.remove('show');
            }
        });
        
        backToTopBtn.addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * Format a number with commas as thousands separators
 * @param {number} x - The number to format
 * @returns {string} The formatted number
 */
function formatNumber(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Format a date string to a more readable format
 * @param {string} dateString - The date string to format
 * @returns {string} The formatted date
 */
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

/**
 * Show a loading spinner in a container
 * @param {string} containerId - The ID of the container element
 * @param {string} message - Optional message to display
 */
function showLoading(containerId, message = 'Loading...') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const spinner = document.createElement('div');
    spinner.className = 'text-center py-5';
    spinner.innerHTML = `
        <div class="spinner-border text-primary mb-3" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p>${message}</p>
    `;
    
    container.innerHTML = '';
    container.appendChild(spinner);
}

/**
 * Hide the loading spinner and replace with content
 * @param {string} containerId - The ID of the container element
 * @param {string} content - The HTML content to display
 */
function hideLoading(containerId, content) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = content;
}

/**
 * Show a toast notification
 * @param {string} title - The toast title
 * @param {string} message - The toast message
 * @param {string} type - The toast type (success, error, warning, info)
 */
function showToast(title, message, type = 'info') {
    // Check if the toast container exists, if not create it
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = '1050';
        document.body.appendChild(toastContainer);
    }
    
    // Create a unique ID for the toast
    const toastId = 'toast-' + Date.now();
    
    // Determine the toast class based on type
    let bgClass = 'bg-info';
    switch (type) {
        case 'success':
            bgClass = 'bg-success';
            break;
        case 'error':
            bgClass = 'bg-danger';
            break;
        case 'warning':
            bgClass = 'bg-warning';
            break;
    }
    
    // Create the toast element
    const toastEl = document.createElement('div');
    toastEl.id = toastId;
    toastEl.className = `toast ${bgClass} text-white`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    toastEl.innerHTML = `
        <div class="toast-header">
            <strong class="me-auto">${title}</strong>
            <small>Just now</small>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // Add the toast to the container
    toastContainer.appendChild(toastEl);
    
    // Initialize and show the toast
    const toast = new bootstrap.Toast(toastEl, {
        autohide: true,
        delay: 5000
    });
    toast.show();
    
    // Remove the toast element after it's hidden
    toastEl.addEventListener('hidden.bs.toast', function() {
        toastEl.remove();
    });
}

/**
 * Make an AJAX request
 * @param {string} url - The URL to request
 * @param {string} method - The HTTP method (GET, POST, etc.)
 * @param {object} data - The data to send (for POST requests)
 * @param {function} callback - The callback function to handle the response
 */
function makeRequest(url, method = 'GET', data = null, callback) {
    // Create the XHR object
    const xhr = new XMLHttpRequest();
    
    // Setup the callback
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                // Success
                let response;
                try {
                    response = JSON.parse(xhr.responseText);
                } catch (e) {
                    response = xhr.responseText;
                }
                callback(null, response);
            } else {
                // Error
                callback(new Error(`Request failed with status ${xhr.status}`), null);
            }
        }
    };
    
    // Open the request
    xhr.open(method, url, true);
    
    // Set headers
    xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
    
    // Send the request
    if (method === 'POST' && data) {
        xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
        xhr.send(JSON.stringify(data));
    } else {
        xhr.send();
    }
}

/**
 * Run a simulation with the specified parameters
 * @param {string} module - The module to run (bio, genetic, institutional, legal)
 * @param {object} params - The simulation parameters
 * @param {function} callback - The callback function to handle the response
 */
function runSimulation(module, params, callback) {
    showToast('Simulation', `Starting ${module} simulation...`, 'info');
    
    makeRequest(`/api/simulation/${module}`, 'POST', params, function(err, response) {
        if (err) {
            showToast('Simulation Error', `Failed to run ${module} simulation: ${err.message}`, 'error');
            callback(err, null);
            return;
        }
        
        showToast('Simulation Complete', `${module} simulation completed successfully`, 'success');
        callback(null, response);
    });
}

/**
 * Update the UI with simulation results
 * @param {string} module - The module that was run
 * @param {object} results - The simulation results
 */
function updateSimulationResults(module, results) {
    // This function would be implemented differently for each module
    console.log(`Updating UI with ${module} simulation results`, results);
    
    // Reload the page to show updated results
    // In a real implementation, this would update the UI dynamically
    window.location.reload();
}
