#!/usr/bin/env python3
"""
Frontend Application for GASLIT-AF WARSTACK

This module provides a web interface for the GASLIT-AF WARSTACK project,
allowing users to run simulations, visualize results, and submit testimonies.
"""

import os
import sys
import json
import time
import datetime
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from werkzeug.utils import secure_filename

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules
try:
    from src.biological_modeling.neuroimmune_simulator import NeuroimmuneDynamics, run_sample_simulation as run_bio_simulation
except ImportError:
    print("Warning: Biological modeling module not found or incomplete.")
    run_bio_simulation = None

try:
    from src.genetic_risk.genetic_scanner import GeneticRiskScanner, run_sample_analysis as run_genetic_analysis
except ImportError:
    print("Warning: Genetic risk scanning module not found or incomplete.")
    run_genetic_analysis = None

try:
    from src.institutional_feedback.institutional_model import InstitutionalFeedbackModel, run_sample_simulation as run_institutional_simulation
except ImportError:
    print("Warning: Institutional feedback module not found or incomplete.")
    run_institutional_simulation = None

try:
    from src.legal_policy.legal_simulator import LegalPolicySimulator, run_sample_simulation as run_legal_simulation
except ImportError:
    print("Warning: Legal policy module not found or incomplete.")
    run_legal_simulation = None

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'gaslit-af-warstack-dev-key')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'frontend.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('frontend')

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'vcf', 'fastq', 'fq', 'json'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_plot_base64(fig):
    """Convert a matplotlib figure to a base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')


@app.route('/biological', methods=['GET', 'POST'])
def biological():
    """Render the biological modeling page."""
    if request.method == 'POST':
        # Get form data
        grid_size = int(request.form.get('grid_size', 100))
        time_steps = int(request.form.get('time_steps', 500))
        initial_condition = request.form.get('initial_condition', 'random')
        
        # Create configuration
        config = {
            'grid_size': grid_size,
            'time_steps': time_steps,
            'initial_condition': initial_condition,
            'output_dir': 'results/biological_modeling'
        }
        
        # Run simulation
        try:
            simulator = NeuroimmuneDynamics(config)
            simulator.initialize_grid()
            simulator.run_simulation()
            
            # Generate visualizations
            simulator.visualize_grid(save_path='static/img/biological/final_state.png')
            
            param1_range = np.linspace(0.1, 1.0, 5)
            param2_range = np.linspace(0.1, 1.0, 5)
            simulator.generate_phase_portrait(param1_range, param2_range)
            simulator.visualize_phase_portrait(save_path='static/img/biological/phase_portrait.png')
            
            # Save results
            results_file = simulator.save_results()
            
            flash('Simulation completed successfully!', 'success')
            return render_template('biological.html', 
                                  final_state_img='img/biological/final_state.png',
                                  phase_portrait_img='img/biological/phase_portrait.png',
                                  results_file=os.path.basename(results_file),
                                  config=config)
        
        except Exception as e:
            logger.error(f"Error running biological simulation: {e}")
            flash(f'Error running simulation: {str(e)}', 'error')
    
    return render_template('biological.html')


@app.route('/genetic', methods=['GET', 'POST'])
def genetic():
    """Render the genetic risk scanning page."""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Create configuration
            config = {
                'output_dir': 'results/genetic_risk'
            }
            
            # Run analysis
            try:
                scanner = GeneticRiskScanner(config)
                results = scanner.analyze_file(filepath)
                
                # Generate visualizations
                scanner.generate_heatmap(save_path='static/img/genetic/variant_heatmap.png')
                
                # Generate risk profile
                risk_profile = scanner.generate_risk_profile(save_path='results/genetic_risk/risk_profile.json')
                
                # Save results
                results_file = scanner.save_results()
                
                flash('Genetic analysis completed successfully!', 'success')
                return render_template('genetic.html', 
                                      heatmap_img='img/genetic/variant_heatmap.png',
                                      results=results,
                                      risk_profile=risk_profile,
                                      results_file=os.path.basename(results_file))
            
            except Exception as e:
                logger.error(f"Error running genetic analysis: {e}")
                flash(f'Error running analysis: {str(e)}', 'error')
        else:
            flash('File type not allowed', 'error')
    
    return render_template('genetic.html')


@app.route('/institutional', methods=['GET', 'POST'])
def institutional():
    """Render the institutional feedback modeling page."""
    if request.method == 'POST':
        # Get form data
        network_size = int(request.form.get('network_size', 10))
        simulation_steps = int(request.form.get('simulation_steps', 50))
        initial_entropy = float(request.form.get('initial_entropy', 0.1))
        
        # Create configuration
        config = {
            'network_size': network_size,
            'simulation_steps': simulation_steps,
            'initial_entropy': initial_entropy,
            'output_dir': 'results/institutional_feedback'
        }
        
        # Run simulation
        try:
            model = InstitutionalFeedbackModel(config)
            model.run_simulation()
            
            # Generate visualizations
            model.visualize_network(save_path='static/img/institutional/network.png')
            model.visualize_simulation_results(save_path='static/img/institutional/simulation_results.png')
            
            # Save results
            results_file = model.save_results()
            
            flash('Simulation completed successfully!', 'success')
            return render_template('institutional.html', 
                                  network_img='img/institutional/network.png',
                                  results_img='img/institutional/simulation_results.png',
                                  results_file=os.path.basename(results_file),
                                  config=config)
        
        except Exception as e:
            logger.error(f"Error running institutional simulation: {e}")
            flash(f'Error running simulation: {str(e)}', 'error')
    
    return render_template('institutional.html')


@app.route('/legal', methods=['GET', 'POST'])
def legal():
    """Render the legal policy simulation page."""
    if request.method == 'POST':
        # Get form data
        simulation_steps = int(request.form.get('simulation_steps', 50))
        initial_evidence = float(request.form.get('initial_evidence', 0.1))
        shield_decay_rate = float(request.form.get('shield_decay_rate', 0.01))
        
        # Create configuration
        config = {
            'simulation_steps': simulation_steps,
            'initial_evidence_level': initial_evidence,
            'shield_decay_rate': shield_decay_rate,
            'output_dir': 'results/legal_policy'
        }
        
        # Run simulation
        try:
            simulator = LegalPolicySimulator(config)
            simulator.run_simulation()
            
            # Generate visualizations
            simulator.visualize_simulation_results(save_path='static/img/legal/simulation_results.png')
            
            # Save results
            results_file = simulator.save_results()
            
            flash('Simulation completed successfully!', 'success')
            return render_template('legal.html', 
                                  results_img='img/legal/simulation_results.png',
                                  results_file=os.path.basename(results_file),
                                  config=config)
        
        except Exception as e:
            logger.error(f"Error running legal simulation: {e}")
            flash(f'Error running simulation: {str(e)}', 'error')
    
    return render_template('legal.html')


@app.route('/tell-your-story', methods=['GET', 'POST'])
def tell_your_story():
    """Render the testimony submission page."""
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        age_range = request.form.get('age_range', '')
        story = request.form.get('story', '')
        institutional_response = request.form.get('institutional_response', '')
        symptoms = request.form.getlist('symptoms')
        other_symptoms = request.form.get('other_symptoms', '')
        onset_date = request.form.get('onset_date', '')
        contact_consent = 'contact_consent' in request.form
        
        # Create testimony data
        testimony = {
            'name': name,
            'email': email,
            'age_range': age_range,
            'story': story,
            'institutional_response': institutional_response,
            'symptoms': symptoms,
            'other_symptoms': other_symptoms,
            'onset_date': onset_date,
            'contact_consent': contact_consent,
            'submission_date': datetime.datetime.now().isoformat()
        }
        
        # Save testimony
        try:
            os.makedirs('data/testimonies', exist_ok=True)
            filename = f"testimony_{int(time.time())}.json"
            filepath = os.path.join('data/testimonies', filename)
            
            with open(filepath, 'w') as f:
                json.dump(testimony, f, indent=2)
            
            flash('Thank you for sharing your story!', 'success')
            return redirect(url_for('testimonies'))
        
        except Exception as e:
            logger.error(f"Error saving testimony: {e}")
            flash(f'Error saving testimony: {str(e)}', 'error')
    
    return render_template('tell_your_story.html')


@app.route('/testimonies')
def testimonies():
    """Render the testimonies page."""
    # Load testimonies
    testimony_list = []
    testimonies_dir = 'data/testimonies'
    
    if os.path.exists(testimonies_dir):
        for filename in os.listdir(testimonies_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(testimonies_dir, filename), 'r') as f:
                        testimony = json.load(f)
                        testimony_list.append(testimony)
                except Exception as e:
                    logger.error(f"Error loading testimony {filename}: {e}")
    
    # Sort by submission date (newest first)
    testimony_list.sort(key=lambda x: x.get('submission_date', ''), reverse=True)
    
    return render_template('testimonies.html', testimonies=testimony_list)


@app.route('/upload-genome', methods=['GET', 'POST'])
def upload_genome():
    """Render the genome upload page."""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get form data
            data_source = request.form.get('data_source', '')
            notes = request.form.get('notes', '')
            research_consent = 'research_consent' in request.form
            
            # Save metadata
            try:
                os.makedirs('data/genomes', exist_ok=True)
                metadata = {
                    'filename': filename,
                    'original_filename': file.filename,
                    'file_path': filepath,
                    'file_size': os.path.getsize(filepath),
                    'data_source': data_source,
                    'notes': notes,
                    'upload_date': datetime.datetime.now().isoformat(),
                    'research_consent': research_consent
                }
                
                metadata_file = os.path.join('data/genomes', f"{os.path.splitext(filename)[0]}_metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                flash('Genome uploaded successfully!', 'success')
                return redirect(url_for('genetic'))
            
            except Exception as e:
                logger.error(f"Error saving genome metadata: {e}")
                flash(f'Error saving genome metadata: {str(e)}', 'error')
        else:
            flash('File type not allowed', 'error')
    
    return render_template('upload_genome.html')


@app.route('/run-sample/<module>')
def run_sample(module):
    """Run a sample simulation for the specified module."""
    try:
        if module == 'biological' and run_bio_simulation:
            simulator = run_bio_simulation()
            flash('Sample biological simulation completed successfully!', 'success')
            return redirect(url_for('biological'))
        
        elif module == 'genetic' and run_genetic_analysis:
            scanner = run_genetic_analysis()
            flash('Sample genetic analysis completed successfully!', 'success')
            return redirect(url_for('genetic'))
        
        elif module == 'institutional' and run_institutional_simulation:
            model = run_institutional_simulation()
            flash('Sample institutional simulation completed successfully!', 'success')
            return redirect(url_for('institutional'))
        
        elif module == 'legal' and run_legal_simulation:
            simulator = run_legal_simulation()
            flash('Sample legal simulation completed successfully!', 'success')
            return redirect(url_for('legal'))
        
        else:
            flash(f'Sample simulation for {module} module not available', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Error running sample {module} simulation: {e}")
        flash(f'Error running sample simulation: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/results/<path:filename>')
def download_result(filename):
    """Download a result file."""
    return send_from_directory('results', filename)


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create static directories if they don't exist
    os.makedirs('static/img/biological', exist_ok=True)
    os.makedirs('static/img/genetic', exist_ok=True)
    os.makedirs('static/img/institutional', exist_ok=True)
    os.makedirs('static/img/legal', exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
