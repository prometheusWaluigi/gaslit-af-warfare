#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend Advocacy Layer for GASLIT-AF WARSTACK

This module implements a Flask-based web application for the frontend advocacy layer,
providing dynamic dashboards, user-uploadable genome analysis, and a "Tell Your Story" portal.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'gaslit-af-development-key')

# Import modules from other layers (with error handling)
try:
    from src.biological_modeling.neuroimmune_simulator import NeuroimmuneDynamics
    HAS_BIO_MODULE = True
except ImportError:
    logger.warning("Biological modeling module not available")
    HAS_BIO_MODULE = False

try:
    from src.genetic_risk.genetic_scanner import GeneticRiskScanner
    HAS_GENETIC_MODULE = True
except ImportError:
    logger.warning("Genetic risk module not available")
    HAS_GENETIC_MODULE = False

try:
    from src.institutional_feedback.institutional_model import InstitutionalFeedbackModel
    HAS_INSTITUTIONAL_MODULE = True
except ImportError:
    logger.warning("Institutional feedback module not available")
    HAS_INSTITUTIONAL_MODULE = False

try:
    from src.legal_policy.legal_simulator import LegalPolicySimulator
    HAS_LEGAL_MODULE = True
except ImportError:
    logger.warning("Legal policy module not available")
    HAS_LEGAL_MODULE = False


# Data storage (in-memory for development, would use a database in production)
testimonies = []
uploaded_genomes = []
simulation_results = {}


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', 
                          modules={
                              'bio': HAS_BIO_MODULE,
                              'genetic': HAS_GENETIC_MODULE,
                              'institutional': HAS_INSTITUTIONAL_MODULE,
                              'legal': HAS_LEGAL_MODULE
                          })


@app.route('/dashboard')
def dashboard():
    """Render the main dashboard."""
    # Load sample data if no simulation results exist
    if not simulation_results:
        _load_sample_data()
    
    return render_template('dashboard.html', 
                          results=simulation_results,
                          testimonies_count=len(testimonies),
                          genomes_count=len(uploaded_genomes))


@app.route('/biological')
def biological_dashboard():
    """Render the biological modeling dashboard."""
    if not HAS_BIO_MODULE:
        flash("Biological modeling module not available", "warning")
        return redirect(url_for('dashboard'))
    
    # Get simulation results or run a sample simulation
    bio_results = simulation_results.get('biological', {})
    if not bio_results and HAS_BIO_MODULE:
        try:
            simulator = NeuroimmuneDynamics()
            bio_results = simulator.run_simulation(time_steps=100)
            simulation_results['biological'] = bio_results
        except Exception as e:
            logger.error(f"Error running biological simulation: {e}")
            flash(f"Error running simulation: {str(e)}", "danger")
    
    return render_template('biological.html', results=bio_results)


@app.route('/genetic')
def genetic_dashboard():
    """Render the genetic risk dashboard."""
    if not HAS_GENETIC_MODULE:
        flash("Genetic risk module not available", "warning")
        return redirect(url_for('dashboard'))
    
    # Get analysis results or run a sample analysis
    genetic_results = simulation_results.get('genetic', {})
    
    return render_template('genetic.html', 
                          results=genetic_results,
                          uploaded_genomes=uploaded_genomes)


@app.route('/upload_genome', methods=['GET', 'POST'])
def upload_genome():
    """Handle genome file uploads."""
    if request.method == 'POST':
        if 'genome_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['genome_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file:
            # In a real implementation, save the file and process it
            # For now, just store metadata
            genome_info = {
                'filename': file.filename,
                'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'size': len(file.read()),
                'status': 'pending',
                'id': len(uploaded_genomes) + 1
            }
            
            uploaded_genomes.append(genome_info)
            flash('Genome file uploaded successfully', 'success')
            return redirect(url_for('genetic_dashboard'))
    
    return render_template('upload_genome.html')


@app.route('/institutional')
def institutional_dashboard():
    """Render the institutional feedback dashboard."""
    if not HAS_INSTITUTIONAL_MODULE:
        flash("Institutional feedback module not available", "warning")
        return redirect(url_for('dashboard'))
    
    # Get simulation results or run a sample simulation
    inst_results = simulation_results.get('institutional', {})
    if not inst_results and HAS_INSTITUTIONAL_MODULE:
        try:
            model = InstitutionalFeedbackModel()
            inst_results = model.run_simulation(time_steps=50)
            simulation_results['institutional'] = inst_results
        except Exception as e:
            logger.error(f"Error running institutional simulation: {e}")
            flash(f"Error running simulation: {str(e)}", "danger")
    
    return render_template('institutional.html', results=inst_results)


@app.route('/legal')
def legal_dashboard():
    """Render the legal and policy dashboard."""
    if not HAS_LEGAL_MODULE:
        flash("Legal policy module not available", "warning")
        return redirect(url_for('dashboard'))
    
    # Get simulation results or run a sample simulation
    legal_results = simulation_results.get('legal', {})
    if not legal_results and HAS_LEGAL_MODULE:
        try:
            simulator = LegalPolicySimulator()
            legal_results = simulator.run_simulation()
            simulation_results['legal'] = legal_results
        except Exception as e:
            logger.error(f"Error running legal simulation: {e}")
            flash(f"Error running simulation: {str(e)}", "danger")
    
    return render_template('legal.html', results=legal_results)


@app.route('/tell-your-story', methods=['GET', 'POST'])
def tell_your_story():
    """Handle user testimonies."""
    if request.method == 'POST':
        name = request.form.get('name', 'Anonymous')
        email = request.form.get('email', '')
        story = request.form.get('story', '')
        symptoms = request.form.getlist('symptoms')
        
        if not story:
            flash('Please provide your story', 'danger')
            return redirect(request.url)
        
        # In a real implementation, validate and store in a database
        testimony = {
            'name': name,
            'email': email,
            'story': story,
            'symptoms': symptoms,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'id': len(testimonies) + 1
        }
        
        testimonies.append(testimony)
        flash('Your story has been submitted successfully', 'success')
        return redirect(url_for('testimonies'))
    
    return render_template('tell_your_story.html')


@app.route('/testimonies')
def testimonies_page():
    """Display user testimonies."""
    return render_template('testimonies.html', testimonies=testimonies)


@app.route('/api/simulation/<module>', methods=['POST'])
def run_simulation_api(module):
    """API endpoint to run simulations."""
    if module == 'biological' and HAS_BIO_MODULE:
        try:
            params = request.json or {}
            simulator = NeuroimmuneDynamics({'params': params})
            results = simulator.run_simulation()
            simulation_results['biological'] = results
            return jsonify({'success': True, 'results': results})
        except Exception as e:
            logger.error(f"Error running biological simulation: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif module == 'genetic' and HAS_GENETIC_MODULE:
        try:
            params = request.json or {}
            scanner = GeneticRiskScanner({'params': params})
            # In a real implementation, this would process an actual VCF file
            # For now, use the sample analysis
            results = scanner.run_sample_analysis()
            simulation_results['genetic'] = results
            return jsonify({'success': True, 'results': results})
        except Exception as e:
            logger.error(f"Error running genetic analysis: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif module == 'institutional' and HAS_INSTITUTIONAL_MODULE:
        try:
            params = request.json or {}
            model = InstitutionalFeedbackModel({'params': params})
            results = model.run_simulation()
            simulation_results['institutional'] = results
            return jsonify({'success': True, 'results': results})
        except Exception as e:
            logger.error(f"Error running institutional simulation: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif module == 'legal' and HAS_LEGAL_MODULE:
        try:
            params = request.json or {}
            simulator = LegalPolicySimulator({'params': params})
            results = simulator.run_simulation()
            simulation_results['legal'] = results
            return jsonify({'success': True, 'results': results})
        except Exception as e:
            logger.error(f"Error running legal simulation: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    else:
        return jsonify({'success': False, 'error': f"Module '{module}' not available"}), 404


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)


def _load_sample_data():
    """Load sample data for development."""
    # Sample testimonies
    testimonies.extend([
        {
            'name': 'John Doe',
            'email': 'john@example.com',
            'story': 'I experienced severe neurological symptoms after exposure...',
            'symptoms': ['fatigue', 'brain fog', 'tremors'],
            'date': '2023-05-15 10:30:45',
            'id': 1
        },
        {
            'name': 'Jane Smith',
            'email': 'jane@example.com',
            'story': 'My autonomic nervous system has been severely affected...',
            'symptoms': ['pots', 'tachycardia', 'fatigue'],
            'date': '2023-06-22 14:15:30',
            'id': 2
        }
    ])
    
    # Sample uploaded genomes
    uploaded_genomes.extend([
        {
            'filename': 'sample_genome_1.vcf',
            'upload_date': '2023-04-10 09:45:22',
            'size': 15000000,
            'status': 'analyzed',
            'id': 1
        },
        {
            'filename': 'sample_genome_2.vcf',
            'upload_date': '2023-07-05 16:20:18',
            'size': 18000000,
            'status': 'analyzed',
            'id': 2
        }
    ])
    
    # Try to load simulation results from each module
    if HAS_BIO_MODULE:
        try:
            simulator = NeuroimmuneDynamics()
            simulation_results['biological'] = simulator.run_simulation(time_steps=100)
        except Exception as e:
            logger.error(f"Error loading sample biological data: {e}")
    
    if HAS_GENETIC_MODULE:
        try:
            scanner = GeneticRiskScanner()
            simulation_results['genetic'] = {'sample': True}  # Placeholder
        except Exception as e:
            logger.error(f"Error loading sample genetic data: {e}")
    
    if HAS_INSTITUTIONAL_MODULE:
        try:
            model = InstitutionalFeedbackModel()
            simulation_results['institutional'] = model.run_simulation(time_steps=50)
        except Exception as e:
            logger.error(f"Error loading sample institutional data: {e}")
    
    if HAS_LEGAL_MODULE:
        try:
            simulator = LegalPolicySimulator()
            simulation_results['legal'] = simulator.run_simulation()
        except Exception as e:
            logger.error(f"Error loading sample legal data: {e}")


def create_app():
    """Create and configure the Flask application."""
    # Additional configuration could be done here
    return app


if __name__ == '__main__':
    # Create the app
    app = create_app()
    
    # Load sample data for development
    _load_sample_data()
    
    # Run the app
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
