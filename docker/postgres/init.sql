-- Create database if it doesn't exist
CREATE DATABASE gaslit_af;

-- Connect to the database
\c gaslit_af;

-- Create schema
CREATE SCHEMA IF NOT EXISTS gaslit_af;

-- Create tables

-- Testimonies table
CREATE TABLE IF NOT EXISTS testimonies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    age_range VARCHAR(50),
    story TEXT NOT NULL,
    institutional_response TEXT,
    symptoms TEXT[],
    other_symptoms TEXT,
    onset_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    contact_consent BOOLEAN DEFAULT FALSE
);

-- Genomes table
CREATE TABLE IF NOT EXISTS genomes (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    data_source VARCHAR(50),
    notes TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    analysis_date TIMESTAMP WITH TIME ZONE,
    research_consent BOOLEAN DEFAULT FALSE
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    genome_id INTEGER REFERENCES genomes(id),
    module VARCHAR(50) NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Simulation runs table
CREATE TABLE IF NOT EXISTS simulation_runs (
    id SERIAL PRIMARY KEY,
    module VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    run_time FLOAT
);

-- Create indexes
CREATE INDEX IF NOT EXISTS testimonies_created_at_idx ON testimonies(created_at);
CREATE INDEX IF NOT EXISTS genomes_status_idx ON genomes(status);
CREATE INDEX IF NOT EXISTS analysis_results_genome_id_idx ON analysis_results(genome_id);
CREATE INDEX IF NOT EXISTS simulation_runs_module_idx ON simulation_runs(module);

-- Create a user for the application
CREATE USER gaslit_user WITH PASSWORD 'gaslit_password';
GRANT ALL PRIVILEGES ON DATABASE gaslit_af TO gaslit_user;
GRANT ALL PRIVILEGES ON SCHEMA gaslit_af TO gaslit_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gaslit_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO gaslit_user;
