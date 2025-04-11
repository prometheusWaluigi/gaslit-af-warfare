-- GASLIT-AF WARSTACK Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create tables

-- Testimonies table
CREATE TABLE IF NOT EXISTS testimonies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    email VARCHAR(255),
    age_range VARCHAR(50),
    story TEXT NOT NULL,
    institutional_response TEXT,
    symptoms TEXT[],
    other_symptoms TEXT,
    onset_date DATE,
    contact_consent BOOLEAN DEFAULT FALSE,
    submission_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address VARCHAR(45),
    approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Genome uploads table
CREATE TABLE IF NOT EXISTS genome_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_path VARCHAR(255) NOT NULL,
    file_size BIGINT,
    file_type VARCHAR(50),
    data_source VARCHAR(255),
    notes TEXT,
    research_consent BOOLEAN DEFAULT FALSE,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module VARCHAR(50) NOT NULL,
    result_type VARCHAR(50) NOT NULL,
    result_data JSONB NOT NULL,
    file_path VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table (for admin access)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_testimonies_submission_date ON testimonies(submission_date);
CREATE INDEX IF NOT EXISTS idx_testimonies_approved ON testimonies(approved);
CREATE INDEX IF NOT EXISTS idx_genome_uploads_upload_date ON genome_uploads(upload_date);
CREATE INDEX IF NOT EXISTS idx_analysis_results_module ON analysis_results(module);
CREATE INDEX IF NOT EXISTS idx_analysis_results_result_type ON analysis_results(result_type);

-- Create a search index on testimony stories
CREATE INDEX IF NOT EXISTS idx_testimonies_story_trgm ON testimonies USING GIN (story gin_trgm_ops);

-- Create admin user (password: gaslit-admin)
INSERT INTO users (username, password_hash, email, is_admin)
VALUES (
    'admin',
    '$2b$12$1NiVBQV.SYR1SkH3hXQVVOzMG8xmdQtX0aQGJTJ1Uu1FX8hspnm4W',
    'admin@gaslit-af-warstack.org',
    TRUE
)
ON CONFLICT (username) DO NOTHING;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at timestamps
CREATE TRIGGER update_testimonies_timestamp
BEFORE UPDATE ON testimonies
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_genome_uploads_timestamp
BEFORE UPDATE ON genome_uploads
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_analysis_results_timestamp
BEFORE UPDATE ON analysis_results
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_users_timestamp
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_timestamp();
