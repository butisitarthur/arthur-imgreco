-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create application user
CREATE USER arthur_app WITH PASSWORD 'arthur_app123';

-- Grant permissions
GRANT CONNECT ON DATABASE arthur_imgreco TO arthur_app;
GRANT USAGE ON SCHEMA public TO arthur_app;
GRANT CREATE ON SCHEMA public TO arthur_app;