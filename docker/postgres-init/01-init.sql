-- Runs once on first boot of an empty pg_data volume.
CREATE EXTENSION IF NOT EXISTS vector;

-- Langfuse (profile "obs") expects its own database on the same server.
SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
