# Docker Configuration for GASLIT-AF WARSTACK

This directory contains Docker-related configuration files for the GASLIT-AF WARSTACK project.

## Directory Structure

- `nginx/`: NGINX web server configuration
  - `nginx.conf`: NGINX configuration file for serving the application
- `postgres/`: PostgreSQL database configuration
  - `init.sql`: Database initialization script

## Usage

The Docker setup is configured using Docker Compose. To start the application with all its services, run:

```bash
docker-compose up --build
```

This will start the following services:

1. **app**: The main Flask application
2. **db**: PostgreSQL database
3. **nginx**: NGINX web server for serving the application
4. **jupyter**: Jupyter Notebook for data analysis (optional)

## Configuration

### NGINX

The NGINX configuration (`nginx/nginx.conf`) sets up a reverse proxy to the Flask application and handles static file serving. It also includes:

- Gzip compression for better performance
- Cache control for static assets
- Health check endpoint
- Increased upload size limit for genome files

### PostgreSQL

The PostgreSQL initialization script (`postgres/init.sql`) sets up the database schema for the application. It creates:

- Tables for testimonies, genome uploads, analysis results, and users
- Indexes for better query performance
- A default admin user
- Triggers for automatic timestamp updates

## Environment Variables

The following environment variables can be configured in the `docker-compose.yml` file:

- `FLASK_ENV`: Flask environment (development/production)
- `SECRET_KEY`: Secret key for Flask sessions
- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DB`: PostgreSQL database name

## Volumes

The Docker Compose setup uses the following volumes:

- `postgres_data`: Persistent storage for PostgreSQL data
- `./results:/app/results`: Shared volume for simulation results
- `./logs:/app/logs`: Shared volume for application logs
- `./uploads:/app/uploads`: Shared volume for user uploads
- `./data:/app/data`: Shared volume for application data
- `./static:/app/static`: Shared volume for static files

## Networks

All services are connected to the `gaslit-network` bridge network for internal communication.

## Customization

To customize the Docker setup:

1. Modify the `Dockerfile` to change the application container
2. Update `docker-compose.yml` to add or remove services
3. Adjust the NGINX configuration in `nginx/nginx.conf`
4. Modify the database schema in `postgres/init.sql`
