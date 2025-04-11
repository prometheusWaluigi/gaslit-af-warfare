# Docker Configuration for GASLIT-AF WARSTACK

This directory contains Docker-related configuration files for the GASLIT-AF WARSTACK project.

## Directory Structure

- **nginx/**: Nginx web server configuration
  - `nginx.conf`: Main Nginx configuration file
  
- **postgres/**: PostgreSQL database configuration
  - `init.sql`: Database initialization script

## Usage

The Docker setup is configured using Docker Compose. The main `docker-compose.yml` file is located in the project root directory.

### Running with Docker Compose

To start the application with Docker Compose:

```bash
docker-compose up -d
```

This will start the following services:

- **app**: The main application container running the Flask application
- **db**: (Optional) PostgreSQL database container
- **nginx**: (Optional) Nginx web server container for production deployments

### Development vs. Production

For development, you can use just the app service:

```bash
docker-compose up app
```

For production, you should use all services and enable the Nginx configuration:

1. Uncomment the database service in `docker-compose.yml`
2. Add the Nginx service to `docker-compose.yml`:

```yaml
nginx:
  image: nginx:alpine
  container_name: gaslit-af-nginx
  ports:
    - "80:80"
  volumes:
    - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    - ./src/frontend/static:/app/src/frontend/static
  depends_on:
    - app
  restart: unless-stopped
```

## Customization

### Database Configuration

You can customize the database configuration by modifying:

- The `init.sql` script in the `postgres` directory
- The environment variables in the `docker-compose.yml` file

### Nginx Configuration

You can customize the Nginx configuration by modifying the `nginx.conf` file in the `nginx` directory.
