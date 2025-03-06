#!/bin/bash

# Function to stop and remove existing containers
cleanup() {
    echo "Stopping and removing existing containers..."
    docker-compose down
}

# Function to build and start containers
start() {
    echo "Building and starting containers..."
    docker-compose up --build -d
}

# Function to show logs
logs() {
    echo "Showing logs..."
    docker-compose logs -f
}

# Function to show status
status() {
    echo "Container status:"
    docker-compose ps
}

# Main script
case "$1" in
    "cleanup")
        cleanup
        ;;
    "start")
        start
        ;;
    "logs")
        logs
        ;;
    "status")
        status
        ;;
    "restart")
        cleanup
        start
        ;;
    *)
        echo "Usage: $0 {cleanup|start|logs|status|restart}"
        exit 1
        ;;
esac

exit 0 