# Variables
WEB_APP_NAME := streamlit_app

# Default target
.PHONY: all
all: build run

# Build the Docker image and services
.PHONY: build
build:
	docker build -t $(WEB_APP_NAME) .

# Run the services using docker-compose
.PHONY: run
run:
#	docker run --name $(WEB_APP_NAME) -d -p 5000:5000 -e DB_CONNECTION_STRING=$(DB_CONNECTION_STRING) $(WEB_APP_NAME)
	docker run --name $(WEB_APP_NAME) --env-file .env -d -p 5000:5000 $(WEB_APP_NAME)

# Stop all running containers
.PHONY: stop
stop:
	docker stop $(WEB_APP_NAME)

# Remove all Docker artifacts (containers, images, volumes, networks)
.PHONY: clean
clean:
	-docker stop $(WEB_APP_NAME)
	-docker rm $(WEB_APP_NAME)
	-docker rmi $(WEB_APP_NAME)

# Rebuild and run the Docker services
.PHONY: rebuild
rebuild: clean build run

# Tail the logs for the container
.PHONY: logs
logs:
	docker logs -f $(WEB_APP_NAME)

.PHONY: tag
tag:
	docker tag $(WEB_APP_NAME) $(DOCKER_USERNAME)/$(WEB_APP_NAME)
