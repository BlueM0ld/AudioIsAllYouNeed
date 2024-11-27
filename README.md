Run this project

Add data folder and files in app/fastapi

Create docker images - need to navigate to the folders

- fastapi
    `DOCKER_BUILDKIT=0 docker buildx build . -t fastapi `

- streamlit 
    `DOCKER_BUILDKIT=0 docker buildx build . -t streamlit `
- postgres
    `DOCKER_BUILDKIT=0 docker buildx build . -t postgres `


Then from root execute docker compose
    `docker-compose up `

