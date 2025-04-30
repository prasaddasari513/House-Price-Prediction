pipeline {
    agent any

    environment {
        IMAGE_NAME = 'prasaddasari513/myimage:latest'
        CONTAINER_NAME = 'housepriceprediction'
        HOST_PORT = '9090'
        CONTAINER_PORT = '90'
    }

    stages {
        stage('Build Docker Image') {
            steps {
                echo "🔧 Building Docker image..."
                sh "docker build -t ${IMAGE_NAME} ."
            }
        }

        stage('Tag Docker Image') {
            steps {
                echo "Tagging Docker image..."
                sh 'docker tag myimage ${IMAGE_NAME}'
            }
        }

        stage('Docker Login') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'docker-hub-credentials',
                    usernameVariable: 'DOCKER_USERNAME',
                    passwordVariable: 'DOCKER_PASSWORD'
                )]) {
                    echo "Logging into Docker Hub..."
                    sh 'echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin'
                }
            }
        }

        stage('Push Image to Docker Hub') {
            steps {
                echo "Pushing image to Docker Hub..."
                sh 'docker push ${IMAGE_NAME}'
            }
        }

        stage('Remove Old Container') {
            steps {
                echo "Removing old container..."
                sh "docker rm -f ${CONTAINER_NAME} || true"
            }
        }

        stage('Run New Container') {
            steps {
                echo "Running new container..."
                sh """
                    docker run -d --name ${CONTAINER_NAME} \
                    -p ${HOST_PORT}:${CONTAINER_PORT} \
                    ${IMAGE_NAME}
                """
            }
        }
    }

    post {
        success {
            echo "✅ Deployed! Visit: http://localhost:${HOST_PORT}"
        }
        failure {
            echo "❌ Build or deployment failed."
        }
    }
}
