pipeline {
    agent any
    environment {
        PYTHON = 'C:\\Users\\Dell\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe'
    }
    stages {
        stage('Install Dependencies') {
            steps {
                bat '"%PYTHON%" --version'
                bat '"%PYTHON%" -m pip install -r requirements.txt'
            }
        }
        stage('Run Tests') {
            steps {
                bat '"%PYTHON%" -m pytest -q'
            }
        }
        stage('Build') {
            steps {
                bat '"%PYTHON%" -m py_compile app.py'
            }
        }
        stage('Deploy') {
            steps {
                bat 'if not exist deploy mkdir deploy'
                bat 'copy /Y app.py deploy\\app.py'
                bat 'copy /Y requirements.txt deploy\\requirements.txt'
                echo 'Deployment Successful'
            }
        }
    }
} 

