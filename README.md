# Conversational Assistant

This repository contains a Conversational Assistant as a Dockerized Streamlit app with a PostgreSQL database. It helps you predict user intent based on conversational utterances and track the history of those predictions.

## Prerequisites

- Docker
- Docker Compose

## Running the App

1. Clone this repository:
   ```
   git clone https://github.com/kyriakosschwarz/conversational_intent.git
   cd conversational_intent
   ```

2. Create the `secrets` directory and add your database credentials:
   ```
   mkdir secrets
   echo "your_db_username" > secrets/db_user.txt
   echo "your_db_password" > secrets/db_password.txt
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Open your browser and navigate to `http://localhost:8501` to view the Streamlit app.

## Stopping the App

To stop the app, press `Ctrl+C` in the terminal where you ran `docker-compose up`, or run:
```
docker-compose down
```

This will stop and remove the containers, but preserve the database volume.
