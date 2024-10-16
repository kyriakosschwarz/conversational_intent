"""
Module: db.py

Description: This module handles database operations for the intent classification system, including connection establishment, table creation, and data retrieval/storage.
"""

import psycopg2
import datetime
#import tomllib
import os

#with open(".streamlit/secrets.toml", "rb") as f:
#    db_creds = tomllib.load(f)["db_credentials"]

# Access environment variables
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT')
db_name = os.environ.get('DB_NAME')
db_user_file = os.environ.get('DB_USER_FILE')
db_password_file = os.environ.get('DB_PASSWORD_FILE')

# Function to read secrets from files
def read_secret(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Read username and password from secret files
db_user = read_secret(db_user_file)
db_password = read_secret(db_password_file)

# Database connection details
#DATABASE_URL = f"postgresql://{db_creds["user"]}:{db_creds["pass"]}@{db_creds["host"]}:{db_creds["port"]}/{db_creds["database"]}"  # Replace with your database details
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"  # Replace with your database details


# Function to connect to the database
def connect_to_db():
    """
    Establishes a connection to the PostgreSQL database.

    Returns:
        psycopg2.connection: A connection object if successful, None otherwise.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("Connected to DB")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Function to create the user_inputs table if it doesn't exist
def create_table(conn):
    """
    Creates the 'user_inputs' table in the database if it doesn't already exist.

    Args:
        conn (psycopg2.connection): The database connection object.
    """
    try:
        table_name = "user_inputs"
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                input_text TEXT,
                output_text TEXT,
                intent TEXT,
                probability FLOAT,
                timestamp TIMESTAMP WITHOUT TIME ZONE
            )
        """)
        conn.commit()
        cursor.close()
        print(f"Table {table_name} successfully created.")
    except Exception as e:
        print(f"Error creating table: {e}")


# Function to store user input and output in the database
def store_input_and_output(conn, user_input, output_text, intent, probability):
    """
    Stores the user input, model output, predicted intent, and probability in the database.

    Args:
        conn (psycopg2.connection): The database connection object.
        user_input (str): The original input text from the user.
        output_text (str): The output text produced by the model.
        intent (str): The predicted intent.
        probability (float): The probability of the predicted intent.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_inputs (input_text, output_text, intent, probability, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_input, output_text, intent, probability, datetime.datetime.now()))
        conn.commit()
        cursor.close()
        print(f"Successfully inserted {user_input}, {output_text}")
    except Exception as e:
        print(f"Error storing input in database: {e}")
        return None

# Function to retrieve the last five entries based on timestamp
def get_last_entries(conn, nr_entries):
    """
    Retrieves the last 'nr_entries' entries from the database, ordered by timestamp.

    Args:
        conn (psycopg2.connection): The database connection object.
        nr_entries (int): The number of entries to retrieve.

    Returns:
        dict: A dictionary containing the retrieved data and column names.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT input_text, output_text, intent, probability, timestamp
            FROM user_inputs
            ORDER BY timestamp DESC
            LIMIT {nr_entries}
        """)
        results = cursor.fetchall()
        # Get the column names
        column_names = [desc[0] for desc in cursor.description]
        cursor.close()
        return {'data': results, 'columns': column_names}
    except Exception as e:
        print(f"Error retrieving entries from database: {e}")
        return None
