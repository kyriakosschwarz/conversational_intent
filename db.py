import psycopg2
import datetime
import tomllib

with open(".streamlit/secrets.toml", "rb") as f:
    db_creds = tomllib.load(f)["db_credentials"]

# Database connection details
DATABASE_URL = f"postgresql://{db_creds["user"]}:{db_creds["pass"]}@{db_creds["host"]}:{db_creds["port"]}/{db_creds["database"]}"  # Replace with your database details

# Function to connect to the database
def connect_to_db():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("Connected to DB")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Function to create the user_inputs table if it doesn't exist
def create_table(conn):
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



conn = connect_to_db()
if conn:
    create_table(conn)  # Create table if it doesn't exist
    conn.close()
