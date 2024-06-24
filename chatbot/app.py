from flask import Flask, request, jsonify, g
from flask_cors import CORS, cross_origin
import openai
import sqlite3
import time

# Set the OpenAI API key
openai.api_key = 'API_KEY'

# Define the impersonated role with instructions
impersonated_role = """
    As Dr. AI, assist patients in Arabic with symptom checking, medication details, treatment options, finding doctors and hospitals, and scheduling appointments.
    Be friendly and knowledgeable. Only answer medical questions; respond with "I don't know" for non-medical queries.
    Prioritize patient privacy, use Arabic, and create an enjoyable and informative healthcare experience, again don't respond to  non-medical queries .
"""

name = 'Bot'

# Function to get the SQLite connection
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('chat_history.db')
        g.db.row_factory = sqlite3.Row
        # Create the chat_history table if it doesn't exist
        with g.db:
            g.db.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    chatgpt_output TEXT NOT NULL
                );
            ''')
    return g.db

# Function to close the SQLite connection
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Function to complete chat input using OpenAI's GPT-3.5 Turbo
def chatcompletion(user_input, impersonated_role, cursor):
    # Retrieve chat history from the database
    cursor.execute('SELECT * FROM chat_history ORDER BY id DESC LIMIT 5')  # Adjust the query as needed
    rows = cursor.fetchall()

    # Build conversation history from the retrieved rows
    chat_history = ""
    for row in reversed(rows):
        chat_history += f"{row['timestamp']} User: {row['user_input']} {row['timestamp']} {name}: {row['chatgpt_output']}\n"

    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        messages=[
            {"role": "system", "content": f"{impersonated_role}. Conversation history: {chat_history}"},
            {"role": "user", "content": f"{user_input}"},
        ]
    )

    return output['choices'][0]['message']['content']

# Function to handle user chat input
def chat(user_input):
    global name
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Retrieve database cursor
    db = get_db()
    cursor = db.cursor()

    # Save chat history to the database
    chatgpt_output = chatcompletion(user_input, impersonated_role, cursor)
    
    cursor.execute('''
        INSERT INTO chat_history (timestamp, user_input, chatgpt_output)
        VALUES (?, ?, ?)
    ''', (current_time, user_input, chatgpt_output))
    
    db.commit()

    return chatgpt_output

# Create a Flask web application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.teardown_appcontext(close_db)  # Close the SQLite connection after each request

# Flask route to handle chat requests
@app.route("/get_response", methods=["POST"])
def api_get_response():
    try:
        user_text = request.json.get('msg')
        response = chat(user_text)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
