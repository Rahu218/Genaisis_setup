from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    file_link = request.form['file_link']
    topic = request.form['topic']

    url = "http://127.0.0.1:8000/generate-questions"
    data = {'file_link': file_link, 'topic': topic}

    response = requests.post(url, data=data)

    if response.status_code == 200:
        questions = response.json()
        return jsonify(questions)
    else:
        return jsonify({'error': 'Failed to generate questions'}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
