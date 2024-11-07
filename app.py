import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from main.RAGSystem import RAGSystem  # Assuming you import your system from the previous code
from getpass import getpass

app = Flask(__name__)

# Configure file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Setup the OpenAI API Key
OPENAI_API_KEY = getpass("Enter your OpenAI API key: ")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB for file uploads

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route - render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# PDF Upload Route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": f"File '{filename}' uploaded successfully!"}), 200
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

# Query Route
@app.route('/query', methods=['POST'])
def query():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Assuming you've uploaded the PDFs and processed them in RAGSystem
    pdf_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    rag_system = RAGSystem(pdf_paths, OPENAI_API_KEY)

    response = rag_system.query(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
