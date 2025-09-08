from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
import os
import tempfile
import uuid

from clean_discovery import clean_discovery
from clean_vce import clean_vce
from clean_vces import clean_vces

app = Flask(__name__)

# Temporary folder for cleaned files
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "survey_cleaner")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload, clean data, and show results page."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    mode = request.form.get('mode')
    if mode not in ['discovery', 'vce', 'vces']:
        return "Invalid cleaning mode", 400

    # Save uploaded file
    temp_input = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_{file.filename}")
    file.save(temp_input)

    # Run selected cleaner
    if mode == 'discovery':
        df = clean_discovery(temp_input)
    elif mode == 'vce':
        df = clean_vce(temp_input)
    else:
        df = clean_vces(temp_input)

    # Save cleaned CSV with unique name
    output_filename = f"{uuid.uuid4()}_cleaned.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(output_path, index=False)

    # Render results page with download link
    download_url = url_for('download_file', filename=output_filename)
    return render_template('result.html', download_url=download_url)

@app.route('/download/<filename>')
def download_file(filename):
    """Serve cleaned CSV file for download."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True, download_name="cleaned_survey.csv")

if __name__ == '__main__':
    app.run(debug=True)
