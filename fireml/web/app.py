import os
import json
import uuid
import logging
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from fireml.data_loader import load_data
from fireml.main_pipeline import run_full_pipeline

tasks = {}
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')
app.config['OUTPUT_FOLDER'] = os.path.abspath('output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def run_analysis_task(task_id: str, filepath: str, target: str, task_type: str):
    """The function that runs in a background thread."""
    tasks[task_id]['status'] = 'running'
    logger.info(f"Starting analysis for task_id: {task_id}")
    try:
        df, _ = load_data(filepath)
        if df.empty:
            raise ValueError("Failed to load data.")
        
        task_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
        
        report = run_full_pipeline(
            df=df,
            target_column=target,
            task_type=task_type,
            output_dir=task_output_dir
        )
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = {
            'report_path': f"/api/results/{task_id}/evaluation_report.html",
            'summary': report.get('summary', {})
        }
        logger.info(f"Analysis successful for task_id: {task_id}")
    except Exception as e:
        logger.error(f"Analysis failed for task_id: {task_id}. Error: {str(e)}", exc_info=True)
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Handle file upload and trigger analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(file.filename) #type:ignore
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'pending'}
    
    target = request.form.get('target')
    task_type = request.form.get('task', 'auto')
    
    thread = Thread(target=run_analysis_task, args=(task_id, filepath, target, task_type))
    thread.start()
    
    return jsonify({
        'message': 'Analysis started.',
        'task_id': task_id,
        'status_url': f'/api/status/{task_id}'
    }), 202

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id: str):
    """Check the status of an analysis task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)

@app.route('/api/results/<task_id>/<path:filename>')
def get_output_file(task_id: str, filename: str):
    """Serve result files from the task's output directory."""
    directory = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    return send_from_directory(directory, filename)

def main():
    """Run the Flask development server."""
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    main()
