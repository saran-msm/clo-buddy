from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from utils.document_processor import DocumentProcessor
from utils.guardrails import GuardRails
from utils.evaluation import Evaluator, DocumentEvaluator
import os
from werkzeug.utils import secure_filename
import logging
from concurrent.futures import TimeoutError
from concurrent.futures import ThreadPoolExecutor

# Load environment variables before any other imports
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'  # Create this folder

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize components
document_processor = DocumentProcessor()
guardrails = GuardRails()
evaluator = DocumentEvaluator()

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Process in chunks with timeouts
            content = document_processor.extract_text(file, file.filename)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                summary_future = executor.submit(document_processor.generate_summary, content)
                analysis_future = executor.submit(document_processor.extract_highlights, content)
                
                # Get results with timeout
                summary = summary_future.result(timeout=30)
                analysis = analysis_future.result(timeout=30)

            # Add evaluation
            evaluation_report = evaluator.evaluate_document(
                generated_text=summary,  # Your generated summary
                reference_text=content  # Original document text
            )
            
            formatted_evaluation = evaluator.format_evaluation_results(evaluation_report)
            
            return jsonify({
                'summary': summary,
                'key_highlights': analysis['highlights'],
                'legal_references': analysis['references'],
                'actionable_insights': analysis['insights'],
                'status': 'success',
                'disclaimer': 'This analysis is for informational purposes only and should not be considered as legal advice.',
                'evaluation': formatted_evaluation
            })

        except TimeoutError:
            return jsonify({
                'error': 'Processing timeout. Please try with a smaller document.'
            }), 408

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        response = document_processor.process_question(question)
        safe_response = guardrails.validate_response(response)
        
        return jsonify({
            'summary': safe_response,
            'highlights': '',
            'actionable_insights': ''
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search_documents():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        results = document_processor.search_similar_content(query)
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this to handle cleanup when the app shuts down
@app.teardown_appcontext
def shutdown_executor(exception=None):
    document_processor.executor.shutdown(wait=False)

if __name__ == '__main__':
    app.run(port=5001) 