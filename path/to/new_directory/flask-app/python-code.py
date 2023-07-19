pip install Flask PyPDF2
pip install transformers
from flask import Flask, render_template, request, redirect, url_for
import os
import PyPDF2
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    file = request.files['pdf_file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    return redirect(url_for('process_pdf', filename=file.filename))

@app.route('/process/<filename>')
def process_pdf(filename):
    pdf_text = extract_text_from_pdf(filename)
    response = get_answer_from_model(pdf_text, "What is your question?")  # Replace "What is your question?" with the user's question
    return render_template('result.html', filename=filename, response=response)

def extract_text_from_pdf(filename):
    # Use PyPDF2 to extract text from the PDF file
    pdf_text = ""
    with open(os.path.join('uploads', filename), 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def get_answer_from_model(context, question):
    # Use the transformers library to get the answer to the question from the context
    # For this example, we'll use the "distilbert-base-cased-distilled-squad" model for question-answering.
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    answer = qa_model(question=question, context=context)
    return answer["answer"]

if __name__ == '__main__':
    app.run(debug=True)
