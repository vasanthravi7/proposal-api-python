import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="""
    Keep my static headings and subheadings, then create the dynamic table of contents based on the "{topic}", keep the below headings and subheadings before and after the dynamic table of contents. 
    My static headings
    1. Executive Summary
    2. Company Background
    2.1 About our company
    2.2 Our Journey
    2.3 Facts & Figures
    Add up to 4 more relevant headings and subheadings. Then, append these static headings at the end:
    then some more static headings:
    6. Why our company?
    7. Team Structure
    8. Phase 1-Estimation
    9. Pricing and Estimation
    10. Relevant Experience and Case Studies
    """
)

@app.route('/generate-toc', methods=['POST'])
def generate_toc():
    data = request.json
    topic = data.get('topic', None)
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        # Generate ToC using the prompt and OpenAI
        prompt = prompt_template.format(topic=topic)
        toc_response = llm.invoke(prompt)

        # Return the generated Table of Contents
        return jsonify({"table_of_contents": toc_response.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)