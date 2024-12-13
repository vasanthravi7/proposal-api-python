import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough



load_dotenv()


app = Flask(__name__)

# Initialize the LLM
os.environ["TAVILY_API_KEY"] = "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"
# Initialize the LLM
llm = ChatGroq(model="Gemma2-9b-It",api_key="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")
# Create the Tavily search tool
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a professional document generator tasked with creating a comprehensive document structure.

    Instructions:
    1. Maintain the following static headings and subheadings:
    1. Executive Summary
    2. Company Background
       2.1 About our company
       2.2 Our Journey
       2.3 Facts & Figures

    2. Generate dynamic content for the under the appropriate sections.
    3. Add up to 4 additional relevant headings and subheadings that are contextually appropriate to the.
    4. Ensure the final document includes these static headings:
    6. Why our company?
    7. Team Structure
    8. Phase 1-Estimation
    9. Pricing and Estimation
    10. Relevant Experience and Case Studies

    Topic Details:
    Based on the topic develop a comprehensive, well-structured document that provides:
    - In-depth insights
    - Relevant market research
    - Strategic analysis
    - Potential opportunities and challenges

    Formatting Guidelines:
    - Use clear, professional language
    - Provide specific, actionable insights
    - Ensure logical flow between sections
    - Include data-driven observations where possible
      Search Results:
      {search_results}
      query:{query}
    Generate a document that comprehensively addresses the while maintaining the specified structure."""
    
)

# Create a function to retrieve search results
def get_search_results(query):
    return tool.invoke({"query": query})

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    
    # Perform the Tavily search
    search_results = get_search_results(query)
    
    # Prepare the context for the LLM
    context = {
        "query": query,
        "search_results": search_results
    }
    
    # You might want to use the LLM to synthesize the search results
    # This is a simplified example
    response = prompt_template.format(**context)

    # return jsonify({
    #      response
    # })
    
    return jsonify({
        "search_results": search_results,
        "formatted_prompt": response
    })

if __name__ == "__main__":
    app.run(port=5000)