# mtu_chatbot
# MTU Chatbot

## Overview
The MTU Chatbot leverages advanced technologies including Anthropic’s Claude 2 and OpenAI’s GPT-4 to deliver tailored recommendations. These LLMs are hosted on AWS Bedrock API and OpenAI API respectively. The application utilizes exclusive proprietary data to enhance interactions through NLP and RAG techniques. The proprietary data is loaded with Langchain. This application enabled a comparative analysis of the LLMs with Langsmith is conducted focusing on performance, accuracy, bias, safety, cost, latency, and adaptability.



## MTU Chatbot Architecture and Design
### Tools and Technologies
- **Python SDK**
- **Data Collection and Cleaning**
- **LLM Models:** Claude 2 and GPT-4
- **Embedding Models:**
  - Titan Text Embedding for Claude chatbot
  - Text Embedding Ada V2 for OpenAI chatbot
- **Prompt Templates:** Prompt, Memory templates
- **FAISS (Facebook AI Similarity Search) by Meta**
- **Langchain**
- **Streamlit**


![MTU Chatbot Architecture](images\acrhitecture.png)


## Installation
### Requirements
Ensure you have a virtual environment set up for the project to manage dependencies:
```bash
python -m venv mtu-env
mtu-env\Scripts\activate

Install required packages:

pip install -r requirements.txt


##Running the Application

##Navigate to chatbot\app.py and run the Streamlit UI:
python3 -m streamlit run chatbot/app.py


The Streamlit UI allows users to choose between GPT-4 or Claude 2 for generating responses and provides an interactive session through user input.
Testing Framework
Location

### File Structure 

![File Structure](images\file_structure.png)

Tests are located under chatbot/langsmith_test with specific scripts dedicated to evaluating different aspects of model behavior:

    Accuracy and Bias Tests:
        claude_accuracy_cot_test.py: Contextual accuracy using chain of thought (COT) for Claude 2.
        gender_bias_claude.py: Gender bias evaluation for Claude 2.
        gender_bias_gpt.py: Gender bias evaluation for GPT-4.
        gpt_accuracy_cot_test.py: Contextual accuracy for GPT-4.
        claude_bias_culture.py, gpt_bias_culture.py: Cultural bias tests for both models.
    Safety Tests:
        safetybench_claude.py: Safety testing for Claude 2.
        safetybench_gpt.py: Safety testing for GPT-4.

Data Management
Proprietary Data

Located in chatbot/data, the directory contains several proprietary PDF files such as student reviews, lecturer profiles, career progression data, fictional courses, and MTU’s prospectus handbook.
Folder Structure

Key directories:

    chatbot/app.py: Main application file for the Streamlit UI.
    chatbot/test: Used for processing proprietary files.
    chatbot/langsmith_test: Contains scripts for testing various model aspects.

Contribution

Contributions to the MTU Chatbot are welcome. Please ensure to follow the existing code style and add unit tests for any new or changed functionality.




To use this, simply copy the content into a new text file, name it `README.md`, and place it in the root directory of your project. This file will be formatted correctly for display on platforms like GitHub, where Markdown is commonly used to format repository documentation.

