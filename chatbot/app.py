
import os
import time
import warnings
import boto3
import numpy as np
import streamlit as st
from clean_data import CleanData
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI

# Disable warnings
warnings.filterwarnings('ignore')





#############################   Templates     ##################################

contexts = {
    "course_details": "Information about MTU's postgraduate courses, including course content, duration, mode of delivery and prerequisites.",
    "lecturer_info": "Profiles and backgrounds of lecturers at MTU, including their academic and professional experience.",
    "student_reviews": "Reviews of MTU courses from past students.",
    "earnings": "Earnings of past students.",
    "toxic_courses": "List of toxic courses at MTU.",
    "career_paths": "Career paths and jobs that past students attained following their journey at MTU.",
}

user_question = "Can you tell me about data science courses at MTU and their lecturers?"

# Determine the context based on the question
if "courses" in user_question and "lecturers" in user_question:
    selected_context = contexts["course_details"] + " " + contexts["lecturer_info"]
elif "courses" in user_question:
    selected_context = contexts["course_details"]
elif "lecturers" in user_question:
    selected_context = contexts["lecturer_info"]
elif "reviews" in user_question:
    selected_context = contexts["student_reviews"]
elif "earnings" in user_question:
    selected_context = contexts["earnings"]
elif "toxic" in user_question:
    selected_context = contexts["toxic_courses"]

prompt_template = """
Human: As an AI with knowledge about MTU to provide guidance based on the following context. MTU is also called Munster Technological University, CIT and Cork Institute of Technology:
{context}
Current conversation:
Question: {question}
Assistant:
"""




_template = """
Human: Given the following conversation and a follow-up question, rephrase the follow-up question into a standalone question without changing the content of the given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: Assistant:"
"""


# Example usage of the prompt template to enable context
context = "The user is interested in Data Science postgraduate courses."
question = "What courses are available in Data Science?"




############################### Setting the AI model #########################################


#streamlit UI for model selection
company_logo = 'https://www.app.nl/wp-content/uploads/2019/01/Blendle.png'

st.set_page_config(page_title="MTU Chatbot", page_icon=company_logo)



model_choices = {
    'GPT-4': {
        'llm_model': 'gpt-4',
        'embedding_model': 'text-embedding-ada-002'
    },
    'Claude 2': {
        'llm_model': 'anthropic.claude-v2',
        'embedding_model': 'amazon.titan-embed-text-v1'
    }
}

# Streamlit UI for model selection
selected_model = st.selectbox("Select the AI model:", list(model_choices.keys()))

# AWS session and client setup for Claude v2
session = boto3.Session(profile_name='sarah')
boto3_bedrock = session.client(
    service_name="bedrock-runtime",
    region_name="eu-central-1"
)


# Format the prompt with the actual context and question
full_prompt = prompt_template.format(context=context, question=question)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)



# Conditional setup based on the selected model
if selected_model == 'Claude 2':
    llm_model = Bedrock(model_id=model_choices[selected_model]['llm_model'], client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':300, 'temperature':0.5})
    embedding_model = BedrockEmbeddings(model_id=model_choices[selected_model]['embedding_model'], client=boto3_bedrock)
    llm_model_st = 'Bedrock Claude Instant by Anthropic'
    embedding_model_st = 'amazon.titan-embed-text-v1'
elif selected_model == 'GPT-4':
    llm_model = ChatOpenAI(temperature=0.5,model_name="gpt-4", max_tokens=300)
    embedding_model = OpenAIEmbeddings(model=model_choices[selected_model]['embedding_model'])
    llm_model_st = 'Azure OpenAI GPT-4'
    embedding_model_st = 'text-embedding-ada-002'
    

# data preparation

def process_pdf_documents(urls, download_directory, embedding_model):
    # Create the necessary directory
    os.makedirs(download_directory, exist_ok=True)

    # Create an instance of the CleanData class and download files
    clean_data_instance = CleanData(data_dir=download_directory, urls=urls)
    clean_data_instance.download_files()

    # Load the documents and split them into smaller chunks
    loader = PyPDFDirectoryLoader(f"./{download_directory}/")
    documents = loader.load()
    
    # Character split setup
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Check documents are loaded and split correctly
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(docs)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
    print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

    # Process embedding
    try:
        sample_embedding = np.array(embedding_model.embed_query(docs[0].page_content))
        print("Sample embedding of a document chunk: ", sample_embedding)
        print("Size of the embedding: ", sample_embedding.shape)

    except ValueError as error:
        if "AccessDeniedException" in str(error):
            print("\x1b[41mError: Access to embedding model is denied.\
              \nPlease check your access permissions or ensure that the embedding model is properly configured.\
              \nFor troubleshooting, refer to the documentation or contact your administrator.\x1b[0m\n")
        else:
            print("\x1b[41mError: Failed to process embedding.\
              \nPlease check your input data or ensure that the embedding model is properly configured.\
              \nFor troubleshooting, refer to the documentation or contact support.\x1b[0m\n")
    return docs

urls = [
    "https://www.mycit.ie/contentfiles/careers/choosing%20a%20postgraduate%20course.pdf",
    "https://cieem.net/wp-content/uploads/2019/02/Finding-the-Right-Course-Postgraduate-Study.pdf",
    "https://www.cit.ie/contentfiles/postgrad/Final-Postgraduate-Handbook.pdf",
    "chatbot/data/earnings-4.pdf", # earnings
    "chatbot/data/Fictional_toxic_postgrad_courses-1-1.pdf", # toxic
    "chatbot/data/List of Lecturers for Post graduate Courses at MTU-1.pdf", # lecturers
    "chatbot/data/Reviews of career jobs.pdf", # career paths
    "chatbot/data/MTU Student Course Reviews.pdf",
]


download_directory = "test"
process_pdf_documents(urls, download_directory, embedding_model)

docs_to_be_processed = process_pdf_documents(urls, download_directory, embedding_model)



######################################   Conversation Chain           ###########################################
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)



def load_chain(llm_model, embedding_model):
    chain = ConversationalRetrievalChain.from_llm(
        llm= llm_model, 
        retriever=FAISS.from_documents(docs_to_be_processed, embedding_model).as_retriever(search_kwargs={'k': 4}),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True,
        chain_type='stuff',
        condense_question_prompt= CONDENSE_QUESTION_PROMPT)
    
    chain.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template)
    
    return chain



if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Define title and subheader
st.title("👩‍💻 MTU Postgraduate Course Selector Advice Chatbot")
st.subheader(f"Powered by AI Model from: {llm_model_st}, embeddings from {embedding_model_st}, Streamlit and Langchain 🦜🔗")

 # Load the chain
chain = load_chain(llm_model, embedding_model)

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human! I am MTU's smart AI. How can I help you today?"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = chain({"question": query})
        response = result['answer']
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


#runwith: python3 -m streamlit run app.py  


