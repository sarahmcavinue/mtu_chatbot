
from datetime import datetime
import os
import uuid
import warnings
import numpy as np
import requests
from clean_data import CleanData
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.client import Client as LangSmithClient
warnings.filterwarnings('ignore')



LANGCHAIN_TRACING_V2="true", 
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com",
LANGCHAIN_API_KEY= "ls__****"
LANGCHAIN_PROJECT="evaluators", 
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
LANGCHAIN_HUB_API_KEY="ls__***"
uid = uuid.uuid4()
langsmith_client = LangSmithClient(api_key=LANGCHAIN_API_KEY)

uid = uuid.uuid4()
print("Created uid" , uid)




#############################   Templates     ##################################


# Define your prompt templates
prompt_template = """
Human: You are an AI chatbot designed to simulate guidance based on 8 specific PDF documents related to Munster Technological University (MTU), 
also known as CIT, Cork Institute of Technology, and [other names]. Your responses should reflect information typically found in these documents, 
focusing on postgraduate courses and their lecturers. If specific information from these documents isn't available, respond with what you would 
expect based on your training. The user seeks advice on postgraduate courses at MTU/CIT aligned with their interests, goals, professional background, 
and academic qualifications.

Current conversation:
<context>{context}</context>
Begin!
Question: {question}
Assistant:
"""

_template = """
Human: Given the following conversation and a follow-up question, rephrase the follow-up question into a standalone question without changing the content of the given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: Assistant:
"""

# Example usage of the prompt template
context = "The AI chatbot provides information about postgraduate courses at MTU..."

question = "What courses are available in Data Science?"

# Format the prompt with the actual context and question
full_prompt = prompt_template.format(context=context, question=question)



############################### Setting the AI model #########################################

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

#GPT-4
llm_model_gpt = ChatOpenAI(temperature=0.5,model_name="gpt-4", max_tokens=300)
embedding_model_gpt = OpenAIEmbeddings(model='text-embedding-ada-002')


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
process_pdf_documents(urls, download_directory, embedding_model_gpt)

docs_to_be_processed = process_pdf_documents(urls, download_directory, embedding_model_gpt)

#-----------------------------------------------------------------------------------------------------------------------


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)



def load_chain():

    new_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    chain = ConversationalRetrievalChain.from_llm(
        llm= llm_model_gpt, 
        retriever=FAISS.from_documents(docs_to_be_processed, embedding_model_gpt).as_retriever(search_kwargs={'k': 4}),
        memory=new_memory,
        verbose=True,
        chain_type="stuff",
        condense_question_prompt= CONDENSE_QUESTION_PROMPT)
    
    chain.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template)
    
    return chain



#----------------------------------------------------------------------------------------

# Dataset to test

tests = [
     {
        "inputs": {
            "question": "How many courses focus on cyber-related subjects?",
            "chat_history": [],  # Assuming chat history is not used; otherwise, populate as needed
            "context": "Assuming Masters in Hacking and Cyber Warfare, and Masters in Toxic Chemical Handling are considered cyber-related."
        },
        "outputs": {
            "expected": "2"
        }
    },
    {
        "inputs": {
            "question": "What is the fee for the Masters in Hacking and Cyber Warfare course?",
            "chat_history": [],
            "context": ""  # Context is optional, provide if relevant
        },
        "outputs": {
            "expected": "€5,500"
        }
    },
    {
        "inputs": {
            "question": "Who is the instructor for the MA in Integrative Psychotherapy?",
            "chat_history": [],
            "context": ""
        },
        "outputs": {
            "expected": "Dr. Anita Desai"
        }
    },
    {
        "inputs": {
            "question": "What is the background of the instructor for the MA in Play Therapy?",
            "chat_history": [],
            "context": ""
        },
        "outputs": {
            "expected": "Child Psychologist with extensive experience in pediatric mental health care settings."
        }
    },
    {
        "inputs": {
            "question": "Which course focuses on environmental sustainability?",
            "chat_history": [],
            "context": ""
        },
        "outputs": {
            "expected": "MEng in Civil Engineering"
        }
    },
    {
        "inputs": {
            "question": "What expertise is required for the Masters in Toxic Chemical Handling course?",
            "chat_history": [],
            "context": "This question assumes a knowledge of prerequisites for a specialized course."
        },
        "outputs": {
            "expected": "Financial Reporting, Taxation, and Auditing."
        }
    },
    {
        "inputs": {
            "question": "How many courses are offered in engineering?",
            "chat_history": [],
            "context": "Considering MEng in Civil Engineering, MEng in Structural Engineering, MSc in Building Information Modelling and Digital AEC."
        },
        "outputs": {
            "expected": "3"
        }
    },
    {
        "inputs": {
            "question": "What is the professional outcome for graduates of the MA in Play Therapy?",
            "chat_history": [],
            "context": ""
        },
        "outputs": {
            "expected": "Child Play Therapist in a pediatric hospital"
        }
    },
    {
        "inputs": {
            "question": "Which course equips students for a career as a Licensed Psychotherapist?",
            "chat_history": [],
            "context": ""
        },
        "outputs": {
            "expected": "MA in Integrative Psychotherapy"
        }
    }
    


]







#----------------------------------------------------------------------------------------

def input_mapper(example):
    adapted_example = {
        "question": example["input"]["question"],
        "chat_history": example["input"].get("chat_history", []),  
        "prediction": example["prediction"]["expected"]  
    }
    return adapted_example




#----------------------------------------------------------------------------------------

# Initialize dataset
dataset_name = f"GPT-4 COT and Relevance Test - {uuid.uuid4()}"
try:
    # Create the dataset
    dataset = langsmith_client.create_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' created successfully.")
except requests.exceptions.HTTPError as e:
    print(f"Error creating dataset: {e}")
    exit()

adapted_tests = [input_mapper(test) for test in tests]




for adapted_test in adapted_tests:
    try:
        response = langsmith_client.create_example(
            dataset_id=dataset.id,
            inputs={"question": adapted_test["question"], "chat_history": adapted_test["chat_history"]},  
            outputs={"expected": adapted_test["prediction"]} 
        )
        print("Example added successfully.")
    except requests.exceptions.HTTPError as e:
        print(f"Error adding example: {e}")


#------------------------
evaluation_config = RunEvalConfig(
    input_key="question",
    evaluators=[
        "qa",
        "context_qa",
        "cot_qa",
    ],
)


#----------------------------------


run_on_dataset(
    client=langsmith_client,
    dataset_name=dataset_name,
    llm_or_chain_factory=load_chain,
    evaluation=evaluation_config,
    input_key="question",
    project_name=f"QA_gpt4_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    
)


# python3 gpt_accuracy_cot_test.py
