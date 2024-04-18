import os
import uuid
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from clean_data import CleanData
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
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
LANGCHAIN_HUB_API_KEY="ls__****"
uid = uuid.uuid4()
langsmith_client = LangSmithClient(api_key=LANGCHAIN_API_KEY)

uid = uuid.uuid4()
print("Created uid" , uid)




#############################   Templates     ##################################


#prompt template containg errors
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
    try:
        sample_embedding = np.array(embedding_model.embed_query(docs[0].page_content))
        print("Sample embedding of a document chunk: ", sample_embedding)
        print("Size of the embedding: ", sample_embedding.shape)

    except ValueError as error:
        if "AccessDeniedException" in str(error):
            print(f"\x1b[41m{error}\
            \nTo troubleshoot this issue please refer to the following resources.\
             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
            class StopExecution(ValueError):
                def _render_traceback_(self):
                    pass
            raise StopExecution
        else:
            raise error
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


# Dataset for testing using the error questions and the expected answers

tests = [
    {
        "input": {
            "question": "What insits mite Dr. Sofea Ahmed shair in her MA Jurnlism & Dijital Creation clasess?", #ambigious
            "chat_history": [],
            "context": ""
        },
        "prediction": {
            "expected": "In the MA in Journalism and Digital Content Creation program, students will benefit from Dr. Sophia Ahmed's extensive experience as a journalist and digital content strategist focusing on multimedia journalism. Her expertise in Digital Storytelling, Media Ethics, and Online Journalism equips students with the necessary skills to excel in modern media environments."
        }
    },
    {
        "input": {
            "question": "Wht notble contributions haz Professor Maria Gonzalez made to MEng Civil Engineering at MTU, especially sustainable urban dev?",
            "chat_history": [],
            "context": ""
        },
        "prediction": {
            "expected": "Professor Maria Gonzalez, leading the MEng in Civil Engineering course at MTU, has made notable contributions to sustainable urban development projects. Her expertise in Structural Engineering, Environmental Sustainability, and Urban Planning deeply influences the course content, equipping students to address modern challenges with innovative and sustainable solutions. Professor Gonzalez’s real-world experience and research offer students a comprehensive understanding of how civil engineering positively impacts urban environments."
        }
    },

    {
        "input": {
            "question": "Wat does Dr. Alexei Petrov do?", #ambigious
            "chat_history": [],
            "context": ""
        },
        "prediction": {
            "expected": " In the MSc in Data Science & Analytics program at MTU, Dr. Alexei Petrov, a Data Scientist with a PhD in Computer Science specializing in machine learning applications in finance, teaches modules related to Machine Learning, Statistical Modelling, and Big Data Analytics. His expertise ensures students acquire in-depth knowledge and practical skills in applying machine learning techniques to solve complex problems in finance and beyond."
        }
    },
    {
        "input": {
            "question": "What's usually next for MA Play Therapy grads at MTU?", #ambigious
            "chat_history": [],
            "context": ""
        },
        "prediction": {
            "expected": "Graduates of the MA in Play Therapy at MTU typically pursue careers as Child Play Therapists in various settings. The program supports their preparation through a blend of theoretical learning and practical experience, ensuring graduates are ready to employ play therapy effectively to aid children's emotional and psychological healing and development."
        }
    },
    
    {
        "input": {
            "question": "What iz mode of delivery 4 MA in Integrative Psychotherapy @ MTU?",
            "chat_history": [],
            "context": ""
        },
        "prediction": {
            "expected": "The mode of delivery for the MA in Integrative Psychotherapy at MTU is as follows: The program is offered over a two-year cycle, with the next intake scheduled for September 2022. It is a part-time program, allowing students to balance their studies with other commitments. During the first year (60 credits), students attend taught modules at the college, along with supervision sessions. These sessions are typically held on weekday evenings (6:30 pm to 9:30 pm) and on Saturdays and Sundays. In the second year (30 credits), students engage in directed/supervised learning, focusing on research and writing the Reflective Practicing dissertation. Additionally, during this year, students undertake their Mental Health Placement. Overall, the program combines in-person instruction, supervision, and self-directed learning to provide a comprehensive learning experience."
        }
    }
]

#----------------------------------------------------------------------------------------

def input_mapper(example):
    adapted_example = {
        "question": example["input"]["question"],
        "chat_history": example["input"].get("chat_history", []),  # Assuming default if not present
        "prediction": example["prediction"]["expected"]  # Correct extraction of prediction
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


# Adjusted sequence
adapted_tests = [input_mapper(test) for test in tests]




for adapted_test in adapted_tests:
    try:
        response = langsmith_client.create_example(
            dataset_id=dataset.id,
            inputs={"question": adapted_test["question"], "chat_history": adapted_test["chat_history"]},  # Now correctly structured
            outputs={"expected": adapted_test["prediction"]}  # Now correctly accesses 'prediction'
        )
        print("Example added successfully.")
    except requests.exceptions.HTTPError as e:
        print(f"Error adding example: {e}")


#------------------------
criterion = {"conciseness": "Is the MTU chatbot response concise and to the point?"}

evaluation_config = RunEvalConfig(
    input_key="question",
    criteria_evaluators=[
        RunEvalConfig.Criteria(criterion)
    ],
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
    project_name=f"QA_claude 2_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    
)


# python3 robust_test_gpt.py

