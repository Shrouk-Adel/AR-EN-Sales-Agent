from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langdetect import detect
from langchain.docstore.document import Document
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document
from operator import itemgetter
from langchain_huggingface import HuggingFacePipeline
# Use a pipeline as a high-level helper
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever
import uuid

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


from huggingface_hub import login
login(token='your api token')

# Path to your JSON file
file_path = r"/kaggle/input/electronics/electronics_data.json"

# Define a schema to extract each JSON object
jq_schema = ".[]"  # Extracts each JSON object from a list

# Function to extract text and metadata
def metadata_func(record, index):  # Add index as the second parameter
    return {"source": "json_data", "index": index}


# Load JSON data
loader = JSONLoader(
    file_path=file_path,
    jq_schema=jq_schema,
    text_content=False,  # Prevents automatic extraction as a string
    metadata_func=metadata_func
)

# Load documents
docs = loader.load()


# Groq_API_Key =os.getenv('Groq_API_Key')
model =ChatGroq(
    model ='gemma2-9b-it',
    groq_api_key ='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',
    temperature=0.1
    )

template="summarize the following document:{doc}"
prompt =ChatPromptTemplate.from_template(template)

chain =(
    {'doc':lambda x:x.page_content}
    |prompt
    |model
    |StrOutputParser()
)

summarize=chain.batch(docs,{'max_concurrency':1})

# use vectore store to index child chunks
vectorstore =Chroma(
    collection_name='summarise',
    embedding_function=HuggingFaceBgeEmbeddings(model_name ='BAAI/bge-m3')
    )

# storage layer for parent documents
store =InMemoryByteStore()
id_key ='doc_id'

retriver =MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key =id_key,
    search_kwargs={'k':15}

)

# This ensures each document has a unique ID for retrieval.
doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_doc =[Document(page_content=s,metadata={id_key:doc_ids[i]}) for i,s in enumerate(summarize)]

# store summary_doc in vectoredb and store original docs in memory store
retriver.vectorstore.add_documents(summary_doc)
retriver.docstore.mset(list(zip(doc_ids,docs)))

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model and tokenizer
model_name = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto"
)

# Create the Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=20000,
    temperature=0.3,
    return_full_text=False 
)

# Wrap the pipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

ar_template = """
إنت دلوقتي شغال وكيل مبيعات مصري محترف لشركة  الكتروفيرس وهي شركة متخصصة في صناعة  الالكترونيات مطلوب منك:
١- تجاوب على كل الأسئلة اللي ليها علاقة بالمنتجات
خلي اسماء المنتجات باللغة الانجليزية
وحاول تكون اجابتك واضحة ومنظمة
السياق بتاع المنتجات: {context}
تاريخ المحادثة:{history}
السؤال بتاعك: {question}
"""

en_template="""you act as a sales man for mobica products answer
the following question from the following context and follow this rules:
- be organized and clear
- reply with detials for each product

question:{question}
chat history:{history}
context:{context}
 """

def detect_language(question):
    language =detect(question)
    if language =='ar':
        return 'Arabic'
    else:
        return 'English'


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# Restructure the chain to ensure proper processing
retriever_chain =(
    retriver 
    | format_docs
)

def get_response(query):
    # Get relevant documents
    retrieved_docs = retriever_chain.invoke(query)
    
    # detect language of query
    lang=detect_language(query)
    if lang =='Arabic':
       prompt = ChatPromptTemplate.from_template(ar_template)
    else: 
       prompt = ChatPromptTemplate.from_template(en_template)

    # Format the prompt with context and question
    formatted_prompt = prompt.format(
        context=retrieved_docs, 
        history =memory.load_memory_variables({}).get("history", ""),
        question=query
    )
    
    # Get response from the LLM directly
    response = llm.invoke(formatted_prompt)

     # Save the new interaction to memory
    memory.save_context({"question": query}, {"response": response})
    
    return response



# Pydantic model for input and output
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str


app =FastAPI()

@app.post('/answer')
async def get_answer(question: Question) -> Answer:
    """
    Example post request body:
    {
        "question": "What is the capital of France?"
    }
    Example response:
    {
        "answer": "The capital of France is Paris."
    }
    """
    answer = get_response(question.question)
    return Answer(answer=answer)









