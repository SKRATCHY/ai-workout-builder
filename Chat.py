import ollama 
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def load_document():
    #loader = PyPDFLoader("Workouts.pdf")
    loader = PyPDFLoader("LT workout.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    embeddings = OllamaEmbeddings(model="zephyr")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return(retriever)

def format_docs (docs) :
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context) :
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='zephyr', messages=[{'role': 'user','content' : formatted_prompt}])
    return response['message'] ['content']

def rag_chain(question) :
    retriever = load_document()
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

iface = gr. Interface (
    fn=rag_chain,
    inputs=["text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Something"
)
# Launch the app
iface.launch()
