from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def create_vector_store(chunks):
    """
    Create a FAISS vector store from a list of text chunks.
    """
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    """
    Build a conversational retrieval chain with memory.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain

def get_answer_from_chunks(question, chunks):
    """
    Convenience function to create a vector store on the fly
    and generate an answer for a single question.
    """
    # Create the vector store
    vector_store = create_vector_store(chunks)

    # Create the conversational chain
    chain = get_conversational_chain(vector_store)

    # Run the chain with the question
    result = chain({"question": question})

    # Debug print to see what keys are returned
    print("Chain result:", result)

    # Extract the answer text safely
    if "answer" in result:
        answer = result["answer"]
    elif "result" in result:
        answer = result["result"]
    else:
        answer = "No answer returned."

    return answer
