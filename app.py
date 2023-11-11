import streamlit as st
import pickle
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


with st.sidebar:
    st.title("PDF-CHAT")
    st.markdown('''
    ## Upload your PDF file
    This is PDF-CHAT, a web app that allows you to upload a PDF file and get a summary of the text in the PDF file.     
    ''')
    #add_vertical_space(5)
    st.write("Made by Preben Andersen")


def main():

    load_dotenv()


    st.title("Chat with any PDF")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        ## Lage metode av dette
        ## -----------------
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # embedding

        store_name = +pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f: ## read binary
                ## load embeddings --> knowledgde base
                VectoreStore = pickle.load(f)
           #st.write("Embeddings loaded from disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectoreStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f: ## write binary
                pickle.dump(VectoreStore, f)
            
            #st.write("Embeddings calculated and saved to disk")

        ## -----------------
        ##st.write(chunks)

        # Accept user input
        
        ### Semantic search
        user_input = st.text_input("Ask questions asbout the PDF file")
        if user_input:
            docs = VectoreStore.similarity_search(user_input, k=3) ## k = number of results - top 3
            
            llm = OpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo")
            
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as callback:

                response = chain.run(input_documents=docs, question=user_input)
                print(callback)
            st.write(response)


if __name__ == '__main__':
    main()