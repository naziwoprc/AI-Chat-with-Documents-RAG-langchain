from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os
import requests


class SimpleNvidiaLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def invoke(self, prompt: str):
        url = "https://integrate.api.nvidia.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        data = response.json()

        return data["choices"][0]["message"]["content"]


class RAGEngine:
    def __init__(self, pdf_path, persist_dir, api_key):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        # Embedding
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embedding, persist_directory=persist_dir
        )

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM
        self.llm = SimpleNvidiaLLM(api_key)

        # Prompt
        self.prompt = ChatPromptTemplate.from_template("""
            Answer ONLY using the provided context.
            If the answer is not in the context, say "I don't know."

            Context:
            {context}

            Question:
            {input}
            """)

        self.chain = (
            {
                "context": self.retriever | RunnableLambda(self.format_docs),
                "input": RunnablePassthrough(),
            }
            | self.prompt
            | RunnableLambda(lambda x: self.llm.invoke(x.to_string()))
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question):
        return self.chain.invoke(question)
