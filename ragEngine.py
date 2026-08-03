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
        embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True}
        )

        # Vector Store
        if os.path.exists(persist_dir):
            self.vectorstore = Chroma(
                persist_directory=persist_dir, embedding_function=embedding
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

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

    def answer(self, question, distance_threshold=0.6, margin_threshold=0.05):
        results = self.vectorstore.similarity_search_with_score(question, k=3)

        if not results:
            return "I don't know."

        best_doc, best_distance = results[0]

        if best_distance > distance_threshold:
            return "I don't know."

        # margin
        if len(results) > 1:
            _, second_distance = results[1]
            margin = second_distance - best_distance
            if margin < margin_threshold:
                return "I don't know."

        return self.chain.invoke(question)

    def debug_retrieval(self, question, k=5):
        results = self.vectorstore.similarity_search_with_score(question, k=k)

        for rank, (doc, distance) in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print("Distance:", distance)
            print("Preview:", doc.page_content[:200])
