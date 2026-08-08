from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sentence_transformers import CrossEncoder

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

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            print("API Error:", response.text)
            return "I don't know."
        data = response.json()

        if "choices" not in data:
            print("Unexpected Error:", data)
            return "I don't know."

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

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

    def answer(
        self,
        question,
        rerank_threshold=0.0,
        rerank_margin_threshold=0.5,
        embedding_margin_threshold=0.05,
    ):
        results = self.vectorstore.similarity_search_with_score(question, k=10)

        if not results:
            return "I don't know."

        distance = [distance for _, distance in results[:5]]

        mean_distance = sum(distance) / len(distance)
        std_distance = (
            sum((d - mean_distance) ** 2 for d in distance) / len(distance)
        ) ** 0.5

        dynamic_margin_threshold = std_distance

        # -------- Stage 1: Embedding confidence --------
        best_doc, best_distance = results[0]

        if len(results) > 1:
            _, second_best_distance = results[1]
            embedding_margin = second_best_distance - best_distance
        else:
            embedding_margin = 1.0

        if embedding_margin > dynamic_margin_threshold:
            docs = [doc for doc, _ in results[:3]]
            context = self.format_docs(docs)
            formatted_prompt = self.prompt.format(context=context, input=question)
            return self.llm.invoke(formatted_prompt)

        # -------- Stage 2: Cross‑Encoder Rerank --------
        reranked = self.rerank(question, results, top_k=3)

        reranked_scores = [score for _, score in reranked]

        mean_rerank = sum(reranked_scores) / len(reranked_scores)
        std_rerank = (
            sum((s - mean_rerank) ** 2 for s in reranked_scores) / len(reranked_scores)
        ) ** 0.5

        dynamic_rerank_margin_threshold = 0.5 * std_rerank

        best_doc, best_score = reranked[0]

        if best_score < rerank_threshold:
            return "I don't know."

        if len(reranked) > 1:
            _, second_best_score = reranked[1]
            rerank_margin = best_score - second_best_score

            # if rerank_margin < dynamic_rerank_margin_threshold:
            #     return "I don't know."

        docs = [doc for doc, _ in reranked]
        context = self.format_docs(docs)
        formatted_prompt = self.prompt.format(context=context, input=question)

        answer_text = self.llm.invoke(formatted_prompt)
        pages = [doc.metadata["page"] for doc in docs]
        unique_pages = sorted(set(pages))

        sources = "\n".join(f"- Page {p}" for p in unique_pages)
        final_output = f"{answer_text}\n\nSources:\n{sources}"

        return final_output

    def debug_retrieval(self, question, k=5):
        results = self.vectorstore.similarity_search_with_score(question, k=k)

        for rank, (doc, distance) in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print("Distance:", distance)
            print("Preview:", doc.page_content[:200])

    def rerank(self, question, docs_with_scores, top_k=3):
        pairs = [(question, doc.page_content) for doc, _ in docs_with_scores]

        scores = self.reranker.predict(pairs)

        docs = [doc for doc, _ in docs_with_scores]

        reranked = list(zip(docs, scores))

        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]
