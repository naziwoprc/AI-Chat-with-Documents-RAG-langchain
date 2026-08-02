from ragEngine import RAGEngine
from dotenv import load_dotenv
import os

load_dotenv()

engine = RAGEngine(
    pdf_path="Data/liver_disease.pdf",
    persist_dir="./chroma_db_langchain",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

question = input("Ask a question: ")
answer = engine.answer(question)

print("\n---Answer---")
print(answer)
