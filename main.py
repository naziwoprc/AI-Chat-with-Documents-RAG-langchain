from ragEngine import RAGEngine
from dotenv import load_dotenv
import os

load_dotenv()

engine = RAGEngine(
    pdf_path="Data/liver_disease.pdf",
    persist_dir="./chroma_db_langchain",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

question = "What is the liver?"

engine.debug_retrieval(question, k=5)
print("\n--- Final Answer ---\n")
print(engine.answer(question))
print("\n--- End ---\n")
