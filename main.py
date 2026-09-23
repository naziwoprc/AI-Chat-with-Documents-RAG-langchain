from ragEngine import RAGEngine
from dotenv import load_dotenv
import os

load_dotenv()


def run_cli():
    engine = RAGEngine(
        file_path="Data/test.docx",
        persist_dir="./chroma_db_langchain",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    print("RAG CLI Mode (type 'exit' to quit)\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        answer = engine.answer(question)

        print("\n--- Answer ---\n")
        print(answer)
        print("\n----------------\n")


if __name__ == "__main__":
    run_cli()
