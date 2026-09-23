from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_txt(file_path):
    loader = TextLoader(file_path)
    return loader.load()


def load_markdown(file_path):
    loader = TextLoader(file_path)
    return loader.load()


def load_docx(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def load_documents(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_pdf(file_path)
    elif extension == ".txt":
        return load_txt(file_path)
    elif extension == ".md":
        return load_markdown(file_path)
    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")


def add_metadata(documents, file_path):
    file_type = os.path.splitext(file_path)[1].lower().lstrip(".")

    for document in documents:
        document.metadata["source"] = file_path
        document.metadata["file_type"] = file_type

    return documents
