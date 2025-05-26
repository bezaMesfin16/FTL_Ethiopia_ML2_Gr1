import re
from pathlib import Path
from typing import List, Tuple
from gensim.utils import simple_preprocess
from textblob import TextBlob
from tqdm import tqdm
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# === Load text files ===
def load_text_files(directory: Path) -> List[Tuple[str, str]]:
    texts = []
    for file_path in tqdm(directory.glob("*.txt"), desc="Loading files"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    texts.append((content, file_path.name))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return texts


# === Clean and tokenize text ===
def clean_and_tokenize(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
    text = text.lower()                           # Lowercase
    text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
    tokens = simple_preprocess(text)
    return ' '.join(tokens)


# === Optional spell correction (slow) ===
def correct_spelling(text: str) -> str:
    try:
        return str(TextBlob(text).correct())
    except Exception as e:
        print(f"Spell correction error: {e}")
        return text


# === Chunk text into overlapping windows ===
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")
    return [
        ' '.join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]


# === Main pipeline: return Document objects with metadata ===
def process_documents(input_dir: Path, chunk_size: int = 100, overlap: int = 20, correct: bool = False) -> List[Document]:
    raw_docs = load_text_files(input_dir)  # (content, filename) tuples
    documents = []

    for content, source in tqdm(raw_docs, desc="Processing documents"):
        cleaned = clean_and_tokenize(content)

        if correct:
            cleaned = correct_spelling(cleaned)

        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "title": source.replace(".txt", ""),
                    "chunk_id": i
                }
            ))

    return documents


#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
