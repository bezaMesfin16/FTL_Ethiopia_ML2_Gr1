from src.helper import process_documents,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pathlib import Path
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# === CONFIGURATION ===
DATA_DIR = Path("data")   # Update with your actual directory path
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
ENABLE_SPELL_CORRECTION = False   # Set to True if needed (very slow)

# === Run the pipeline ===
if __name__ == "__main__":
    
    input_path = Path(DATA_DIR)
    documents = process_documents(
        input_dir=input_path,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        correct=ENABLE_SPELL_CORRECTION
    )
    print(f"\n✅ Total document chunks created: {len(documents)}")

embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "cdssrag"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
