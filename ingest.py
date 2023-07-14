# setting device on GPU if available, else CPU
import os
from timeit import default_timer as timer
from typing import List

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.utils import *


def load_documents(source_pdfs_path, urls) -> List:
    loader = PyPDFDirectoryLoader(source_pdfs_path, silent_errors=True)
    documents = loader.load()
    if urls is not None and len(urls) > 0:
        for doc in documents:
            source = doc.metadata["source"]
            filename = source.split("/")[-1]
            for url in urls:
                if url.endswith(filename):
                    doc.metadata["url"] = url
                    break
    return documents


def split_chunks(documents: List, chunk_size, chunk_overlap) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def generate_index(
    chunks: List, embeddings: HuggingFaceInstructEmbeddings
) -> VectorStore:
    if using_faiss:
        faiss_instructor_embeddings = FAISS.from_documents(
            documents=chunks, embedding=embeddings
        )

        faiss_instructor_embeddings.save_local(index_path)
        return faiss_instructor_embeddings
    else:
        chromadb_instructor_embeddings = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=index_path
        )

        chromadb_instructor_embeddings.persist()
        return chromadb_instructor_embeddings


# Constants
init_settings()

device_type, hf_pipeline_device_type = get_device_types()
hf_embeddings_model_name = (
    os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
)
index_path = os.environ.get("FAISS_INDEX_PATH") or os.environ.get("CHROMADB_INDEX_PATH")
using_faiss = os.environ.get("FAISS_INDEX_PATH") is not None
source_pdfs_path = os.environ.get("SOURCE_PDFS_PATH")
source_urls = os.environ.get("SOURCE_URLS")
chunk_size = os.environ.get("CHUNCK_SIZE")
chunk_overlap = os.environ.get("CHUNK_OVERLAP")

start = timer()
embeddings = HuggingFaceInstructEmbeddings(
    model_name=hf_embeddings_model_name, model_kwargs={"device": device_type}
)
end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()

if not os.path.isdir(index_path):
    print("The index persist directory is not present. Creating a new one.")
    os.mkdir(index_path)

    if source_urls is not None:
        # Open the file for reading
        file = open(source_urls, "r")

        # Read the contents of the file into a list of strings
        lines = file.readlines()

        # Close the file
        file.close()

        # Remove the newline characters from each string
        source_urls = [line.strip() for line in lines]

    print(f"Loading PDF files from {source_pdfs_path}")
    sources = load_documents(source_pdfs_path, source_urls)
    print(f"Splitting {len(sources)} PDF pages in to chunks ...")

    chunks = split_chunks(
        sources, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
    )
    print(f"Generating index for {len(chunks)} chunks ...")

    index = generate_index(chunks, embeddings)
else:
    print("The index persist directory is present. Loading index ...")
    index = (
        FAISS.load_local(index_path, embeddings)
        if using_faiss
        else Chroma(embedding_function=embeddings, persist_directory=index_path)
    )

end = timer()

print(f"Completed in {end - start:.3f}s")
