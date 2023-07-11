import os
from timeit import default_timer as timer
from typing import List

import torch
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

## utility functions

import os
import textwrap


def load_documents(source_pdfs_path, urls) -> List:
    loader = DirectoryLoader(source_pdfs_path, glob="./*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    for doc in documents:
        source = doc.metadata["source"]
        filename = source.split("/")[-1]
        src = doc.metadata["source"]
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


def generate_index(chunks: List, embeddings: HuggingFaceInstructEmbeddings) -> Chroma:
    chromadb_instructor_embeddings = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=index_path
    )

    chromadb_instructor_embeddings.persist()
    return chromadb_instructor_embeddings


# Constants
load_dotenv(override=True)

device_type = os.environ.get("HF_EMBEDDINGS_DEVICE_TYPE") or device.type
hf_embeddings_model_name = (
    os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
)
index_path = os.environ.get("CHROMADB_INDEX_PATH")
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

    # Open the file for reading
    file = open(source_urls, "r")

    # Read the contents of the file into a list of strings
    lines = file.readlines()

    # Close the file
    file.close()

    # Remove the newline characters from each string
    source_urls = [line.strip() for line in lines]

    # Print the modified list
    # print(source_urls)

    print(f"Loading {len(source_urls)} PDF files from {source_pdfs_path}")
    sources = load_documents(source_pdfs_path, source_urls)
    print(f"Splitting {len(sources)} PDF pages in to chunks ...")

    chunks = split_chunks(
        sources, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
    )
    print(f"Generating index for {len(chunks)} chunks ...")

    index = generate_index(chunks, embeddings)
else:
    print("The index persist directory is present. Loading index ...")
    index = Chroma(embedding_function=embeddings, persist_directory=index_path)

end = timer()

print(f"Completed in {end - start:.3f}s")
