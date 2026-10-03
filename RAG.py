#This is the whole code for RAG
#detailed page: https://github.com/Li-Zou/Retrieval-Augmented-Generation/tree/main
#date: 2026.8.24

import os
os.chdir(r'C:\Users\li\Downloads')
from langchain_community.document_loaders import PyPDFLoader#, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #from langchain.text_splitter import RecursiveCharacterTextSplitter  (this one does not work)
from pathlib import Path


### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}") 
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
all_pdf_documents = process_all_pdfs("data") # Process all PDFs in the data directory

### Text splitting get into chunks
def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    return split_docs
chunks=split_documents(all_pdf_documents)
del all_pdf_documents





from sentence_transformers import SentenceTransformer # embedding model
import chromadb  # for vector store
import uuid

emb_model = SentenceTransformer("all-MiniLM-L6-v2")  #embedding model


persist_directory="data/vector_store"
os.makedirs(persist_directory, exist_ok=True)
client = chromadb.PersistentClient(path=persist_directory)  # Create persistent ChromaDB client

# Get or create collection
collection = client.get_or_create_collection(name="pdf_documents",metadata={"description": "PDF document embeddings for RAG"})


### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]
embeddings=emb_model.encode(texts, show_progress_bar=False) ## Generate the Embeddings


#documents=chunks.copy()
def prepare_embedding_data(chunks, embeddings):
    ids = []
    metadatas = []
    documents_text = []
    embeddings_list = []
    for i, (doc, embedding) in enumerate(zip(chunks, embeddings)):
        # Generate unique ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        ids.append(doc_id)
        # Prepare metadata
        metadata = dict(doc.metadata)
        metadata['doc_index'] = i
        metadata['content_length'] = len(doc.page_content)
        metadatas.append(metadata)
        # Document content
        documents_text.append(doc.page_content)
        # Embedding
        embeddings_list.append(embedding.tolist())
    return (ids,metadatas, documents_text,embeddings_list)
ids,metadatas, documents_text,embeddings_list=prepare_embedding_data(chunks, embeddings)

collection.add(ids=ids, embeddings=embeddings_list,metadatas=metadatas, documents=documents_text) # Add vector data to collection
print(f"Successfully added {len(chunks)} documents to vector store")
print(f"Total documents in collection: {collection.count()}") 
del ids,metadatas, documents_text,embeddings_list




query='what is temporal network prediction'
query_embedding = emb_model.encode([query])[0]

results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3) # searched results

def re_organize_search_result(results):
    retrieved_docs = []
    if results['documents'] and results['documents'][0]:
        documents00 = results['documents'][0]
        metadatas00 = results['metadatas'][0]
        distances00 = results['distances'][0]
        ids = results['ids'][0]
        for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents00, metadatas00, distances00)):
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            similarity_score = 1 - distance
            if similarity_score >= 0:
                retrieved_docs.append({
                    'id': doc_id,
                    'content': document,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'distance': distance,
                    'rank': i + 1})
        print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
    return retrieved_docs
retrieved_docs=re_organize_search_result(results)                                    
del  results, query_embedding                



from langchain_openai import ChatOpenAI  #from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate #from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage#, SystemMessage #from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-3.5-turbo", temperature=0.1)

prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

            Context:
            {context}
            
            Question: {question}
            
            Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so.""")
    
context="\n\n".join([doc['content'] for doc in retrieved_docs]) if retrieved_docs else ""
formatted_prompt = prompt_template.format(context=context, question=query)
messages = [HumanMessage(content=formatted_prompt)]

response = llm.invoke(messages)
print(response.content)

response1 = llm.invoke(query)
print(response1.content)































