# What this project is doing

This project implements a Retrieval-Augmented Generation (RAG) pipeline that combines a curated corpus of domain-specific 
papers with OpenAI's LLM capabilities. Trusted academic and technical documents are ingested, chunked, and embedded into 
a vector database to enable semantic search. When a user query is received, the system retrieves the most relevant 
document chunks and supplies them as context to the LLM, ensuring generated responses are accurate, verifiable, and 
grounded exclusively in the trusted source material.


## How to run?

- conda create -n test python=3.12 -y
- conda activate test
- pip install -r requirements.txt 
- python app.py

## Note
- You need to have openAI api key
- The data (PDF files) are just examples—you can replace them with any type of data from your own projects

## More learning source:
- https://www.youtube.com/watch?v=o126p1QN_RI
- https://www.meilisearch.com/blog/what-is-rag
