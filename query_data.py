import argparse
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
md_file = "alice_in_wonderland.md"
def ingest_md_to_chroma(md_file_name):
    with open(md_file, "r", encoding="utf-8") as file:  # Specify UTF-8 encoding
        md_content = file.read()

    # Process the Markdown file (assumes one document per file).
    doc = Document(page_content=md_content, metadata={"source": md_file})
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    db.add_documents([doc])
    # Construct the path to the markdown file
    md_file_path = os.path.join("data", "books", md_file_name)
    
    # Check if the file exists
    if not os.path.exists(md_file_path):
        print(f"File '{md_file_path}' does not exist.")
        return

    # Read the markdown file
    with open(md_file_path, "r") as file:
        md_content = file.read()

    # Convert markdown content into a Document
    doc = Document(page_content=md_content, metadata={"source": md_file_name})

    # Initialize Chroma with embedding function
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Add the document to the database
    db.add_documents([doc])
    db.persist()
    print(f"Document '{md_file_name}' added to the Chroma database.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--md_file", type=str, help="Markdown file to ingest (provide file name only).", required=False
    )
    args = parser.parse_args()
    query_text = args.query_text

    # Ingest the file if provided
    if args.md_file:
        ingest_md_to_chroma(args.md_file)

    # Initialize the database and perform the query
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
        return

    # Format the response
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

import chardet

file_path = "./data/books/alice_in_wonderland.md"
with open(file_path, "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(result)

if __name__ == "__main__":
    main()
