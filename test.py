def ingest_md_to_chroma(md_file_name):
    # Construct the path to the markdown file.
    md_file_path = os.path.join("data", "books", md_file_name)
    
    # Check if the file exists.
    if not os.path.exists(md_file_path):
        print(f"File '{md_file_path}' does not exist.")
        return

    # Read the markdown file.
    with open(md_file_path, "r") as file:
        md_content = file.read()

    # Convert markdown content into a Document.
    doc = Document(page_content=md_content, metadata={"source": md_file_name})

    # Initialize Chroma with embedding function.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Add the document to the database.
    db.add_documents([doc])
    db.persist()
    print(f"Document '{md_file_name}' added to the Chroma database.")
