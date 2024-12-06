import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore.document import Document


def read_and_set_env_vars():
    with open(".env", "r") as file:
        for line in file.readlines():
            key, value = line.split("=")
            os.environ[key] = value


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as _file:
        return "".join(_file.readlines())


if __name__ == '__main__':
    # Read and set environment variables (API keys for OpenAI)
    read_and_set_env_vars()

    cur_file_path = os.path.abspath(__file__)
    example_file = read_file(cur_file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Pre-process the file
    docs = []
    chunks = splitter.split_text(example_file)
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "meta": {
                "chunk_id": i
            }
        })

    docs_objs = [Document(page_content=doc["text"], metadata=doc["meta"]) for doc in docs]

    # Create a vector store
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(
        documents=docs_objs,
        embedding=embeddings,
        collection_name="codebase"
    )

    # Create a retrieval chain
    llm = OpenAI(temperature=0, model_name="gpt-4")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = "What is the purpose of this file?"
    answer = qa_chain.run(query)
    print(answer)
