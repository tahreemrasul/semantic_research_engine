import ast
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


def get_arxiv_results(query):
    docs = ArxivLoader(query=query, load_max_docs=3).load()
    return docs


def ingest_docs(arxiv_docs, db):
    for doc in arxiv_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([doc.page_content])
        db.add_documents(texts)


def get_qa_chain_answers(llm, db, question):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    result = qa({"query": question})
    return result['result']


def display_arxiv_results(arxiv_docs):
    arxiv_papers = [f"{doc.metadata}" for doc in arxiv_docs]
    data_dicts = [ast.literal_eval(entry) for entry in arxiv_papers]
    for entry in data_dicts:
        print(f"Published: {entry['Published']}")
        print(f"Title: {entry['Title']}")
        print(f"Authors: {entry['Authors']}")
        print(f"Summary: {entry['Summary'][:100]}...")
        print("\n---\n")
