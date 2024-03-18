import ast
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()


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


def main():
    query = "vision transformers"
    arxiv_docs = get_arxiv_results(query)
    # Prepare arXiv results for display
    arxiv_papers = [f"{doc.metadata}" for doc in arxiv_docs]
    data_dicts = [ast.literal_eval(entry) for entry in arxiv_papers]
    for entry in data_dicts:
        print(f"Published: {entry['Published']}")
        print(f"Title: {entry['Title']}")
        print(f"Authors: {entry['Authors']}")
        print(f"Summary: {entry['Summary'][:100]}...")  # Display the first 100 characters of summary for brevity
        print("\n---\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    db = Chroma(persist_directory=query, embedding_function=embeddings)
    ingest_docs(arxiv_docs, db)

    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                 temperature=0)
    question = "how many and which benchmark datasets and tasks were compared for light weight transformer? what was the performance of the light weight transformer?"

    result = get_qa_chain_answers(llm, db, question)
    print(result)


main()

