from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def rag(query, question):
    arxiv_docs = ArxivLoader(query=query, load_max_docs=1).load()

    print(arxiv_docs[0].metadata['Title'])

    pdf_data = []
    for doc in arxiv_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([doc.page_content])
        pdf_data.append(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    db = Chroma.from_documents(pdf_data[0], embeddings)

    llm = ChatOpenAI(model='gpt-3.5-turbo',
                     temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=db.as_retriever())
    result = qa({"query": question})
    return result


query = "lightweight transformer for language tasks"
question = "how many and which benchmark datasets and tasks were compared for light weight transformer?"
output = rag(query, question)
print(output)
