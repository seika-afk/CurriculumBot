from Data_ingestion import Document



from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def get_chunks(Document):
    chunk_list = []
    for i in range(len(Document)):
        chunk_list.append(Document[i].page_content)

    text = "\n".join(chunk_list)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "header1"), ("##", "header2"), ("###", "header3")])
    chunks_ = splitter.split_text(text)
    return chunks_

def vectWork(chunks_):
    #using free embedding
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_store=FAISS.from_documents(chunks_,embeddings)

    return vector_store


chunks_=get_chunks(Document)
Vector_Store=vectWork(chunks_)



Vector_Store.save_local("faiss_index")

