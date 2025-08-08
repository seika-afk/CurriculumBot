from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import yaml


with open("config.yaml",'r') as f:
    config=yaml.safe_load(f)




embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
Vector_Stores = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# retrieval:
retriever=Vector_Stores.as_retriever(
        search_type="similarity",
        search_kwargs={"k":config["retrieving"]["num_of_retrieved_docs"]}

        )


