from langchain_community.document_loaders import PyPDFLoader
import yaml

with open("params.yaml",'r') as f:
    config=yaml.safe_load(f)
# loading data
class load():

    def __init__(self,filename):
        loader=PyPDFLoader(filename)
        self.doc=loader.load()
    def print_doc(self,i):
        print(self.doc[i].page_content)
    def ret_doc(self):
        return self.doc

pdf_document=load(config["data_ingestion"]["pdf_file"])
Document=pdf_document.ret_doc()
# use this Document
