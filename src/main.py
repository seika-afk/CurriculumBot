#local imports
from Retriever import retriever

#not local imports
import logging
import os
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()


#Logging configuration for errors:
class log_it():
    def __init__(self):
        log_dir='logs'
        os.makedirs(log_dir,exist_ok=True)
        
        logger=logging.getLogger('Errors')
        self.logger=logger
        logger.setLevel('DEBUG')
        #printing to console
        console_handler=logging.StreamHandler()
        console_handler.setLevel('DEBUG')
        #file printing
        log_file_path=os.path.join(log_dir,'data_ingestion.log')
        file_handler=logging.FileHandler(log_file_path)
        file_handler.setLevel('DEBUG')

        #formatting the way to set the logs
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)



        logger.addHandler(console_handler)
        logger.addHandler(file_handler)


logger=log_it().logger


client = InferenceClient(
    provider="novita",
    api_key=os.getenv("API_KEY")
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant for students at the ECE department of NIT Hamirpur.

You are given the official curriculum documents, syllabus, and academic information for all years and semesters (1st to 4th year).

Your job is to answer questions based only on the relevant academic year and subject mentioned in the question or context.

Make sure to tell only necessary things,Make sure ,if asked for syllabus or details, show all chapters/units.

If the relevant information is present in the context, answer clearly and accurately.

If it is not found or is ambiguous, respond with:
"Please Provide more context for this."

---
Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

x = ""

def query(x):

    try:
        retrieved_docs = retriever.invoke(x)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        final_prompt = prompt.invoke({"context": context_text, "question": x})

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {
                    "role": "user",
                    "content": str(final_prompt)
                }
            ],
            max_tokens=400
        )

        text = completion.choices[0].message.content

        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return text

    except Exception as e :
        print(e)
        logger.error(f"Unexpected error occured.")
        return "Error Occured."
