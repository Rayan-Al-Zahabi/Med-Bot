from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import faiss
from typing import Optional


DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        # model = "llama-2-7b-chat.ggmlv3.q8_0_2.bin",
        model = "llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == ("admin", "admin"):
    return cl.AppUser(username="admin", role="ADMIN", provider="credentials")
  else:
    return None

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    print(f"Chain: {chain}") 
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Med-Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)




@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    
    # Assuming res is a dictionary with keys "result_llm" and "result_qa"
    result_llm = res.get("result_llm", "")
    result_qa = res.get("result_qa", "")

    # Check if both LLM and QA chain provided results
    if result_llm and result_qa:
        # Choose one source over the other, e.g., prioritize QA chain
        answer = result_qa
    else:
        # If one of them is missing, use the available result
        answer = result_llm or result_qa

    await cl.Message(content=answer).send()

