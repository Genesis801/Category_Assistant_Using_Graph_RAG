from langchain_classic.chains import RetrievalQA

def build_rag_chain(llm, retriever):

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain
