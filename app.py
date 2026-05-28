from load_csv import load_csv
from preprocess import preprocess_dataframe
from graph_builder import build_graph

from embeddings import create_documents
from vector_store import build_vector_store
from retriever import load_retriever
from llm import load_llm

from price_analysis import *

from router import route_query
from conversational_chain import build_rag_chain

df = load_csv("Amazon_data_category_wise.csv")

df = preprocess_dataframe(df)
print("Data Loaded and Preprocessed Successfully!")

G = build_graph(df)
print("Graph Built Successfully!")

documents = create_documents(df)
print("Documents Created Successfully!")

vector_db = build_vector_store(documents)
print("Vector Store Built Successfully!")

retriever = load_retriever()
print("Retriever Loaded Successfully!")

llm = load_llm()
print("LLM Loaded Successfully!")

rag_chain = build_rag_chain(llm, retriever)
print("RAG Chain Built Successfully!")

while True:

    query = input("\nAsk Question: ")

    intent = route_query(query)

    if intent == "median":

        print(median_price_by_category(df))

    elif intent == "most_expensive":

        print(most_expensive_categories(df))

    elif intent == "least_expensive":

        print(least_expensive_categories(df))

    elif intent == "discount_ranking":

        print(category_discount_ranking(df))

    else:

        response = rag_chain.invoke({"query": query})

        print(response["result"])
