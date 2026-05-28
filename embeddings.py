from langchain_core.documents import Document

def create_documents(df):

    documents = []

    for _, row in df.iterrows():

        text = f'''
        Product Name: {row["name"]}
        Category: {row["main_category"]}
        Sub Category: {row["sub_category"]}
        Discount Price: {row["discount_price"]}
        Actual Price: {row["actual_price"]}
        Ratings: {row["ratings"]}
        Number of Ratings: {row["no_of_ratings"]}
        '''

        doc = Document(
            page_content=text,
            metadata={
                "category": row["main_category"],
                "sub_category": row["sub_category"]
            }
        )

        documents.append(doc)

    return documents
