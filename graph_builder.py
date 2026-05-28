import networkx as nx

def build_graph(df):

    G = nx.Graph()

    for _, row in df.iterrows():

        category = row["main_category"]
        sub_category = row["sub_category"]
        product = row["name"]

        G.add_node(category, type="category")

        G.add_node(sub_category, type="sub_category")

        G.add_node(
            product,
            type="product",
            price=row["discount_price"],
            actual_price=row["actual_price"],
            ratings=row["ratings"],
            no_of_ratings=row["no_of_ratings"]
        )

        G.add_edge(category, sub_category)

        G.add_edge(sub_category, product)

    return G
