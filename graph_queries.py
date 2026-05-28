import networkx as nx

def get_products_in_category(G, category):

    products = []

    subcategories = list(G.neighbors(category))

    for sub in subcategories:

        sub_products = list(G.neighbors(sub))

        products.extend(sub_products)

    return products
