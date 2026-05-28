def route_query(query):
    
    query = query.lower()

    if "median" in query:
        return "median"

    elif "most expensive" in query:
        return "most_expensive"

    elif "least expensive" in query:
        return "least_expensive"

    elif "discount" in query and "rank" in query:
        return "discount_ranking"

    else:
        return "rag"
