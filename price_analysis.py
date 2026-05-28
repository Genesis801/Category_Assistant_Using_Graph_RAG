def median_price_by_category(df):

    result = (
        df.groupby("main_category")["discount_price"]
        .median()
        .sort_values(ascending=False)
    )

    return result

def median_price_by_subcategory(df):

    result = (
        df.groupby("sub_category")["discount_price"]
        .median()
        .sort_values(ascending=False)
    )

    return result

def most_expensive_categories(df):


    result = (
        df.groupby("main_category")["discount_price"]
        .mean()
        .sort_values(ascending=False)
    )

    return result


def least_expensive_categories(df):

    result = (
        df.groupby("main_category")["discount_price"]
        .mean()
        .sort_values(ascending=True)
    )

    return result

def category_discount_ranking(df):

    result = (
        df.groupby("main_category")["discount_percentage"]
        .mean()
        .sort_values(ascending=False)
    )

    return result
