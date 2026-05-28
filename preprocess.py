import pandas as pd
import numpy as np

def clean_price(price):

    if pd.isna(price):
        return np.nan

    price = str(price)

    price = (
        price.replace("₹", "")
            .replace(",", "")
            .replace("â‚¹", "")
            .strip()
    )

    try:
        return float(price)
    except:
        return np.nan

def preprocess_dataframe(df):

    df["discount_price"] = df["discount_price"].apply(clean_price)

    df["actual_price"] = df["actual_price"].apply(clean_price)

    df["discount_percentage"] = (
        (df["actual_price"] - df["discount_price"])
        / df["actual_price"]
    ) * 100

    return df
