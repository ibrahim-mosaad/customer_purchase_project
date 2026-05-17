import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # remove duplicates
    df = df.drop_duplicates()

    # handle missing values (لو موجودة)
    df = df.dropna()

    return df