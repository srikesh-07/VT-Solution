import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


def strat_shuffle_split(
    df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
):
    df_split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=random_state
    )
    X = df.index
    y = df["Category"]
    train_indices, test_indices = list(df_split.split(X, y))[0]
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[
        test_indices
    ].reset_index(drop=True)


def replace_nan(df: pd.DataFrame):
    for index in tqdm(df.index):
        t = df.iloc[index]
        df.iloc[index, : t["len"] + 2].fillna("default")
    return df
