import pandas as pd

def train_test_split(df, unique_id="unique_id", test_window=24*60):
    """
    Parameters
    ------
    df : original data set
    unique_id : column name of the segments unique id
    test_window : test size in days
    
    Returns
    ------
    df_train : train data set
    df_test : test data set
    """
    for i in df[unique_id].unique():
        if i == df[unique_id].unique()[0]:
            df_test = df[df[unique_id]==i][-test_window:]
            continue
        df_test = pd.concat([df_test, df[df[unique_id]==i][-test_window:]])
    df_test.sort_index(inplace=True)
    
    train_idx = [idx for idx in df.index if idx not in df_test.index]
    df_train = df.loc[train_idx]
    
    return df_train, df_test


class columnDropperTransformer:
    """custom scikit-learn's transformer object to drop 'unique_id' 'ds' columns from X when calling fit and predict methods"""
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self
