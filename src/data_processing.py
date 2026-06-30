import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def fill_missing_values(df):
    # missing vals:
    for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence']:
        df[col] = df[col].fillna("None")
    # Categorical Missing Data
    df['MasVnrType'] = df['MasVnrType'].fillna("None")
    df['FireplaceQu'] = df['FireplaceQu'].fillna("None")
    # Numerical Missing Data
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    none_fill = [
        'GarageQual', 'GarageFinish', 'GarageType', 'GarageCond',
        'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure'
    ]
    for col in none_fill:
        df[col] = df[col].fillna("None")
    # Numeric garage
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    # Numeric area feature
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    # Small categorical
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    return df
def create_features(df):
    # feature engineering
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    # encoding
    return df
def preprocess(df):
    df=fill_missing_values(df)
    df = create_features(df)
    return df

def load_data(path:str):
    df=pd.read_csv(path)
    X=df.drop(columns='SalePrice')
    y=df['SalePrice']
    return X,y

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object','string']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor