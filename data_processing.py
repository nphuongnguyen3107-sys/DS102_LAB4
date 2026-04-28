import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_dir='data', test_size=0.2, random_state=42,
                             handle_missing='drop', verbose=True):

    # Kiểm tra file tồn tại
    red_wine_path = os.path.join(data_dir, 'winequality-red.csv')
    white_wine_path = os.path.join(data_dir, 'winequality-white.csv')

    for path in [red_wine_path, white_wine_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Đọc dữ liệu
    try:
        red_wine = pd.read_csv(red_wine_path, sep=';')
        white_wine = pd.read_csv(white_wine_path, sep=';')
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {e}")

    if verbose:
        print(f"Read successfully: {len(red_wine)} red wine samples, {len(white_wine)} white wine samples")

    # Kiểm tra cột 'quality' tồn tại trong cả hai dataset
    for df, name in [(red_wine, 'red_wine'), (white_wine, 'white_wine')]:
        if 'quality' not in df.columns:
            raise ValueError(f"Missing 'quality' column in {name}")

    # Gộp dữ liệu lại với nhau
    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

    if verbose:
        print(f"Total samples: {len(wine_data)}")
        print(f"Number of features: {len(wine_data.columns) - 1}")
        print(f"Quality distribution:")
        print(wine_data['quality'].value_counts().sort_index())

    # Xử lý giá trị thiếu
    missing_total = wine_data.isnull().sum().sum()
    if missing_total > 0:
        print(f"Found {missing_total} missing values")

        if handle_missing == 'drop':
            wine_data = wine_data.dropna()
            print(f"Dropped {missing_total} rows with missing values")
        elif handle_missing == 'fill':
            # Điền giá trị thiếu bằng median của cột đó
            for col in wine_data.columns:
                if col != 'quality':
                    median_val = wine_data[col].median()
                    wine_data[col] = wine_data[col].fillna(median_val)
            print(f"Filled missing values with median")
        else:
            raise ValueError("handle_missing must be 'drop' or 'fill'")

    # Tách X và y
    X = wine_data.drop('quality', axis=1).values
    y = wine_data['quality'].values

    # Kiểm tra sau xử lý có còn dữ liệu không
    if len(X) == 0:
        raise ValueError("No data left after processing!")

    if verbose:
        print(f"\nShape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")

    # Chia dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"\nData split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test
