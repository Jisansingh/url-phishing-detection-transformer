import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "final_urls_10k_cleaned.txt"

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return None
    
    # robustly load data by splitting on the LAST comma
    data = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on the last comma only
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    data.append({'url': parts[0], 'label': parts[1]})
                else:
                    # Should not happen given our preprocessing, but good to be safe
                    print(f"Skipping malformed line: {line}")
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def print_distribution(name, df):
    print(f"\n--- {name} Set ---")
    print(f"Shape: {df.shape}")
    print("Class Distribution:")
    print(df['label'].value_counts(normalize=True))
    print(f"Counts:\n{df['label'].value_counts()}")

def split_dataset():
    df = load_data()
    if df is None:
        return

    print(f"Loaded dataset with {len(df)} records.")
    
    X = df['url']
    y = df['label']

    # First split: 70% Train, 30% Temp (which will be Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Second split: Split the 30% Temp into equal parts (15% Val, 15% Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Reconstruct DataFrames
    train_df = pd.DataFrame({'url': X_train, 'label': y_train})
    val_df = pd.DataFrame({'url': X_val, 'label': y_val})
    test_df = pd.DataFrame({'url': X_test, 'label': y_test})

    # Verification
    print_distribution("Training", train_df)
    print_distribution("Validation", val_df)
    print_distribution("Test", test_df)

    # Save to CSV - We will use pandas' standard CSV writer which handles quoting/escaping correctly
    # detailed: url,label
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("\nSuccessfully saved 'train.csv', 'val.csv', and 'test.csv'.")

if __name__ == "__main__":
    split_dataset()
