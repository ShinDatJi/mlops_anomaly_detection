import os
from load_data import load_data
from analyze_data import analyze_data
from visualize_data import visualize_data

def clean_database(df):
    df = df.copy()
    print("> remove stats columns")
    df = df[["subset", "anomaly", "img_size", "grayscale", "anomaly_coverage", "file"]]
    return df

def main():
    data_raw_path = os.environ["DATA_RAW_PATH"]
    data_path = os.environ["DATA_PATH"]
    clean_db_file = os.path.join(data_path, os.environ["DATA_CLEAN_DB"])
    reports_path = os.environ["REPORTS_PATH"]

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(reports_path, exist_ok=True)

    df = load_data(data_raw_path)
    df = analyze_data(df, reports_path)
    visualize_data(df, reports_path)
    df = clean_database(df)

    print("> save database")
    print(clean_db_file)
    df.to_csv(clean_db_file)
    
if __name__ == "__main__":
    main()
