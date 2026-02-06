# Data default environment

for cat in "bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper"
do
    echo $cat
    export CATEGORY="$cat"

    export DATA_RAW_DIR="./data/raw/$CATEGORY/"

    export DATA_PROCESSED_DIR="./data/processed/$CATEGORY/"
    export DATA_RAW_FILE="./data/processed/$CATEGORY/raw.csv"
    export DATA_CLEAN_FILE="./data/processed/$CATEGORY/clean.csv"
    export DATA_STATS_FILE="./data/processed/$CATEGORY/stats.csv"

    export REPORTS_DIR="./reports/data/$CATEGORY/"

    uv run src/data/load_data.py
    uv run src/data/analyze_data.py
    uv run src/data/visualize_data.py

    export CONFIG_FILE="./models/$CATEGORY/config.json"

    export DATA_DIR="./data/processed/$CATEGORY/"
    export DATA_TRAIN_DIR="./data/processed/$CATEGORY/train/"
    export DATA_TEST_DIR="./data/processed/$CATEGORY/test/"
    export DATA_TEST_PATCHING_DIR="./data/processed/$CATEGORY/test_patching/"
    export DATA_CLEAN_FILE="./data/processed/$CATEGORY/clean.csv"
    export DATA_TRAIN_FILE="./data/processed/$CATEGORY/train.csv"
    export DATA_TEST_FILE="./data/processed/$CATEGORY/test.csv"

    export REPORTS_DIR="./reports/modeling/$CATEGORY/"

    export PREPROCESSING_REPORT_FILE="./reports/modeling/$CATEGORY/preprocessing_report.json"
    export TRAINING_REPORT_FILE="./reports/modeling/$CATEGORY/training_report.json"
    export EVALUATION_REPORT_FILE="./reports/modeling/$CATEGORY/evaluation_report.json"

    export MODEL_FILE="./reports/modeling/$CATEGORY/model.keras"

    uv run src/modeling/preprocessing/preprocess_data.py
	uv run src/modeling/training/train_model.py
	uv run src/modeling/evaluation/evaluate_model.py
done
