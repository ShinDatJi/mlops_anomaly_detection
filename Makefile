run-all:
	uv run src/data/load_data.py
	uv run src/data/analyze_data.py
	uv run src/data/visualize_data.py
	uv run src/data/train_test_split.py
	uv run src/preprocessing/preprocess_data.py
	uv run src/training/train_model.py
	uv run src/evaluation/evaluate_model.py
