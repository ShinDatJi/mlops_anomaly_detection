run-data-load:
	uv run --env-file src/data/default.env src/data/load_data.py

run-data-analyze:
	uv run --env-file src/data/default.env src/data/analyze_data.py

run-data-visualize:
	uv run --env-file src/data/default.env src/data/visualize_data.py

run-data:
	run-data-load
	run-data-analyze
	run-data-visualize

run-modeling-preprocessing:
	uv run --env-file src/modeling/default.env src/modeling/preprocessing/train_test_split.py
	uv run --env-file src/modeling/default.env src/modeling/preprocessing/preprocess_data.py

run-modeling-training:
	uv run --env-file src/modeling/default.env src/modeling/training/train_model.py

run-modeling-evaluation:
	uv run --env-file src/modeling/default.env src/modeling/evaluation/evaluate_model.py

run-modeling:
	run-modeling-preprocessing
	run-modeling-training
	run-modeling-evaluation

run-all:
	run-data
	run-modeling
	