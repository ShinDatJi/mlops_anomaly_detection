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

run-modeling-preprocess:
	uv run --env-file src/modeling/default.env src/modeling/preprocessing/preprocess_data.py

run-modeling-train:
	uv run --env-file src/modeling/default.env src/modeling/training/train_model.py

run-modeling-evaluate:
	uv run --env-file src/modeling/default.env src/modeling/evaluation/evaluate_model.py

run-modeling:
	run-modeling-preprocess
	run-modeling-train
	run-modeling-evaluate

run-all:
	run-data
	run-modeling
	