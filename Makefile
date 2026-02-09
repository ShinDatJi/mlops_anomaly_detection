init:
	cp default.env .env

# data
init-data:
	cp apps/data/default.env apps/data/.env
build-data:
	docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./ build data
dev-data:
	docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./ run --build data
connect-data:
	docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./ run --build data sh
run-data:
	docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./ up
clean-data:
	docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./ down

# modeling
init-modeling:
	cp apps/modeling/default.env apps/modeling/.env
build-modeling:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ build
dev-modeling-preprocessing:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --build --no-deps modeling-preprocessing
dev-modeling-training:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --build --no-deps modeling-training
dev-modeling-evaluation:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --build --no-deps modeling-evaluation
connect-modeling-preprocessing:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --build --no-deps modeling-preprocessing sh
connect-modeling-training:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --build --no-deps modeling-training sh
connect-modeling-evaluation:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --build --no-deps modeling-evaluation sh
run-modeling-preprocessing:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --rm --no-deps modeling-preprocessing
run-modeling-training:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --rm --no-deps modeling-training
run-modeling-evaluation:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ run --rm --no-deps modeling-evaluation
run-modeling:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ up
clean-modeling:
	docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./ down

#all
init-all:
	init
	init-data
	init-modeling
build-all:
	build-data
	build-modeling
run-all:
	run-data
	run-modeling
clean-all:
	clean-data
	clean-modeling
	