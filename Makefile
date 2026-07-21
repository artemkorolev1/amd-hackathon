.PHONY: build run local-test push verify deploy evaluate staging-test staging-run staging-verify smoke-test-runner smoke-test-evaluate smoke-test-deploy smoke-all cpu-build cpu-push cpu-run

IMAGE_NAME := amd-agent

build:
	docker build -t $(IMAGE_NAME) -f Dockerfile .

# Quick test: mount /input /output and pipe tasks
run:
	docker run --rm -i \
		-v /input:/input:ro \
		-v /output:/output \
		$(IMAGE_NAME) < tasks.txt

# Debug shell inside the built image
shell:
	docker run --rm -it --entrypoint /bin/bash $(IMAGE_NAME)

# Build without cache (fresh)
rebuild:
	docker build --no-cache -t $(IMAGE_NAME) -f Dockerfile .

# Estimate image size
size:
	docker images $(IMAGE_NAME) --format "{{.Size}}"

# Local test: run agent directly (requires llama.cpp server running)
local-test:
	PYTHONPATH=. python3 -m agent.main < tasks.txt

# Create a test tasks file
test-tasks:
	@echo "What is 42 * 13?" > tasks.txt
	@echo "Write a Python function to check if a string is a palindrome." >> tasks.txt
	@echo "Summarize: The quick brown fox jumps over the lazy dog." >> tasks.txt
	@echo "Classify the sentiment of this review: The product was okay." >> tasks.txt
	@echo "Translate 'Hello world' to French." >> tasks.txt

# --- Deployment targets (runner/deploy.py) ---

# Build Docker image (auto-versioned)
build-image:
	python -m runner.deploy --build-only

# Push to GHCR
push:
	python -m runner.deploy --push

# Full deploy: build + push + verify + document
deploy:
	python -m runner.deploy --push --verify

# Evaluate results against ground truth
evaluate:
	python -m runner.evaluate --results eval_results/results.json --gold input/dev_40.json --output eval_results/report.xlsx

# Run batch (parallel inference)
run-batch:
	python3 -m runner.batch_runner --input input/tasks.json --output results.json --workers 2

# --- Staging targets ---

staging-test:
	python3 staging/test_judge.py

staging-run:
	python3 -m staging.entrypoint $(TASKS)

staging-verify:
	python3 -c "from staging import ReadyConfig, ReadyMonitor, ReadyJudge, ReadyQueue; print('staging imports OK')"

# --- Integration smoke tests ---

smoke-test-runner:
	python3 -m runner.batch_runner --help

smoke-test-evaluate:
	python3 -m runner.evaluate --help

smoke-test-deploy:
	python3 -m runner.deploy --help

# --- CPU submission targets ---

CPU_IMAGE := ghcr.io/artemkorolev1/amd-hackathon-submit:cpu

cpu-build:
	docker buildx build --platform linux/amd64 -t $(CPU_IMAGE) --load -f Dockerfile.cpu .

cpu-push: cpu-build
	docker tag $(CPU_IMAGE) ghcr.io/artemkorolev1/amd-hackathon-submit:cpu
	docker push ghcr.io/artemkorolev1/amd-hackathon-submit:cpu

cpu-run:
	docker run --rm \
		-v /input:/input:ro \
		-v /output:/output \
		$(CPU_IMAGE)

# --- Full integration smoke ---

smoke-all: staging-verify smoke-test-runner smoke-test-evaluate smoke-test-deploy
	@echo "All modules pass import smoke tests"
