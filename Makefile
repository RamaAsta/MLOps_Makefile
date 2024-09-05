DATA_PREP_SCRIPT = scripts/data_prep.py
TRAIN_MODEL_SCRIPT = scripts/train_model.py
EVALUATE_MODEL_SCRIPT = scripts/evaluate_model.py
DEPLOY_MODEL_SCRIPT = scripts/deploy_model.py
REQUIREMENTS_FILE = requirements.txt

all: install data train evaluate deploy

install:
	@echo "Installing dependencies"
	pip install -r $(REQUIREMENTS_FILE)

data:
	@echo "Preparing data..."
	python $(DATA_PREP_SCRIPT)

train:
	@echo "Training model..."
	python $(TRAIN_MODEL_SCRIPT)

evaluate:
	@echo "Evaluating model..."
	python $(EVALUATE_MODEL_SCRIPT)

deploy:
	@echo "Deploying model..."
	python $(DEPLOY_MODEL_SCRIPT)

.PHONY: all install data train evaluate deploy clean