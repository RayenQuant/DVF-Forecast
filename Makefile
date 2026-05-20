.PHONY: help install download clean features train evaluate app test lint format all

help:
	@echo "Usage:"
	@echo "  make install    - Installer les dépendances Python"
	@echo "  make download   - Télécharger les données DVF"
	@echo "  make clean      - Nettoyer & filtrer les transactions"
	@echo "  make features   - Construire les variables (feature engineering)"
	@echo "  make train      - Entraîner le modèle XGBoost"
	@echo "  make evaluate   - Évaluer le modèle (RMSE, R², MAPE)"
	@echo "  make app        - Lancer le dashboard Streamlit"
	@echo "  make test       - Exécuter les tests unitaires"
	@echo "  make lint       - Vérifier le style avec ruff"
	@echo "  make format     - Reformater avec black"
	@echo "  make all        - Pipeline complet : download -> clean -> features -> train -> evaluate"

install:
	pip install -r requirements.txt

download:
	python -m src.data_loader --year 2024

clean:
	python -m src.cleaning

features:
	python -m src.features

train:
	python -m src.train --cv 5

evaluate:
	python -m src.evaluate

app:
	streamlit run streamlit_app/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ streamlit_app/ tests/

format:
	black src/ streamlit_app/ tests/

all: download clean features train evaluate
