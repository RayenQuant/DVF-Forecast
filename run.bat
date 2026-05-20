@echo off
:: ============================================================
:: run.bat — Equivalent Windows du Makefile pour dvf-forecast
:: Usage : run.bat <commande>
:: ============================================================

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="download" goto download
if "%1"=="clean" goto clean
if "%1"=="features" goto features
if "%1"=="train" goto train
if "%1"=="evaluate" goto evaluate
if "%1"=="app" goto app
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="all" goto all

echo Commande inconnue : %1
goto help

:help
echo.
echo  Usage : run.bat ^<commande^>
echo.
echo  Commandes disponibles :
echo    install    Installer les dependances Python
echo    download   Telecharger les donnees DVF (annee 2024 par defaut)
echo    clean      Nettoyer et filtrer les transactions
echo    features   Construire les variables (feature engineering)
echo    train      Entrainer le modele XGBoost
echo    evaluate   Evaluer le modele (RMSE, R2, MAPE)
echo    app        Lancer le dashboard Streamlit
echo    test       Executer les tests unitaires
echo    lint       Verifier le style avec ruff
echo    format     Reformater avec black
echo    all        Pipeline complet : download -^> clean -^> features -^> train -^> evaluate
echo.
goto end

:install
echo [1/1] Installation des dependances...
pip install -r requirements.txt
goto end

:download
echo [1/4] Telechargement multi-annees (2020-2024)...
python -m src.data_loader
if errorlevel 1 goto error

:clean
echo [1/1] Nettoyage des donnees...
python -m src.cleaning
goto end

:features
echo [1/1] Feature engineering...
python -m src.features
goto end

:train
echo [1/1] Entrainement XGBoost (5-fold CV)...
python -m src.train --cv 5
goto end

:evaluate
echo [1/1] Evaluation du modele...
python -m src.evaluate
goto end

:app
echo [1/1] Lancement du dashboard Streamlit...
echo      Ouvrez http://localhost:8501 dans votre navigateur
streamlit run streamlit_app/app.py
goto end

:test
echo [1/1] Tests unitaires...
pytest tests/ -v
goto end

:lint
echo [1/1] Lint ruff...
ruff check src/ streamlit_app/ tests/
goto end

:format
echo [1/1] Formatage black...
black src/ streamlit_app/ tests/
goto end

:all
echo === Pipeline complet DVF Forecast ===
echo.
echo [1/4] Telechargement multi-annees (2020-2024)...
python -m src.data_loader
if errorlevel 1 goto error

echo [2/4] Nettoyage...
python -m src.cleaning
if errorlevel 1 goto error

echo [3/4] Feature engineering...
python -m src.features
if errorlevel 1 goto error

echo [4/4] Entrainement + evaluation...
python -m src.train --cv 5
if errorlevel 1 goto error

python -m src.evaluate
if errorlevel 1 goto error

echo.
echo === Pipeline termine avec succes ! ===
echo Pour lancer le dashboard : run.bat app
goto end

:error
echo.
echo ERREUR : le pipeline s'est arrete a cette etape.
echo Consultez le message ci-dessus pour details.
exit /b 1

:end
