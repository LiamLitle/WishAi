@echo off
setlocal
set ROOT=%~dp0

:: Detection automatique du virtualenv s'il existe
set PYTHON_CMD=python
if exist "%ROOT%.venv\Scripts\python.exe" (
    set PYTHON_CMD="%ROOT%.venv\Scripts\python.exe"
)

if "%1"=="" goto help
if "%1"=="go"     ( %PYTHON_CMD% "%ROOT%go.py"              %2 %3 %4 & goto end )
if "%1"=="chat"   ( %PYTHON_CMD% "%ROOT%scripts\chat.py"    %2 %3 %4 & goto end )
if "%1"=="quick"  ( %PYTHON_CMD% "%ROOT%scripts\quick.py"   %2 %3 %4 & goto end )
if "%1"=="config" ( %PYTHON_CMD% "%ROOT%scripts\config.py"  %2 %3 %4 & goto end )
if "%1"=="serve"  ( %PYTHON_CMD% "%ROOT%scripts\serve.py"   %2 %3 %4 & goto end )
if "%1"=="visual" ( %PYTHON_CMD% -m http.server 8080 --directory "%ROOT%" & goto end )
if "%1"=="token"  ( %PYTHON_CMD% "%ROOT%scripts\reset_token.py"    %2 %3 %4 & goto end )
if "%1"=="repair" ( %PYTHON_CMD% "%ROOT%src\bot_logger.py"   %2 %3 %4 & goto end )
if "%1"=="logs"   ( %PYTHON_CMD% "%ROOT%scripts\logs.py"     %2 %3 %4 & goto end )
if "%1"=="export" ( %PYTHON_CMD% "%ROOT%scripts\export.py"   %2 %3 %4 & goto end )

echo  Commande inconnue : %1
goto help

:help
echo.
echo   WishAI — Raccourcis
echo   -------------------
echo   ./wish go        Lance le menu principal
echo   ./wish chat      Interface de chat
echo   ./wish quick     Entrainement rapide (zero config)
echo   ./wish config    Gestion des modeles et donnees
echo   ./wish export    Exporter un modele (GGUF / ONNX)
echo   ./wish serve     Serveur local (dashboard / library)
echo   ./wish visual    Visualiseur d'embeddings (port 8080)
echo   ./wish token     Reinitialise et reentrainer le tokenizer BPE
echo   ./wish repair    Verifie et repare les dependances cassees
echo   ./wish logs      Affiche les derniers logs du bot
echo.

:end
endlocal
