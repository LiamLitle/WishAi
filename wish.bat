@echo off
setlocal
set ROOT=%~dp0

if "%1"=="" goto help
if "%1"=="go"     ( python "%ROOT%go.py"              %2 %3 %4 & goto end )
if "%1"=="chat"   ( python "%ROOT%scripts\chat.py"    %2 %3 %4 & goto end )
if "%1"=="quick"  ( python "%ROOT%scripts\quick.py"   %2 %3 %4 & goto end )
if "%1"=="config" ( python "%ROOT%scripts\config.py"  %2 %3 %4 & goto end )
if "%1"=="serve"  ( python "%ROOT%scripts\serve.py"   %2 %3 %4 & goto end )
if "%1"=="visual"  ( python -m http.server 8080 --directory "%ROOT%" & goto end )
if "%1"=="token"   ( python "%ROOT%scripts\token.py"    %2 %3 %4 & goto end )
if "%1"=="repair"  ( python "%ROOT%src\bot_logger.py"   %2 %3 %4 & goto end )
if "%1"=="logs"    ( python "%ROOT%scripts\logs.py"     %2 %3 %4 & goto end )

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
echo   ./wish serve     Serveur local (dashboard / library)
echo   ./wish visual    Visualiseur d'embeddings (port 8080)
echo   ./wish token     Reinitialise et reentrainer le tokenizer BPE
echo   ./wish repair    Verifie et repare les dependances cassees
echo   ./wish logs      Affiche les derniers logs du bot
echo.

:end
endlocal
