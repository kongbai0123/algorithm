@echo off
setlocal enabledelayedexpansion

:: Change to the directory where the script is located
cd /d "%~dp0"

echo ================================
echo [Git Push Script]
echo Working Dir: %cd%
echo ================================
echo.

:: Check if Git command exists
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git command not found.
    goto END
)

:: Check if inside a Git repository
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a Git repository.
    goto END
)

echo Checking status...
git status --short
echo.

echo Adding changes...
git add .
if errorlevel 1 (
    echo [ERROR] git add failed.
    goto END
)

set "commit_msg="
set /p "commit_msg=Enter commit message (Press Enter for default): "

if "%commit_msg%"=="" (
    set "commit_msg=Auto commit at %date% %time%"
)

echo.
echo Committing...
git commit -m "%commit_msg%"

if errorlevel 1 (
    echo.
    echo [WARN] git commit failed or no changes.
    echo Attempting to push anyway...
)

echo.
echo Pushing to origin main...
git push origin main

if errorlevel 1 (
    echo.
    echo [ERROR] git push failed.
    echo Common fixes:
    echo 1. Check your network or permissions
    echo 2. git pull origin main (if remote has changes)
    goto END
)

echo.
echo [SUCCESS] Done!

:END
echo.
pause
endlocal