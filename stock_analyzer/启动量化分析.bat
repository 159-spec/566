@echo off
chcp 65001 >nul
title A股量化回测分析
cd /d "%~dp0"

echo.
echo ========================================
echo   A股量化回测分析 - 桌面版
echo   策略: 优化v2 (胜率81.0%)
echo ========================================
echo.

REM 优先使用 conda stock 环境的 python
set "PYEXE=C:\Users\cks\miniconda3\envs\stock\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

"%PYEXE%" -c "import webview" >nul 2>nul
if %errorlevel%==0 (
    echo [方式1] 使用 pywebview 打开原生窗口...
    "%PYEXE%" "%~dp0desktop_app.py"
    goto :end
) else (
    echo [提示] 未安装 pywebview，尝试安装...
    "%PYEXE%" -m pip install pywebview -q >nul 2>nul
    if %errorlevel%==0 (
        "%PYEXE%" "%~dp0desktop_app.py"
        goto :end
    ) else (
        echo [提示] pywebview 安装失败，将使用浏览器打开...
    )
)

REM 回退：用默认浏览器打开
echo [方式2] 使用默认浏览器打开...
start "" "%~dp0index.html"

:end
echo.
echo 程序已启动，如窗口未自动打开，请手动双击 index.html
pause
