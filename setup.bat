@echo off
echo Installing Kindle Review Sentiment Analyzer...
echo.

echo Installing Python packages...
python -m pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Installation completed successfully!
    echo.
    echo To run the app, use: streamlit run streamlit_app.py
    echo.
    pause
) else (
    echo.
    echo ❌ Installation failed. Please check your Python installation.
    echo Make sure Python and pip are installed and accessible from command line.
    echo.
    pause
)
