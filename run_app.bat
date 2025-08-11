@echo off
cd /d "%~dp0"
echo Starting Kindle Review Sentiment Analyzer...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the app
echo.
.conda\python.exe -m streamlit run streamlit_app.py --server.port 8501
pause
