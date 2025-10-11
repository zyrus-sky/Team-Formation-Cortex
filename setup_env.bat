@echo off
setlocal

REM ---- Config ----
set VENV_DIR=.venv
set PYTHON=py -3
set HADOOP_HOME=C:\hadoop
set HADOOP_BIN=%HADOOP_HOME%\bin

echo [1/7] Creating virtual environment if missing...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo Failed to create venv & exit /b 1 )
)

echo [2/7] Activating venv...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 ( echo Failed to activate venv & exit /b 1 )

echo [3/7] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel

echo [4/7] Installing app dependencies...
REM Core app + UI
pip install streamlit PySide6 PySide6-Addons PySide6-Essentials plotly
REM Data / lakehouse
pip install pyspark delta-spark
REM AI stack
pip install sentence-transformers chromadb
REM Optimizer
pip install ortools
REM LangChain providers (optional if used)
pip install langchain langchain-community langchain-google-genai
REM Other helpers
pip install pandas numpy scikit-learn

if errorlevel 1 ( echo Dependency install failed & exit /b 1 )

echo [5/7] Hadoop winutils setup for Windows local Spark...
REM Create C:\hadoop\bin and place winutils.exe + hadoop.dll
if not exist "%HADOOP_BIN%" mkdir "%HADOOP_BIN%"
REM Download prebuilt winutils (example for Hadoop 3.3.1); adjust version/source as needed
powershell -Command "try { \
  $wc = New-Object System.Net.WebClient; \
  $wc.DownloadFile('https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.1/bin/winutils.exe','%HADOOP_BIN%\winutils.exe'); \
  $wc.DownloadFile('https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.1/bin/hadoop.dll','%HADOOP_BIN%\hadoop.dll'); \
} catch { exit 1 }"
if errorlevel 1 ( echo Failed to download winutils; you can place files manually in %HADOOP_BIN% & goto :SKIP_WINUTILS )

:SKIP_WINUTILS

REM Set session env vars (persist to user env if admin)
setx HADOOP_HOME "%HADOOP_HOME%" >NUL
setx PATH "%PATH%;%HADOOP_BIN%" >NUL

echo [6/7] Prefetching embedding model (all-MiniLM-L6-v2)...
python - << "PY"
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding ready:", m.get_sentence_embedding_dimension(), "dims")
PY

echo [7/7] Setup complete.
echo To run the app, use: run_launcher.bat
endlocal
exit /b 0
