@echo off
setlocal

set VENV_DIR=.venv
set APP_PORT=8501
set APP_HOST=127.0.0.1
set DATA_ROOT=%LOCALAPPDATA%\TFC
set LOCK_FILE=%LOCALAPPDATA%\TFC\.app_lock

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Virtual environment not found. Run setup_env.bat first.
  exit /b 1
)

if not exist "%DATA_ROOT%" mkdir "%DATA_ROOT%"
if not exist "%LOCALAPPDATA%\TFC\logs" mkdir "%LOCALAPPDATA%\TFC\logs"

REM Single-instance guard (naive file lock)
if exist "%LOCK_FILE%" (
  echo App appears to be running already. If not, delete %LOCK_FILE% and retry.
  exit /b 0
)
type nul > "%LOCK_FILE%"

call "%VENV_DIR%\Scripts\activate.bat"

REM Spark/Hadoop session vars
set HADOOP_HOME=C:\hadoop
set PATH=%PATH%;%HADOOP_HOME%\bin

REM App data locations
set TFC_DATA_ROOT=%DATA_ROOT%
set TFC_EMP_DELTA=%DATA_ROOT%\employees.delta
set TFC_PROJ_DELTA=%DATA_ROOT%\projects.delta
set TFC_CHROMA=%DATA_ROOT%\chroma_db

REM Streamlit preferences
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set BROWSER=none
set TFC_PORT=%APP_PORT%
set TFC_HOST=%APP_HOST%

echo Starting launcher...
python launcher.py 1>>"%LOCALAPPDATA%\TFC\logs\stdout.log" 2>>"%LOCALAPPDATA%\TFC\logs\stderr.log"
set EXITCODE=%ERRORLEVEL%

echo Cleaning lock...
del "%LOCK_FILE%" >NUL 2>&1

endlocal
exit /b %EXITCODE%
