import os, sys, subprocess, time, signal, ctypes
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtWebEngineWidgets import QWebEngineView

APP_HOST = os.environ.get("TFC_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("TFC_PORT", "8501"))
MUTEX_NAME = "Global\\TFCDesktopSingleton"

def is_frozen():
    return getattr(sys, "frozen", False)

def app_dir() -> Path:
    # Folder where exe lives when frozen; else script folder
    return Path(sys.executable).parent if is_frozen() else Path(__file__).resolve().parent

def python_cmd() -> Path:
    # Use embedded/venv Python shipped in runtime/
    rt = app_dir() / "runtime" / ("python.exe" if os.name == "nt" else "python3")
    if rt.exists():
        return rt
    # Fallback for dev runs only (not frozen)
    return Path(sys.executable)

def app_script() -> Path:
    # app.py is copied alongside exe (PyInstaller --add-data)
    p = app_dir() / "app.py"
    if not p.exists():
        QMessageBox.critical(None, "Error", f"app.py not found at: {p}")
        sys.exit(1)
    return p

def wait_for_server(url: str, timeout_s: int = 40) -> bool:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False

def single_instance_guard() -> bool:
    # Returns True if another instance exists
    kernel32 = ctypes.windll.kernel32
    kernel32.SetLastError(0)
    h_mutex = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    last_err = kernel32.GetLastError()
    # Keep handle alive by storing globally
    globals()["_TFC_MUTEX_HANDLE"] = h_mutex
    ERROR_ALREADY_EXISTS = 183
    return last_err == ERROR_ALREADY_EXISTS

def main():
    # Single instance
    if os.name == "nt" and single_instance_guard():
        QMessageBox.information(None, "Already running", "The app is already open.")
        sys.exit(0)

    # Resolve interpreter and script paths
    py = str(python_cmd())
    script = str(app_script())

    # Prepare writable data roots (avoid Program Files)
    data_root = os.environ.get("TFC_DATA_ROOT", str(Path(os.environ.get("LOCALAPPDATA", app_dir())) / "TFC"))
    os.makedirs(data_root, exist_ok=True)
    os.environ.setdefault("TFC_DATA_ROOT", data_root)
    os.environ.setdefault("TFC_EMP_DELTA", str(Path(data_root) / "employees.delta"))
    os.environ.setdefault("TFC_PROJ_DELTA", str(Path(data_root) / "projects.delta"))
    os.environ.setdefault("TFC_CHROMA", str(Path(data_root) / "chroma_db"))
    for d in ["TFC_EMP_DELTA", "TFC_PROJ_DELTA", "TFC_CHROMA"]:
        Path(os.environ[d]).parent.mkdir(parents=True, exist_ok=True)

    # Start Streamlit with a real Python interpreter (prevents self-recursion)
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["BROWSER"] = "none"
    cmd = [
        py, "-m", "streamlit", "run", script,
        "--server.headless=true",
        "--server.address", APP_HOST,
        "--server.port", str(APP_PORT),
        "--browser.gatherUsageStats=false",
        "--browser.serverAddress=localhost",
        "--browser.serverPort=0",
        "--server.runOnSave=false",
    ]
    # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK cleanly if needed
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        p = subprocess.Popen(cmd, env=env, creationflags=creationflags)
    except Exception as e:
        QMessageBox.critical(None, "Launch error", f"Failed to start server: {e}")
        sys.exit(1)

    # Qt window
    app = QApplication(sys.argv)
    view = QWebEngineView()
    view.setWindowTitle("Team Formation Cortex")
    view.resize(1280, 800)

    url = f"http://{APP_HOST}:{APP_PORT}"
    if not wait_for_server(url, timeout_s=45):
        try:
            p.terminate()
        except Exception:
            pass
        QMessageBox.critical(None, "Error", "Streamlit server failed to start.")
        sys.exit(1)

    view.load(QUrl(url))
    view.show()

    def cleanup():
        try:
            if p and p.poll() is None:
                if os.name == "nt":
                    # Send CTRL_BREAK to the process group so Streamlit stops gracefully
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                    time.sleep(0.5)
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        except Exception:
            pass

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
