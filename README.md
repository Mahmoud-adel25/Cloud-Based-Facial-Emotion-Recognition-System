## Local setup (Windows) with `.venv` (no reinstall every time)

### Create venv (once)

````markdown
## Local setup (Windows) with `.venv` (no reinstall every time)

### Create venv (once)

```powershell
cd "C:\Users\mahmo\Downloads\project cloud docker"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Run API

```powershell
cd "C:\Users\mahmo\Downloads\project cloud docker"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Test

- Health: `http://127.0.0.1:8000/healthz`
- Swagger: `http://127.0.0.1:8000/docs`



````

## Local Development Environment

Follow these steps to create a local Python virtual environment, install dependencies, and run the Streamlit UI or FastAPI server.

- **Create venv (Windows, using system Python or 3.10 if available):**

```powershell
cd "d:\Year 4 Term 1 CS\cloud\cloud project\project cloud docker2\project cloud docker"
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

- **Activate venv (Windows PowerShell):**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
# or
. \.venv\Scripts\Activate.ps1
```

- **Create venv (Linux/macOS, using Python 3.10 or system Python):**

```bash
cd "d/Year 4 Term 1 CS/cloud/cloud project/project cloud docker2/project cloud docker"
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- **Run the Streamlit app (dev UI):**

```bash
# From project root with venv activated
streamlit run app/main.py
```

- **Run the FastAPI server (existing API):**

```powershell
# Windows PowerShell with venv activated
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Notes:

- The Streamlit UI will reuse the same model and detection code present in `app/main.py`.
- Do not modify model file paths; the repository expects `best_model.keras` and `face_yolov8n.pt` to remain in the project root.



