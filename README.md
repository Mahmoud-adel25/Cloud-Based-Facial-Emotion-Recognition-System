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


## Run with Docker

The project includes a production-ready `Dockerfile` that runs the Streamlit UI on port `8501`.

- **Prerequisites:**

  - Docker installed and running
  - (Optional) Docker Hub account if you want to push the image

- **Build the image (local):**

  ```bash
  # From the project root (where the Dockerfile is located)
  docker build -t emotion-detector:latest .
  ```

- **Run the container (local):**

  ```bash
  docker run --rm -p 8501:8501 --name emotion-detector emotion-detector:latest
  ```

- **Access the app:**

  - Streamlit UI: `http://localhost:8501`
  - Health check (for probes): `http://localhost:8501/healthz`

- **Push the image to Docker Hub (optional):**

  ```bash
  # Tag the local image with your Docker Hub username
  docker tag emotion-detector:latest <your-dockerhub-username>/emotion-detector:latest

  # Log in and push
  docker login
  docker push <your-dockerhub-username>/emotion-detector:latest
  ```

  After pushing, update any Kubernetes manifests to use your own image name instead of the example one if needed.


## Deploy to Kubernetes

The repository includes example manifests to deploy the app to a Kubernetes cluster:

- `deployment.yaml` – `Deployment` for the `emotion-detector` app
- `service.yaml` – `Service` of type `LoadBalancer` exposing port `8501`

- **Prerequisites:**

  - A running Kubernetes cluster (e.g., Minikube, Docker Desktop Kubernetes, or a cloud cluster)
  - `kubectl` configured to talk to that cluster
  - A container image accessible by the cluster (for example: `mahmoudadel25/emotion-detector:latest` on Docker Hub)

- **Apply the manifests:**

  ```bash
  # From the project root
  kubectl apply -f deployment.yaml
  kubectl apply -f service.yaml
  ```

- **Verify resources:**

  ```bash
  kubectl get pods
  kubectl get svc emotion-detector
  ```

- **Access the app:**

  - If you are using Minikube:

    ```bash
    minikube service emotion-detector
    ```

    This will open the app URL in your browser.

  - If you are using another Kubernetes setup:
    - If `type: LoadBalancer` is supported, use the external IP/hostname and port from `kubectl get svc emotion-detector`.
    - For local clusters without LoadBalancer support, you can use `kubectl port-forward`:

      ```bash
      kubectl port-forward svc/emotion-detector 8501:8501
      ```

      Then open `http://localhost:8501` in your browser.

