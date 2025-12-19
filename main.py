from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import io

# Import numpy first with error handling
try:
    import numpy as np
    print(f"NumPy version: {np.__version__} imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import NumPy: {e}")
    print("This is a critical error. NumPy must be installed for the app to work.")
    raise ImportError(f"NumPy is not available. Please install numpy: {e}")

# Verify numpy is actually usable
try:
    _ = np.array([1, 2, 3])
    print("NumPy functionality verified")
except Exception as e:
    print(f"ERROR: NumPy imported but not functional: {e}")
    raise

# Import TensorFlow (requires numpy)
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__} imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import TensorFlow: {e}")
    raise ImportError(f"TensorFlow is not available. Please install tensorflow: {e}")

from PIL import Image
import cv2

app = FastAPI()

# =============================================================================
# Preprocessing functions (must match training preprocessing exactly)
# =============================================================================

def _manual_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to grayscale image."""
    img = img.astype(np.uint8)
    h, w = img.shape

    # Calculate histogram
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            intensity = img[y, x]
            hist[intensity] += 1

    # Calculate CDF
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = next(c for c in cdf if c > 0)

    # Create lookup table
    total_pixels = h * w
    lut = [0] * 256
    for i in range(256):
        lut[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255) if total_pixels > cdf_min else 0

    # Apply equalization
    img_eq = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            img_eq[y, x] = lut[img[y, x]]

    return img_eq


def _normalize(img: np.ndarray) -> np.ndarray:
    """Percentile-based normalization."""
    img_float = img.astype(np.float32)
    p_low, p_high = np.percentile(img_float, [5, 95])
    
    # Handle edge case where p_low == p_high
    if p_high - p_low < 1e-6:
        return img.astype(np.uint8)
    
    img_normalized = np.clip((img_float - p_low) * 255.0 / (p_high - p_low), 0, 255)
    return img_normalized.astype(np.uint8)


def _manual_sharpen(img: np.ndarray, amount: float = None) -> np.ndarray:
    """Apply sharpening with adaptive amount based on variance."""
    if amount is None:
        variance = np.var(img)
        amount = min(1.0, max(0.2, variance / 5000))

    blur_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16.0

    img = img.astype(np.float32)
    padded = np.pad(img, ((1, 1), (1, 1)), mode='edge')
    blurred = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+3, j:j+3]
            blurred[i, j] = np.sum(region * blur_kernel)

    sharpened = img + amount * (img - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _apply_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline used during training:
    1. Histogram equalization
    2. Sharpening
    3. Percentile normalization
    4. Denoising
    """
    img_eq = _manual_histogram_equalization(img)
    img_sharp = _manual_sharpen(img_eq, None)
    img_norm = _normalize(img_sharp)
    img_denoised = cv2.fastNlMeansDenoising(img_norm, h=10)
    return img_denoised

# Load model once at startup
# Try multiple path resolution strategies to handle both FastAPI and Streamlit contexts
_ROOT = Path(__file__).resolve().parents[1]

# Fallback: if running from Streamlit or different context, try other locations
def find_root_dir():
    """Find the root directory containing the model files."""
    candidates = [
        Path(__file__).resolve().parents[1],  # Parent of app/ directory
        Path.cwd(),  # Current working directory
        Path.cwd().parent,  # Parent of current working directory
        Path("/app"),  # Docker container default
    ]
    
    for candidate in candidates:
        keras_file = candidate / "best_model.keras"
        yolo_file = candidate / "face_yolov8n.pt"
        if keras_file.exists() or yolo_file.exists():
            print(f"Found model directory at: {candidate}")
            return candidate
    
    # Return the first candidate as default (will show error later if files don't exist)
    print(f"Warning: Could not find model files in any candidate directory. Using: {candidates[0]}")
    return candidates[0]

_ROOT = find_root_dir()

model = tf.keras.models.load_model(str(_ROOT / "best_model.keras"), compile=False)
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Try to import ultralytics with detailed error reporting
YOLO = None
_yolo_import_error = None
try:
    import sys
    print("Checking if ultralytics is installed...")
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__ if hasattr(ultralytics, '__version__') else 'unknown'}")
    except ImportError as ie:
        print(f"Failed to import ultralytics package: {ie}")
        _yolo_import_error = f"ultralytics package not found: {ie}"
        raise
    
    # Try importing YOLO class
    from ultralytics import YOLO  # type: ignore
    print("Successfully imported YOLO from ultralytics")
except ImportError as ie:
    _yolo_import_error = f"Failed to import ultralytics: {ie}. Make sure ultralytics and PyTorch are properly installed."
    print(f"ERROR: {_yolo_import_error}")
    import traceback
    print(f"Import traceback: {traceback.format_exc()}")
except Exception as e:  # pragma: no cover
    _yolo_import_error = f"Unexpected error importing ultralytics: {type(e).__name__}: {e}"
    print(f"ERROR: {_yolo_import_error}")
    import traceback
    print(f"Import traceback: {traceback.format_exc()}")

_yolo_model = None
_yolo_error = None
if YOLO is not None:
    _yolo_path = (_ROOT / "face_yolov8n.pt").resolve()  # Use absolute path
    if not _yolo_path.exists():
        _yolo_error = f"YOLO model file not found at: {_yolo_path}. Current working directory: {Path.cwd()}, Root: {_ROOT}, File exists: {_yolo_path.exists()}"
        print(f"ERROR: {_yolo_error}")
        # List files in root directory for debugging
        try:
            root_files = list(_ROOT.iterdir())
            print(f"Files in root directory ({_ROOT}): {[f.name for f in root_files]}")
        except Exception as list_err:
            print(f"Could not list root directory: {list_err}")
    else:
        try:
            yolo_path_str = str(_yolo_path)
            print(f"Attempting to load YOLO model from: {yolo_path_str}")
            _yolo_model = YOLO(yolo_path_str)
            print(f"Successfully loaded YOLO model from: {yolo_path_str}")
        except Exception as e:
            _yolo_error = f"Failed to load YOLO model from {_yolo_path}: {type(e).__name__}: {e}"
            print(f"ERROR: {_yolo_error}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
else:
    if _yolo_import_error:
        _yolo_error = _yolo_import_error
    else:
        _yolo_error = "ultralytics package is not installed or could not be imported"
    print(f"ERROR: {_yolo_error}")

def _prepare_pil_for_model(img: Image.Image) -> np.ndarray:
    """
    Prepare a PIL image into a batch that matches model.input_shape.
    Applies the same preprocessing pipeline used during training.
    """
    # Ensure numpy is available
    try:
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="NumPy is not available. Please install numpy.")

    # Determine expected (H, W, C) from model input shape. Typical: (None, H, W, C)
    input_shape = getattr(model, "input_shape", None)
    if not input_shape or len(input_shape) != 4:
        raise HTTPException(status_code=500, detail=f"Unsupported model input_shape: {input_shape}")

    _, h, w, c = input_shape
    if not all(isinstance(x, int) and x > 0 for x in (h, w, c)):
        raise HTTPException(status_code=500, detail=f"Dynamic/unknown model input_shape: {input_shape}")

    if c == 1:
        img = img.convert("L")  # grayscale
    elif c == 3:
        img = img.convert("RGB")
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported channel count in model: {c}")

    # Resize to model input size
    img = img.resize((w, h))
    arr = np.array(img, dtype=np.uint8)

    # Apply the same preprocessing pipeline used during training
    if c == 1:
        # For grayscale, apply the full preprocessing pipeline
        arr = _apply_preprocessing(arr)
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
    elif c == 3:
        # For RGB (if ever used), just normalize
        arr = arr.astype(np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise HTTPException(status_code=400, detail=f"Unexpected RGB image array shape: {arr.shape}")

    arr = np.expand_dims(arr, axis=0)  # batch
    return arr

def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

def _detect_largest_face(img_rgb: Image.Image) -> tuple[Image.Image, dict]:
    """
    Detect faces using YOLO and return the largest face crop + metadata.
    Expects an RGB PIL image.
    """
    # Ensure numpy is available
    try:
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="NumPy is not available. Please install numpy.")
    
    if _yolo_model is None:
        error_msg = "YOLO model not available. "
        if _yolo_error:
            error_msg += _yolo_error
        else:
            error_msg += "Ensure `ultralytics` is installed and `face_yolov8n.pt` exists."
        raise HTTPException(
            status_code=500,
            detail=error_msg,
        )

    np_img = np.array(img_rgb)  # RGB
    results = _yolo_model.predict(np_img, verbose=False)
    if not results:
        raise HTTPException(status_code=400, detail="No detections returned by YOLO.")

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        raise HTTPException(status_code=400, detail="No face detected.")

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None

    # pick largest area box
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.argmax(areas))
    x1, y1, x2, y2 = xyxy[i].tolist()

    # add small margin
    w, h = img_rgb.size
    bw, bh = (x2 - x1), (y2 - y1)
    mx, my = 0.15 * bw, 0.15 * bh
    x1 = max(0, int(round(x1 - mx)))
    y1 = max(0, int(round(y1 - my)))
    x2 = min(w, int(round(x2 + mx)))
    y2 = min(h, int(round(y2 + my)))

    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="Invalid face bounding box.")

    crop = img_rgb.crop((x1, y1, x2, y2))
    meta = {"box": [x1, y1, x2, y2]}
    if conf is not None:
        meta["confidence"] = float(conf[i])
    return crop, meta

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = _open_image(image_bytes).convert("RGB")
    face, face_meta = _detect_largest_face(img)
    image = _prepare_pil_for_model(face)

    try:
        preds = model.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    idx = int(np.argmax(preds))
    if idx < 0 or idx >= len(class_names):
        raise HTTPException(
            status_code=500,
            detail=f"Model output index {idx} out of range for class_names (len={len(class_names)})",
        )

    return {"prediction": class_names[idx], "face": face_meta}


# Streamlit UI (keeps existing FastAPI endpoints intact). When this file is
# run with `streamlit run app/main.py` the UI below will execute. If
# Streamlit is not installed or import fails, the API remains usable.
try:
    import streamlit as st
except Exception:  # pragma: no cover - UI is optional
    st = None


# Helper function to get numpy - handles Streamlit module reloading
def _get_numpy():
    """Get numpy, importing it fresh if needed to handle Streamlit reloading."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("NumPy is not installed. Please install numpy in the Docker container.")



def _run_upload_tab():
    """Run the image upload section."""
    st.subheader("📁 Upload an Image")
    st.write("Upload a photo with a face to detect and analyze the emotion.")
    
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return
    
    try:
        np = _get_numpy()
        
        image_bytes = uploaded.read()
        img = _open_image(image_bytes).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Original image", use_container_width=True)
        
        # Detect face and show crop
        face, face_meta = _detect_largest_face(img)
        
        with col2:
            st.image(face, caption="Detected face", use_container_width=True)
        
        # Prepare for model and predict
        data = _prepare_pil_for_model(face)
        preds = model.predict(data, verbose=0)
        idx = int(np.argmax(preds))
        pred = class_names[idx] if 0 <= idx < len(class_names) else "Unknown"
        confidence = float(preds[0][idx])
        
        # Show result with color
        if pred in ["Happy", "Surprise"]:
            st.success(f"🎭 Prediction: **{pred}** ({confidence:.0%} confidence)")
        elif pred == "Neutral":
            st.info(f"🎭 Prediction: **{pred}** ({confidence:.0%} confidence)")
        else:
            st.warning(f"🎭 Prediction: **{pred}** ({confidence:.0%} confidence)")
        
        with st.expander("📋 Detection Details"):
            st.json({"prediction": pred, "confidence": f"{confidence:.2%}", "face": face_meta})
            
    except HTTPException as e:
        st.error(f"Error: {e.detail}")
    except Exception as e:
        st.error(f"Error ({type(e).__name__}): {e}")
        import traceback
        with st.expander("🔍 Show Full Error Details"):
            st.code(traceback.format_exc(), language="python")


def _run_streamlit_app() -> None:
    if st is None:
        return

    try:
        # Page config MUST be first Streamlit command
        st.set_page_config(
            page_title="Emotion Monitor",
            page_icon="🎭",
            layout="wide"
        )
    except Exception as e:
        # Already set, ignore
        pass

    # Get numpy using helper function
    try:
        np = _get_numpy()
        test_arr = np.array([1, 2, 3])
        if test_arr is None:
            st.error("⚠️ NumPy is imported but not functional")
            return
    except ImportError as e:
        st.error(f"⚠️ Critical Error: {e}")
        st.error("The app cannot function without NumPy. Please rebuild the Docker image.")
        return
    except Exception as e:
        st.error(f"⚠️ Error with NumPy: {e}")
        return

    # Header
    st.title("🎭 Emotion Detection")
    st.write("AI-powered facial expression analysis for student engagement monitoring")
    
    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        try:
            if _yolo_model is not None:
                st.success("✅ Face Detection: Ready")
            else:
                st.error(f"❌ Face Detection: {_yolo_error or 'Not loaded'}")
        except Exception as e:
            st.error(f"❌ Face Detection Error: {e}")
    with col2:
        try:
            if model is not None:
                st.success("✅ Emotion Model: Ready")
            else:
                st.error("❌ Emotion Model: Not loaded")
        except Exception as e:
            st.error(f"❌ Emotion Model Error: {e}")
    
    st.write("---")
    
    # Upload section (no tabs needed)
    try:
        _run_upload_tab()
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    # Allow running the Streamlit UI when executed directly (helpful for dev).
    if st is not None:
        _run_streamlit_app()
