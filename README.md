# LatexOCR_Azure

## Demo video
https://youtu.be/MDS35h-oiCY

## Installation

```bash
python -m venv .venv

# PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.pure-python.txt

python app.py

# open browser
http://localhost:8000
```

### Notes (Windows)

- `requirements.pure-python.txt` intentionally avoids `pix2tex` so it can install on newer Python versions (e.g. 3.13). The server will run, but the **local** LaTeX-OCR engine may be unavailable (endpoints will return 503 for local engine).
- If you want the full local LaTeX-OCR pipeline (`pix2tex`), use Python 3.11 and install from `requirements.txt` instead.
- PDF conversion via `pdf2image` on Windows requires Poppler installed and available on PATH.

## Developer: Key functions and call flow

This section documents the main functions and steps inside `app.py` and `formula_detector.py` so developers can quickly understand the core implementation.

**`app.py`**
- **`setup_latex_ocr_model()`**: imports and initializes the local `pix2tex` `LatexOCR` model (if available).
- **`ensure_latex_ocr_model_loaded()`**: thread-safe retry logic used to load the model on demand and avoid permanent 503s on transient failures.
- **`convert_pdf_to_images(pdf_bytes)`**: converts a PDF binary payload into a list of `PIL.Image` pages using `pdf2image`.
- **Azure helpers**: `azure_extract_formula_from_image(image)` and `azure_analyze_image_bytes(img_bytes)` wrap calls to Azure Document Intelligence (`prebuilt-read` + `DocumentAnalysisFeature.FORMULAS`). They return formula text or the analyze result.
- **Endpoints and their flow**:
  - **`/api/simple-extract`**: full pipeline for uploaded PDFs or images. For PDFs it calls `convert_pdf_to_images()`, for each page it:
    - converts the page to a NumPy array
    - instantiates `FormulaDetector()` and calls `detect_formulas(image_array)`
    - crops each returned `(x,y,w,h)` region from the `PIL.Image`, pads/normalizes to grayscale, then calls the local `MODEL(cropped)` to produce LaTeX
  - **`/api/auto-detect-formulas`**: accepts a base64 image, decodes to `PIL.Image`, runs `FormulaDetector.detect_formulas()` and returns preview crops and bounding boxes for the frontend.
  - **`/api/extract-boxes`**: accepts user-specified boxes (canvas coordinates) plus an `engine` parameter (`local` or `azure`). For `local` it crops, converts to `L`, and calls `MODEL(cropped)`; for `azure` it pads the crop and calls `azure_analyze_image_bytes()`.
- **Behavior notes**: the app checks `MODEL_LOADED`/`AZURE_READY` and returns HTTP 503 when the requested engine is unavailable. Environment variables `DOCUMENT_INTELLIGENCE_ENDPOINT` and `DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY` enable Azure mode.

**`formula_detector.py`**
- **`FormulaDetector`**: the main detection class used by `app.py`.
  - `pdf_to_images(pdf_path, dpi=300)`: renders PDF pages via `fitz` into RGB NumPy arrays for the CV pipeline.
  - `preprocess_image(image)`: converts to grayscale, applies adaptive thresholding and denoising to prepare for contour detection.
  - `detect_formulas(image)`: main detection routine. It preprocesses, inverts the image, applies horizontal and vertical dilation kernels to connect formula components, finds contours, filters candidates by area/aspect/size, merges nearby regions via `merge_nearby_formulas()`, and returns a sorted list of `(x, y, w, h)` regions in image pixels.
  - `filter_text_regions(formulas, page_obj, img_shape)`: optional PDF-text based filter that extracts text from the PDF page (via `fitz`) inside each region and decides whether it is likely a formula using heuristics.
  - `is_formula_text(text, width, height)`: heuristic scoring function that inspects operators, math symbols, line structure, and descriptive keywords to accept/reject regions as formulas.
  - `merge_nearby_formulas(formulas, horizontal_gap, vertical_gap)`: merges boxes that are close horizontally/vertically (useful for multi-line formulas or stacked elements).
  - `crop_formula(image, region, padding=10)`: safe crop helper that adds padding and clamps to image bounds.
  - `process_pdf(pdf_path, output_dir)`: end-to-end helper that converts a PDF to images, detects formulas, crops and saves them to disk (used by the CLI-style `main()` in the module).

**Developer notes / tuning**
- Detector hyperparameters (thresholds, kernel sizes, area limits, gaps) are defined inline in `formula_detector.py` — tune them for different scan resolutions or document styles.
- Coordinate convention: `FormulaDetector` returns regions in image pixel space; `app.py` uses those same pixel coordinates to crop `PIL.Image` pages.
- For debugging, inspect `debug_detections.py` and the `extracted_formulas/` output directory to view crops and annotated pages.

## Demo

Running the full pipeline:
```bash
curl.exe -X POST http://localhost:8000/api/simple-extract `
  -F "file=@test_descriptive_text.pdf" `
  -H "accept: application/json"
```

## Docker + Azure Container App


```bash
docker build -t latexocr:latest .

# one time testing command since the image is going to be public
docker run -p 8000:8000 -e DOCUMENT_INTELLIGENCE_ENDPOINT="" -e DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY="" latexocr:latest

# access at http://localhost:8000
```

Pure-python oriented Dockerfile (no pix2tex):

```bash
docker build -f Dockerfile.pure-python -t latexocr:pure-python .
docker run --rm -p 8000:8000 latexocr:pure-python
```

Docker Compose (pure-python):

```bash
# build + start (runs in the background)
docker compose up -d --build

# follow logs
docker compose logs -f

# stop (keeps containers, so you can start again)
docker compose stop

# start again after stop
docker compose start

# restart quickly
docker compose restart

# stop and remove containers
docker compose down
```

## Kubernetes (Minikube)

Run the pure-python container on a local Kubernetes cluster.

Start Minikube:

```bash
minikube start
```

Optional (more stable than `kubectl port-forward`): expose the `NodePort` on your host.

This publishes the cluster's `nodePort: 30080` as host port `8000` (docker driver only):

```bash
# NOTE: port publishing is set when the minikube container is created.
# If your cluster already exists, do `minikube delete` first.
minikube start --driver=docker --container-runtime=containerd --ports=8000:30080 --listen-address=0.0.0.0
```

Build the image into Minikube (recommended):

```bash
minikube image build -t latexocr:pure-python -f Dockerfile.pure-python .
```

Deploy:

```bash
kubectl apply -f k8s/minikube.yaml
kubectl get pods
kubectl get svc latexocr
```

Note: `k8s/minikube.yaml` uses `replicas: 1` by default (pix2tex/torch is heavy and Minikube is usually single-node). Increase replicas only if you have enough CPU/RAM.

Access (local dev):

```bash
minikube service latexocr --url
```

Alternative access (bind on the host):

```bash
kubectl port-forward --address 0.0.0.0 svc/latexocr 8000:8000
```

If `kubectl port-forward` drops (e.g. "broken pipe"), run the auto-restarting helper:

```powershell
./scripts/k8s-port-forward.ps1 -LocalPort 8000 -Address 0.0.0.0
```

### After reboot

In most cases you do **not** need to redo the full setup.

- Start Docker Desktop (required for the `--driver=docker` Minikube driver)
- Start the cluster: `minikube start`
- Check pods: `kubectl get pods`
- Re-run access commands as needed:
  - `minikube service latexocr --url` (the URL can change)
  - `kubectl port-forward ...` (must be re-run every time)

You only need to rebuild/redeploy if you changed the code/image:

```bash
minikube image build -t latexocr:pure-python -f Dockerfile.pure-python .
kubectl rollout restart deployment/latexocr
```

If you run `minikube delete`, the cluster is removed and you will need to deploy again.

Cleanup:

```bash
kubectl delete -f k8s/minikube.yaml
```

```bash
docker tag latexocr:latest andialexandrescu/latexocr:latest
docker push andialexandrescu/latexocr:latest
```

```bash
docker pull andialexandrescu/latexocr:latest
docker run -p 8000:8000 andialexandrescu/latexocr:latest
```

When creating the container app, add environment variables:

DOCUMENT_INTELLIGENCE_ENDPOINT = ...
DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY = ...

Then enable external ingress on port 8000

Updating docker image + container app
```bash
 $TAG="2026-01-23-1"
docker build -t andialexandrescu/latexocr:$TAG .

docker push andialexandrescu/latexocr:$TAG

$RESOURCE_GROUP="latex-ocr"
$CONTAINER_APP_NAME="latex-ocr-container-app"
az containerapp update `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --image "andialexandrescu/latexocr:$TAG"
```
