import os, io, logging, base64, threading
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np
from formula_detector import FormulaDetector
# from dotenv import load_dotenv
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeResult
# from azure.core.exceptions import HttpResponseError

# this logger captures backend events (model load, pdf conversion, box processing)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Math formula LaTeX-OCR service", version="2.0")

# serve static assets (e.g. favicon)
app.mount("/static", StaticFiles(directory="static"), name="static")

# add cors middleware to allow the browser frontend (index.html/ results.html)
# to call the fastapi endpoints via fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models for request/ response exchanged between frontend javascript and backend
# the BoundingBox coordinates are in canvas pixel space on the client and are used to crop the uploaded page image before passing it to pix2tex
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    id: str

class BoxExtractionRequest(BaseModel):
    boxes: List[BoundingBox]
    imageData: str
    engine: str = "local"  # local/ pix2tex or azure

class AutoDetectRequest(BaseModel):
    imageData: str
    pageNum: int = 1

def setup_latex_ocr_model(): # initializes the pix2tex LatexOCR model
    try:
        from pix2tex.cli import LatexOCR
        model = LatexOCR()
        logger.info("LaTeX-OCR model loaded successfully")
        return model
    except ImportError as e:
        logger.error(f"Failed to import LaTeX-OCR, install with: conda/ pip install pix2tex")
        raise
    except Exception as e:
        logger.error(f"Failed to load LaTeX-OCR model: {e}")
        raise

# initialize the model once at process start so all requests reuse the same instance
MODEL_LOAD_ERROR = None
_MODEL_LOCK = threading.Lock()


def ensure_latex_ocr_model_loaded() -> None:
    """Ensure the global pix2tex model is loaded.

    In some environments (notably containers) the initial model load can fail due
    to transient issues (e.g. network during weight download). Without a retry,
    the service would return 503 forever. This function makes model loading
    retryable and thread-safe.
    """

    global MODEL, MODEL_LOADED, MODEL_LOAD_ERROR

    if MODEL_LOADED and MODEL is not None:
        return

    with _MODEL_LOCK:
        if MODEL_LOADED and MODEL is not None:
            return

        try:
            MODEL = setup_latex_ocr_model()
            MODEL_LOADED = True
            MODEL_LOAD_ERROR = None
        except Exception as e:
            MODEL = None
            MODEL_LOADED = False
            MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
            logger.exception("latex-ocr model failed to load")

try:
    MODEL = setup_latex_ocr_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL = None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
    logger.exception("latex-ocr model failed to load")

# load_dotenv()
# AZURE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
# AZURE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY")
# AZURE_CLIENT = None
# AZURE_READY = False

# if AZURE_ENDPOINT and AZURE_KEY:
#     try:
#         AZURE_CLIENT = DocumentIntelligenceClient(
#             endpoint=AZURE_ENDPOINT,
#             credential=AzureKeyCredential(AZURE_KEY)
#         )
#         AZURE_READY = True
#         logger.info("Azure Document Intelligence client initialized")
#     except Exception as e:
#         logger.warning(f"Azure Document Intelligence client failed to initialize: {e}")
# else:
#     logger.info("Azure Document Intelligence credentials not found")

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]: # converts a binary pdf payload to a list of PIL.Image pages using pdf2image
    try:
        from pdf2image import convert_from_bytes
        
        images = convert_from_bytes(pdf_bytes, dpi=300)
        logger.info(f"Converted pdf to {len(images)} page images")
        return images
    except ImportError:
        logger.error("pdf2image not installed")
        return []
    except Exception as e:
        logger.error(f"Error converting pdf to images: {e}")
        return []

def azure_extract_formula_from_image(image: Image.Image) -> str:
    if not AZURE_READY or not AZURE_CLIENT:
        raise HTTPException(status_code=503, detail="Azure Document Intelligence not configured")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    poller = AZURE_CLIENT.begin_analyze_document(
        model_id="prebuilt-read",
        body=image_bytes,
        features=[DocumentAnalysisFeature.FORMULAS]
    )
    result = poller.result()

    formulas = []
    for page in result.pages:
        if hasattr(page, "formulas") and page.formulas:
            for formula in page.formulas:
                formulas.append(getattr(formula, "value", ""))

    if formulas:
        return formulas[0] # returns the first detected formula value

    return result.content.strip() if hasattr(result, "content") else "" # return content text if no formulas

def azure_analyze_image_bytes(img_bytes: bytes):
    if not AZURE_READY or not AZURE_CLIENT:
        raise HTTPException(status_code=503, detail="Azure Document Intelligence not configured")
    poller = AZURE_CLIENT.begin_analyze_document(
        model_id="prebuilt-read",
        body=img_bytes,
        features=[DocumentAnalysisFeature.FORMULAS]
    )
    return poller.result()

@app.post("/api/simple-extract")
async def simple_extract(file: UploadFile = File(...)): # simplified endpoint that uses FormulaDetector to automatically detect and extract formulas from pdf pages
    if not MODEL_LOADED:
        # Retry model loading on demand to recover from transient startup failures
        ensure_latex_ocr_model_loaded()
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"LaTeX-OCR model not loaded. {MODEL_LOAD_ERROR or ''}".strip()
        )
    
    try:
        contents = await file.read()
        filename = file.filename.lower()
        
        results = []
        
        if filename.endswith('.pdf'):
            page_images = convert_pdf_to_images(contents)
            
            for page_num, page_image in enumerate(page_images):
                try:
                    # convert PIL image to numpy array for the detector
                    image_array = np.array(page_image)
                    if len(image_array.shape) == 2: # grayscale
                        image_array = np.stack([image_array] * 3, axis=-1)
                    elif image_array.shape[2] == 4: # rgba
                        image_array = image_array[:, :, :3]
                    
                    detector = FormulaDetector()
                    formulas = detector.detect_formulas(image_array)
                    
                    # process each detected formula region
                    for region_idx, (x, y, w, h) in enumerate(formulas):
                        try:
                            cropped = page_image.crop((x, y, x + w, y + h)) # crop the detected region
                            if cropped.mode != 'L':
                                cropped = cropped.convert('L') # convert to grayscale for pix2tex
                            cropped = ImageOps.expand(cropped, border=20, fill='white')
                            
                            latex_code = MODEL(cropped) # extract latex from the detected region
                            
                            results.append({
                                "page": page_num + 1,
                                "region_index": region_idx,
                                "latex": latex_code.strip(),
                                "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                            })
                        except Exception as e:
                            logger.error(f"Error processing formula region on page {page_num + 1}: {e}")
                            results.append({
                                "page": page_num + 1,
                                "region_index": region_idx,
                                "latex": "",
                                "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                            })
                    
                    logger.info(f"Detected and processed {len(formulas)} formulas on page {page_num + 1}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
        else: # process single image
            pil_image = Image.open(io.BytesIO(contents))
            
            image_array = np.array(pil_image)
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            detector = FormulaDetector()
            formulas = detector.detect_formulas(image_array)
            
            for region_idx, (x, y, w, h) in enumerate(formulas):
                try:
                    cropped = pil_image.crop((x, y, x + w, y + h))
                    if cropped.mode != 'L':
                        cropped = cropped.convert('L')
                    cropped = ImageOps.expand(cropped, border=20, fill='white')
                    
                    latex_code = MODEL(cropped)
                    
                    results.append({
                        "page": 1,
                        "region_index": region_idx,
                        "latex": latex_code.strip(),
                        "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
                except Exception as e:
                    logger.error(f"Error processing formula region: {e}")
                    results.append({
                        "page": 1,
                        "region_index": region_idx,
                        "latex": "",
                        "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
        
        return JSONResponse({
            "filename": file.filename,
            "total_formulas_found": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in simple-extract: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/auto-detect-formulas")
async def auto_detect_formulas(request: AutoDetectRequest): # uses the FormulaDetector algorithm to automatically detect formula regions on a page image, returns a list of bounding boxes that the user can accept or reject
    try:
        image_data = request.imageData.split(',')[1] if ',' in request.imageData else request.imageData
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        detector = FormulaDetector()
        formulas = detector.detect_formulas(image_array)
        
        detected_boxes = []
        for idx, (x, y, w, h) in enumerate(formulas):
            try:
                crop = image.crop((x, y, x + w, y + h))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                cropped_bytes = buf.getvalue()
                processed_b64 = "data:image/png;base64," + base64.b64encode(cropped_bytes).decode()
            except Exception as crop_err:
                logger.warning(f"Preview generation failed for auto box {idx} on page {request.pageNum}: {crop_err}")
                processed_b64 = None

            detected_boxes.append({
                "id": f"auto_box_{request.pageNum}_{idx}",
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "isAutoDetected": True,
                "confidence": 0.85, # confidence score from detection
                "preview": processed_b64  # raw crop for preview, won't be overwritten
            })
        
        logger.info(f"Auto detected {len(detected_boxes)} formula regions on page {request.pageNum}")
        
        return JSONResponse({
            "page": request.pageNum,
            "detectedBoxes": detected_boxes,
            "totalDetected": len(detected_boxes)
        })
        
    except Exception as e:
        logger.error(f"Error in auto detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/api/extract-boxes")
async def extract_boxes(request: BoxExtractionRequest): # extracts latex from user defined bounding boxes on a page image
    # the frontend posts a json body with imageData (base64 dataurl) and boxes (canvas-space coordinates)
    # the backend decodes the base64 payload and crops per box
    engine = (request.engine or "local").lower()

    if engine == "local" and not MODEL_LOADED:
        # Retry model loading on demand to recover from transient startup failures
        ensure_latex_ocr_model_loaded()
    if engine == "local" and not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"LaTeX-OCR model not loaded. {MODEL_LOAD_ERROR or ''}".strip()
        )
    if engine == "azure" and not AZURE_READY:
        raise HTTPException(
            status_code=503,
            detail="Azure Document Intelligence not configured"
        )
    
    try:
        image_data = request.imageData.split(',')[1] if ',' in request.imageData else request.imageData
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        results = []

        if engine == "azure": # analyze each box with white padding to avoid nearby text
            try:
                for box in request.boxes:
                    x = int(box.x)
                    y = int(box.y)
                    w = int(box.width)
                    h = int(box.height)

                    cropped = image.crop((x, y, x + w, y + h))
                    padding_azure=40
                    cropped = ImageOps.expand(cropped, border=padding_azure, fill='white') # add white padding only

                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    cropped_bytes = buf.getvalue()
                    processed_b64 = "data:image/png;base64," + base64.b64encode(cropped_bytes).decode()
                    result = azure_analyze_image_bytes(cropped_bytes)

                    page_num = result.pages[0].page_number if result.pages else 1

                    found = False
                    for page in result.pages:
                        if hasattr(page, "formulas") and page.formulas:
                            for formula in page.formulas:
                                formula_value = getattr(formula, "value", "")
                                results.append({
                                    "box_id": box.id,
                                    "engine": engine,
                                    "latex": formula_value,
                                    "page": page.page_number,
                                    "region": {"x": x, "y": y, "width": w, "height": h},
                                    "processed_image": processed_b64,
                                    "padding": padding_azure
                                })
                                found = True
                    if not found:
                        # return content text if no formulas detected
                        fallback_text = result.content.strip() if hasattr(result, "content") else ""
                        results.append({
                            "box_id": box.id,
                            "engine": engine,
                            "latex": fallback_text,
                            "page": page_num,
                            "region": {"x": x, "y": y, "width": w, "height": h},
                            "processed_image": processed_b64,
                            "padding": padding_azure
                        })

                return JSONResponse({
                    "total_boxes": len(request.boxes),
                    "engine": engine,
                    "results": results,
                    "formula_count": len(results)
                })
            except Exception as e:
                logger.error(f"Azure extraction error: {e}")
                raise HTTPException(status_code=500, detail=f"Azure extraction error: {str(e)}")
        if engine == "local": # local/ pix2tex path
            for box in request.boxes: # calculate pixel coordinates from client sent floats
                x = int(box.x)
                y = int(box.y)
                w = int(box.width)
                h = int(box.height)
                
                try:
                    cropped = image.crop((x, y, x + w, y + h))
                    if cropped.mode != 'L':
                        cropped = cropped.convert('L')

                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    processed_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

                    latex_code = MODEL(cropped)
                    
                    results.append({
                        "box_id": box.id,
                        "engine": engine,
                        "latex": latex_code.strip() if isinstance(latex_code, str) else latex_code,
                        "region": {"x": x, "y": y, "width": w, "height": h},
                        "processed_image": processed_b64,
                        "padding": 0
                    })
                except Exception as e:
                    logger.error(f"Error processing box {box.id}: {e}")
                    results.append({
                        "box_id": box.id,
                        "engine": engine,
                        "latex": "",
                        "region": {"x": x, "y": y, "width": w, "height": h}
                    })
            
            return JSONResponse({
                "total_boxes": len(request.boxes),
                "engine": engine,
                "results": results
            })
        
    except Exception as e:
        logger.error(f"Error in box extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/results", response_class=HTMLResponse)
async def serve_results():
    try:
        with open("results.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>results.html not found</h1>", status_code=404)

@app.get("/floating-formulas-html", response_class=HTMLResponse)
async def floating_formulas_html(): # returns server rendered floating formulas as html that mathjax can process
    try:
        with open("floating-formulas.txt", "r", encoding="utf-8") as f:
            content = f.read()
        import re
        formula_matches = re.findall(r'\\\[(.*?)\\\]', content, re.DOTALL) # split by \[ and \] to extract all formulas
        
        logger.info(f"total formulas extracted: {len(formula_matches)}")
        for i, formula in enumerate(formula_matches, 1):
            logger.info(f"formula {i}: {formula[:50]}...")
        
        formulas_html = ""
        import random
        
        for idx, formula_content in enumerate(formula_matches, 1):
            # generate random position
            top = random.randint(10, 80)
            left = random.randint(10, 80)
            
            anim_num = (idx - 1) % 4 + 1
            duration = 20 + (idx - 1) * 2
            logger.info(f"floating formula {idx} positioned at {top}% top, {left}% left")
            formulas_html += f"""<div class="floating-formula" style="top: {top}%; left: {left}%; animation: float{anim_num} {duration}s ease-in-out infinite;">\\[{formula_content}\\]</div>"""
        
        logger.info(f"Generated {len(formula_matches)} floating formulas")
        return formulas_html
    except Exception as e:
        logger.error(f"Error generating floating formulas html: {e}")
        return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)