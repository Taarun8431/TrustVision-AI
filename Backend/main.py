from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session
import sys
import os

# Add the root 'TrustVision-AI' directory to the path so Backend can see ml/ and database/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference import DeepfakeDetector
from ml.gemini_verifier import get_verifier
from Backend.reporting import MODEL_VERSION, generate_pdf_report
from database.database import SessionLocal, engine
from database import models

# Create the database tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# Initialize the Deepfake ML Detector globally so it loads once on startup
detector = DeepfakeDetector()
gemini = get_verifier()

app = FastAPI(
    title="TrustVision AI",
    description="Enterprise Media Authenticity Platform APIs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to TrustVision AI API"}

@app.get("/health")
def health_check():
    metadata = detector.metadata()
    return {
        "status": "healthy",
        "model_loaded": metadata["model_loaded"],
        "model_version": metadata["model_version"],
        "gemini_enabled": gemini.enabled,
    }

@app.get("/model-info")
def model_info():
    return detector.metadata()

# Dependency to get a Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/scan")
async def scan_media(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Receives an uploaded image, runs the deepfake ML model,
    saves the prediction to the database, and returns the result/heatmap.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported currently.")
        
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # 1. Run inference on the ML model
        result = detector.predict_image(image_bytes)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"ML Processing Error: {result.get('error')}")

        # 2. Gemini Vision secondary verification (overrides NN for modern AI images)
        gemini_result = gemini.verify(image_bytes)
        gemini_verdict = None
        if gemini_result["enabled"] and gemini_result["verdict"] and not gemini_result["error"]:
            gemini_verdict = gemini_result["verdict"]
            # If Gemini is confident (>= 75%) that image is AI/deepfake, override the prediction
            if gemini_result["is_fake"] and (gemini_result["confidence"] or 0) >= 75:
                result["prediction"] = "FAKE"
                result["risk_level"] = "HIGH"
                result["confidence"] = max(result["confidence"], float(gemini_result["confidence"] or 0))
                result["authenticity_score"] = round(
                    min(result["authenticity_score"], 30.0), 3
                )
                result["real_probability"] = min(result.get("real_probability", 100.0), result["authenticity_score"])
                result["fake_probability"] = max(result.get("fake_probability", 0.0), 100.0 - result["authenticity_score"])
            # If Gemini is very confident (>= 85%) the image is REAL but NN says FAKE, trust Gemini
            elif not gemini_result["is_fake"] and result["prediction"] == "FAKE" and (gemini_result["confidence"] or 0) >= 85:
                result["prediction"] = "REAL"
                result["risk_level"] = "LOW"
                result["confidence"] = max(result["confidence"], float(gemini_result["confidence"] or 0))
                result["authenticity_score"] = round(
                    max(result["authenticity_score"], 70.0), 3
                )
                result["real_probability"] = max(result.get("real_probability", 0.0), result["authenticity_score"])
                result["fake_probability"] = min(result.get("fake_probability", 100.0), 100.0 - result["authenticity_score"])
            
        # 3. Save the result to our database
        scan_record = models.ScanResult(
            filename=file.filename,
            authenticity_score=result["authenticity_score"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            user_id=1 # Default user since auth isn't wired yet
        )
        db.add(scan_record)
        db.commit()
        db.refresh(scan_record)
        
        # 4. Return the response to the frontend
        return {
            "id": scan_record.id,
            "filename": scan_record.filename,
            "authenticity_score": scan_record.authenticity_score,
            "prediction": scan_record.prediction,
            "confidence": scan_record.confidence,
            "risk_level": scan_record.risk_level,
            "timestamp": scan_record.timestamp.isoformat() if scan_record.timestamp else None,
            "real_probability": result.get("real_probability"),
            "fake_probability": result.get("fake_probability"),
            "model_version": result.get("model_version", MODEL_VERSION),
            "class_map": result.get("class_map"),
            "class_map_source": result.get("class_map_source"),
            "forensic_analysis": result.get("forensic_analysis"),
            "report_url": f"/reports/{scan_record.id}",
            "heatmap_base64": result.get("heatmap_base64"),
            "heatmap_error": result.get("heatmap_error"),
            "gemini_verdict": gemini_verdict,
            "gemini_reasoning": gemini_result.get("reasoning") if gemini_result["enabled"] else None,
            "gemini_confidence": gemini_result.get("confidence") if gemini_result["enabled"] else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{scan_id}")
def download_report(scan_id: int, db: Session = Depends(get_db)):
    scan_record = db.query(models.ScanResult).filter(models.ScanResult.id == scan_id).first()
    if scan_record is None:
        raise HTTPException(status_code=404, detail="Scan result not found")

    try:
        pdf_bytes, report_filename, _ = generate_pdf_report(scan_record)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{report_filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")
