import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from clientdocument import process_document

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"

for directory in (UPLOAD_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Document Processing API",
    description="Upload client requirement documents, process them into business reports, and retrieve generated PDFs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/process")
async def process_upload(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    mode: str = Form("master"),
    developer_count: int = Form(1),
    project_budget: float = Form(5000),
):
    ...
    logger.info("Received instruction: %r", instruction)

    """Accept a document upload, run the processing pipeline, and return a download URL."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are supported.")

    document_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{document_id}{suffix}"
    report_path = REPORT_DIR / f"{document_id}.pdf"

    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("📥 Received upload %s", upload_path)

        generated_report, json_data = process_document(
            str(upload_path),
            str(report_path),
            instruction=instruction or None,
            mode=mode,
            developer_count=developer_count,
            project_budget=project_budget,
        )
        logger.info("📤 Generated report %s", generated_report)
        if json_data:
            logger.info("📊 JSON schema extracted with %d objects", len(json_data.get("objects", [])))

    except Exception as exc:
        logger.exception("Failed to process document: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process the uploaded document.") from exc
    finally:
        file.file.close()

    report_url = f"/reports/{Path(generated_report).name}"

    response = {
        "document_id": document_id,
        "report_filename": Path(generated_report).name,
        "report_url": report_url,
        "instruction": instruction,
    }
    
    # Include JSON data if available (for table generation with pdf-kit)
    if json_data:
        response["json_data"] = json_data
    
    return response


@app.get("/reports/{report_name}")
async def download_report(report_name: str):
    """Serve the generated PDF report for download."""
    report_path = REPORT_DIR / report_name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(report_path, media_type="application/pdf", filename=report_path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)

