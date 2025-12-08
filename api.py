import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from document import DocumentProcessor, ProcessingError, ProcessingMode, ProcessingRequest, ProcessingResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="Upload client requirement documents, process them into business reports, and retrieve generated PDFs.",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5176", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
document_processor = DocumentProcessor()

@app.post("/process", response_model=ProcessingResponse)
async def process_document_api(
    file: Optional[UploadFile] = File(None),
    text_content: str = Form(""),
    input_mode: str = Form("file"),
    instruction: str = Form(""),
    mode: str = Form("master"),
    development_scope: str = Form("local"),
    currency: str = Form("PKR"),
    project_type: str = Form("web_app"),
    technical_hourly_rate: str = Form(""),
    non_technical_hourly_rate: str = Form(""),
    timeline_weeks: str = Form(""),
    fixed_budget: str = Form(""),
    resources_needed: str = Form(""),
):
    """
    Accept a document upload or text input, run processing pipeline, and return a download URL.
    
    Args:
        file: Optional uploaded file (.pdf or .txt)
        text_content: Text content as alternative to file upload
        input_mode: Either 'file' or 'text'
        instruction: Optional instruction for processing
        mode: Processing mode ('master' or 'mvp')
        developer_count: Number of developers available
        project_budget: Available budget
    
    Returns:
        ProcessingResponse with document details and dual output
    """
    start_time = time.time()
    document_id = hashlib.md5(f"{time.time()}_{input_mode}".encode()).hexdigest()
    
    try:
        # Handle input validation
        if input_mode == "file":
            if not file or not file.filename:
                raise HTTPException(status_code=400, detail="No file uploaded.")
            
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                raise HTTPException(
                    status_code=400, 
                    detail="Only .pdf or .txt files are supported."
                )
            
            # Save uploaded file
            upload_path = UPLOAD_DIR / f"{document_id}{suffix}"
            
            content = await file.read()
            with open(upload_path, "wb") as buffer:
                buffer.write(content)
            
            input_path = str(upload_path)
        elif input_mode == "text":
            if not text_content or not text_content.strip():
                raise HTTPException(status_code=400, detail="No text content provided.")
            
            # Save text content to temporary file
            upload_path = UPLOAD_DIR / f"{document_id}.txt"
            
            with open(upload_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            
            input_path = str(upload_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid input mode. Must be 'file' or 'text'.")
        
        # Convert string parameters to appropriate types
        developer_count = int(resources_needed) if resources_needed and resources_needed.strip() else 1
        project_budget = float(fixed_budget) if fixed_budget and fixed_budget.strip() else 5000.0
        
        # Process the document
        result = await document_processor.process_document(
            input_file=input_path,
            mode=mode,
            developer_count=developer_count,
            project_budget=project_budget,
            development_scope=development_scope,
            currency=currency,
            project_type=project_type,
            technical_hourly_rate=float(technical_hourly_rate) if technical_hourly_rate and technical_hourly_rate.strip() else 50.0,
            non_technical_hourly_rate=float(non_technical_hourly_rate) if non_technical_hourly_rate and non_technical_hourly_rate.strip() else 40.0,
            timeline_weeks=int(timeline_weeks) if timeline_weeks and timeline_weeks.strip() else 12,
            instruction=instruction
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return ProcessingResponse(
            success=True,
            document_id=document_id,
            data=result["data"],
            processing_time=round(processing_time, 2)
        )
        
    except ProcessingError as e:
        # Handle processing errors with specific strategies
        error_response = document_processor.error_handler.handle_error(e)
        processing_time = time.time() - start_time
        
        return ProcessingResponse(
            success=False,
            document_id=document_id,
            error={
                "error_type": e.error_type,
                "message": e.message,
                "details": e.details
            },
            processing_time=round(processing_time, 2)
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error processing document: {e}")
        processing_time = time.time() - start_time
        
        return ProcessingResponse(
            success=False,
            document_id=document_id,
            error={
                "error_type": "unknown",
                "message": "An unknown error occurred",
                "details": {"error": str(e)}
            },
            processing_time=round(processing_time, 2)
        )
@app.get("/reports/{report_id}.pdf")
async def get_report(report_id: str):
    """Serve the generated PDF report"""
    report_path = os.path.join("reports", f"{report_id}.pdf")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
        
    return FileResponse(report_path, media_type="application/pdf")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "document-processing-api"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)