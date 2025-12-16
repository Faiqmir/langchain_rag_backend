import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
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

# Pydantic models for chat interaction
class ChatMessage(BaseModel):
    step: str
    question: str
    answer: Any
    timestamp: float = Field(default_factory=time.time)

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage] = []
    confirmed: bool = False
    created_at: float = Field(default_factory=time.time)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    step: str
    answer: Any

class ChatConfirmation(BaseModel):
    session_id: str
    confirmed: bool

class ReportGenerationRequest(BaseModel):
    session_id: str

# In-memory storage for chat sessions (use Redis/DB in production)
chat_sessions: Dict[str, ChatSession] = {}

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

# ==================== Chat Endpoints ====================

@app.post("/chat/start")
async def start_chat_session():
    """Start a new chat session and return session ID"""
    session_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()
    chat_sessions[session_id] = ChatSession(session_id=session_id)
    
    logger.info(f"Started new chat session: {session_id}")
    
    return {
        "success": True,
        "session_id": session_id,
        "message": "Chat session started successfully"
    }

@app.post("/chat/message")
async def save_chat_message(request: ChatRequest = Body(...)):
    """
    Save a chat message from the user.
    Accepts flexible responses for any step.
    """
    # Create session if it doesn't exist
    if not request.session_id or request.session_id not in chat_sessions:
        session_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()
        chat_sessions[session_id] = ChatSession(session_id=session_id)
        logger.info(f"Created new session: {session_id}")
    else:
        session_id = request.session_id
    
    # Define questions for each step
    questions = {
        "input_method": "How would you like to provide your requirements?",
        "project_type": "What type of project is this?",
        "development_scope": "What is your development scope?",
        "currency": "What currency would you like to use?",
        "technical_rate": "What is the technical hourly rate?",
        "non_technical_rate": "What is the non-technical hourly rate?",
        "timeline": "What is your preferred timeline (in weeks)?",
        "budget": "Do you have a fixed budget?",
        "resources": "How many resources do you need?",
    }
    
    # Get the question for this step
    question = questions.get(request.step, f"Response for step: {request.step}")
    
    # Create and save the message
    message = ChatMessage(
        step=request.step,
        question=question,
        answer=request.answer
    )
    
    chat_sessions[session_id].messages.append(message)
    
    logger.info(f"Saved message for session {session_id}: step={request.step}, answer={request.answer}")
    
    return {
        "success": True,
        "session_id": session_id,
        "message": "Message saved successfully",
        "total_messages": len(chat_sessions[session_id].messages)
    }

@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str):
    """Get all messages from a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    
    return {
        "success": True,
        "session_id": session_id,
        "messages": session.messages,
        "confirmed": session.confirmed,
        "created_at": session.created_at
    }

@app.post("/chat/confirm")
async def confirm_generation(confirmation: ChatConfirmation = Body(...)):
    """
    Mark that the user has confirmed they want to generate the report.
    This is called before actually generating the report.
    """
    if confirmation.session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_sessions[confirmation.session_id].confirmed = confirmation.confirmed
    
    logger.info(f"Session {confirmation.session_id} confirmation set to: {confirmation.confirmed}")
    
    return {
        "success": True,
        "session_id": confirmation.session_id,
        "confirmed": confirmation.confirmed,
        "message": "Confirmation saved" if confirmation.confirmed else "Generation cancelled"
    }

@app.post("/chat/generate", response_model=ProcessingResponse)
async def generate_report_from_chat(
    request: ReportGenerationRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Generate report from chat session data.
    Requires user confirmation before generation.
    """
    session_id = request.session_id
    
    # Validate session exists
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    
    # Check if user confirmed
    if not session.confirmed:
        raise HTTPException(
            status_code=400, 
            detail="User has not confirmed report generation. Please confirm first."
        )
    
    # Extract parameters from chat messages
    params = {}
    for msg in session.messages:
        params[msg.step] = msg.answer
    
    logger.info(f"Generating report for session {session_id} with params: {params}")
    
    start_time = time.time()
    document_id = hashlib.md5(f"{time.time()}_{session_id}".encode()).hexdigest()
    
    try:
        # Handle file or text input
        input_mode = params.get("input_method", "file")
        
        if input_mode == "file" or input_mode == "upload":
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
        elif input_mode == "text" or input_mode == "type":
            text_content = params.get("text_content", "")
            if not text_content or not text_content.strip():
                raise HTTPException(status_code=400, detail="No text content provided.")
            
            # Save text content to temporary file
            upload_path = UPLOAD_DIR / f"{document_id}.txt"
            
            with open(upload_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            
            input_path = str(upload_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid input mode. Must be 'file' or 'text'.")
        
        # Parse numeric parameters with flexible handling
        try:
            resources = params.get("resources", "1")
            developer_count = int(resources) if str(resources).strip() else 1
            if developer_count <= 0:
                raise ValueError("Developer count must be positive")
        except ValueError:
            logger.warning(f"Invalid resources value: {resources}, using default 1")
            developer_count = 1

        try:
            budget = params.get("budget", "")
            project_budget = float(budget) if budget and str(budget).strip() and str(budget).lower() not in ["no", "none", "skip"] else None
            if project_budget is not None and project_budget <= 0:
                raise ValueError("Project budget must be positive")
        except ValueError:
            logger.warning(f"Invalid budget value: {budget}, letting LLM estimate budget")
            project_budget = None

        try:
            timeline = params.get("timeline", "")
            weeks = int(timeline) if timeline and str(timeline).strip() else None
            if weeks is not None and weeks <= 0:
                raise ValueError("Timeline weeks must be positive")
        except ValueError:
            logger.warning(f"Invalid timeline value: {timeline}, letting LLM estimate timeline")
            weeks = None

        try:
            tech_rate = params.get("technical_rate", "")
            technical_hourly_rate = float(tech_rate) if tech_rate and str(tech_rate).strip() else None
            if technical_hourly_rate is not None and technical_hourly_rate <= 0:
                raise ValueError("Technical hourly rate must be positive")
        except ValueError:
            logger.warning(f"Invalid technical_rate value: {tech_rate}, letting LLM estimate rate")
            technical_hourly_rate = None

        try:
            non_tech_rate = params.get("non_technical_rate", "")
            non_technical_hourly_rate = float(non_tech_rate) if non_tech_rate and str(non_tech_rate).strip() else None
            if non_technical_hourly_rate is not None and non_technical_hourly_rate <= 0:
                raise ValueError("Non-technical hourly rate must be positive")
        except ValueError:
            logger.warning(f"Invalid non_technical_rate value: {non_tech_rate}, letting LLM estimate rate")
            non_technical_hourly_rate = None

        # Get string parameters with defaults
        development_scope = params.get("development_scope", "local")
        currency = params.get("currency", "PKR")
        project_type = params.get("project_type", "web_app")
        
        # Log the parameters for debugging
        logger.info(f"Processing with parameters: developer_count={developer_count}, project_budget={'estimated by LLM' if project_budget is None else project_budget}, timeline_weeks={'estimated by LLM' if weeks is None else weeks}, currency={currency}")
        
        # Process the document
        result = await document_processor.process_document(
            input_file=input_path,
            mode="master",
            developer_count=developer_count,
            project_budget=project_budget,
            development_scope=development_scope,
            currency=currency,
            project_type=project_type,
            technical_hourly_rate=technical_hourly_rate,
            non_technical_hourly_rate=non_technical_hourly_rate,
            timeline_weeks=weeks,
            instruction=params.get("instruction", "")
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

# ==================== Original Endpoints ====================

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
        mode: Processing mode ('master')
        development_scope: Development scope ('local' or 'international')
        currency: Currency code ('PKR', 'USD', 'EUR')
        project_type: Type of project ('web_app', 'android_app', etc.)
        technical_hourly_rate: Hourly rate for technical resources
        non_technical_hourly_rate: Hourly rate for non-technical resources
        timeline_weeks: Project timeline in weeks
        fixed_budget: Fixed budget for the project
        resources_needed: Number of resources needed
    
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
        
    
# Convert string parameters to appropriate types with proper validation
        try:
            developer_count = int(resources_needed) if resources_needed and resources_needed.strip() else 1
            if developer_count <= 0:
                raise ValueError("Developer count must be positive")
        except ValueError:
            logger.warning(f"Invalid resources_needed value: {resources_needed}, using default 1")
            developer_count = 1

        try:
            # Don't set default budget - let LLM estimate based on complexity
            project_budget = float(fixed_budget) if fixed_budget and fixed_budget.strip() else None
            if project_budget is not None and project_budget <= 0:
                raise ValueError("Project budget must be positive")
        except ValueError:
                logger.warning(f"Invalid fixed_budget value: {fixed_budget}, letting LLM estimate budget")
                project_budget = None

        try:
            weeks = int(timeline_weeks) if timeline_weeks and timeline_weeks.strip() else None
            if weeks is not None and weeks <= 0:
                raise ValueError("Timeline weeks must be positive")
        except ValueError:
            logger.warning(f"Invalid timeline_weeks value: {timeline_weeks}, letting LLM estimate timeline")
            weeks = None

        try:
            tech_rate = float(technical_hourly_rate) if technical_hourly_rate and technical_hourly_rate.strip() else None
            if tech_rate is not None and tech_rate <= 0:
                raise ValueError("Technical hourly rate must be positive")
        except ValueError:
            logger.warning(f"Invalid technical_hourly_rate value: {technical_hourly_rate}, letting LLM estimate rate")
            tech_rate = None

        try:
            non_tech_rate = float(non_technical_hourly_rate) if non_technical_hourly_rate and non_technical_hourly_rate.strip() else None
            if non_tech_rate is not None and non_tech_rate <= 0:
                raise ValueError("Non-technical hourly rate must be positive")
        except ValueError:
            logger.warning(f"Invalid non_technical_hourly_rate value: {non_technical_hourly_rate}, letting LLM estimate rate")
            non_tech_rate = None

        # Log the parameters for debugging
        logger.info(f"Processing with parameters: developer_count={developer_count}, project_budget={'estimated by LLM' if project_budget is None else project_budget}, timeline_weeks={'estimated by LLM' if weeks is None else weeks}, currency={currency}")
        
        # Process the document
        result = await document_processor.process_document(
            input_file=input_path,
            mode=mode,
            developer_count=developer_count,
            project_budget=project_budget,
            development_scope=development_scope,
            currency=currency,
            project_type=project_type,
            technical_hourly_rate=tech_rate,
            non_technical_hourly_rate=non_tech_rate,
            timeline_weeks=weeks,
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