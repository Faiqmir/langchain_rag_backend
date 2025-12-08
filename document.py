import os
import time
import json
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from costing_agent import generate_costing

# ------------------------------
# Setup
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", "deepseek-chat")
TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "templates")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY")

# Create directories if they don't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------
# Enums and Data Models
# ------------------------------

class ProcessingMode(str, Enum):
    MASTER = "master"
    MVP = "mvp"

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"

class ErrorType(str, Enum):
    DOCUMENT_LOADING = "document_loading"
    LLM_GENERATION = "llm_generation"
    JSON_PARSING = "json_parsing"
    CACHE_ERROR = "cache_error"
    TEMPLATE_ERROR = "template_error"

class ProcessingError(Exception):
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.error_type = error_type
        self.message = message
        self.details = details
        super().__init__(message)

class ProcessingRequest(BaseModel):
    input_mode: str = Field(default="file", pattern="^(file|text)$")
    instruction: Optional[str] = Field(default="", max_length=1000)
    mode: ProcessingMode = Field(default=ProcessingMode.MASTER)
    developer_count: int = Field(default=1, gt=0)
    project_budget: float = Field(default=5000.0, gt=0)

class ProcessingResponse(BaseModel):
    success: bool
    document_id: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

# ------------------------------
# Template Management
# ------------------------------

class TemplateManager:
    """Manages external templates for report generation"""
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.env = None
        self._setup_jinja_env()
        self._create_default_templates()
    
    def _setup_jinja_env(self):
        """Setup Jinja2 environment"""
        try:
            import jinja2
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
        except ImportError:
            logger.warning("Jinja2 not installed, using fallback template system")
            self.env = None
    
    def _create_default_templates(self):
        """Create default templates if they don't exist"""
        if self.env is None:
            logger.warning("Jinja2 not available, skipping template creation")
            return
            
        # Markdown report template
        markdown_template_path = self.template_dir / "markdown_report.j2"
        if not markdown_template_path.exists():
            with open(markdown_template_path, "w") as f:
                f.write("""You are an expert Business Analyst. Create a comprehensive project report.

CONTEXT:
{{ context }}

USER CONSTRAINTS:
- Mode: {{ mode }}
- Available Developers: {{ developer_count }}
- Budget: {{ currency }} {{ "%.2f"|format(project_budget) }}

Generate a detailed report in MARKDOWN format with these sections:
# Project Report

## 1. Executive Summary
High-level overview with objectives, benefits, outcomes

## 2. Project Scope and Objectives
Boundaries, objectives, success criteria

## 3. Technical Requirements
### System Architecture
High-level architecture and components

### Technology Stack
Technologies and frameworks

### Performance & Security
Benchmarks and security measures

## 4. Functional Requirements
### Core Features
Main functionalities

### User Roles & UI
User types and interface requirements

## 5. Resource Requirements
### Human Resources
Create this markdown table:
| Role | Experience | Count | Duration(Weeks) | Rate/Hour | Total Cost |
|-------|------------|--------|---------------|------------|------------|

### Technical Infrastructure
Servers, cloud services, tools

### Timeline
Key phases and milestones

## 6. Implementation Plan
Phases, milestones, deployment

## 7. Budget & Financial Analysis
### Cost Breakdown
Create this markdown table:
| Category | Estimated Cost |
|----------|---------------|

### ROI Analysis
Return on investment

## 8. Conclusion & Recommendations
Key findings and next steps

Use proper markdown formatting with #, ##, ###, and table syntax.
IMPORTANT: All monetary values should be in {{ currency }}.
""")
        
        # Structured data template
        structured_template_path = self.template_dir / "structured_data.j2"
        if not structured_template_path.exists():
            with open(structured_template_path, "w") as f:
                f.write("""Based on the report above, extract and format as JSON:

{
"title": "Extract the main project title",
"sections": {
    "executive_summary": "Extract key points from Executive Summary",
    "project_scope": "Extract key points from Project Scope",
    "technical_requirements": {
    "system_architecture": "Extract from System Architecture",
    "technology_stack": "Extract from Technology Stack",
    "performance_security": "Extract from Performance & Security"
    },
    "functional_requirements": {
    "core_features": "Extract from Core Features",
    "user_roles_ui": "Extract from User Roles & UI"
    },
    "resource_requirements": {
    "human_resources_table": "Extract the entire markdown table from Human Resources section",
    "technical_infrastructure": "Extract from Technical Infrastructure",
    "timeline": "Extract from Timeline"
    },
    "implementation_plan": "Extract from Implementation Plan",
    "budget_analysis": {
    "cost_breakdown_table": "Extract the entire markdown table from Cost Breakdown",
    "roi_analysis": "Extract from ROI Analysis"
    },
    "conclusion": "Extract from Conclusion & Recommendations"
},
"metadata": {
    "generated_at": "{{ timestamp }}",
    "mode": "{{ mode }}",
    "developer_count": {{ developer_count }},
    "project_budget": {{ project_budget }},
    "currency": "{{ currency }}"
}
}

Extract ALL tables and key information accurately. Preserve markdown table format in the extracted fields.
""")
    
    def get_template(self, template_name: str):
        """Get a template by name"""
        if self.env is None:
            return self._get_fallback_template(template_name)
        
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            logger.error(f"Template not found: {template_name}")
            raise ProcessingError(
                error_type=ErrorType.TEMPLATE_ERROR,
                message=f"Template {template_name} not found",
                details={"template_name": template_name}
            ) from e
    
    def _get_fallback_template(self, template_name: str):
        """Fallback template system when Jinja2 is not available"""
        if template_name == "markdown_report.j2":
            return FallbackMarkdownTemplate()
        elif template_name == "structured_data.j2":
            return FallbackStructuredTemplate()
        else:
            raise ProcessingError(
                error_type=ErrorType.TEMPLATE_ERROR,
                message=f"Unknown template: {template_name}",
                details={"template_name": template_name}
            )
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """Render a template with the provided context"""
        if self.env is None:
            template = self._get_fallback_template(template_name)
            return template.render(**kwargs)
        
        try:
            template = self.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise ProcessingError(
                error_type=ErrorType.TEMPLATE_ERROR,
                message=f"Error rendering template {template_name}",
                details={"template_name": template_name, "error": str(e)}
            ) from e

class FallbackMarkdownTemplate:
    """Fallback template for markdown reports"""
    
    def render(self, **kwargs):
        context = kwargs.get('context', '')
        mode = kwargs.get('mode', 'master')
        developer_count = kwargs.get('developer_count', 1)
        project_budget = kwargs.get('project_budget', 5000.0)
        currency = kwargs.get('currency', 'USD')
        
        return f"""You are an expert Business Analyst. Create a comprehensive project report.

CONTEXT:
{context}

USER CONSTRAINTS:
- Mode: {mode}
- Available Developers: {developer_count}
- Budget: {currency} {project_budget:,.2f}

Generate a detailed report in MARKDOWN format with these sections:
# Project Report

## 1. Executive Summary
High-level overview with objectives, benefits, outcomes

## 2. Project Scope and Objectives
Boundaries, objectives, success criteria

## 3. Technical Requirements
### System Architecture
High-level architecture and components

### Technology Stack
Technologies and frameworks

### Performance & Security
Benchmarks and security measures

## 4. Functional Requirements
### Core Features
Main functionalities

### User Roles & UI
User types and interface requirements

## 5. Resource Requirements
### Human Resources
Create this markdown table:
| Role | Experience | Count | Duration(Weeks) | Rate/Hour | Total Cost |
|-------|------------|--------|---------------|------------|------------|

### Technical Infrastructure
Servers, cloud services, tools

### Timeline
Key phases and milestones

## 6. Implementation Plan
Phases, milestones, deployment

## 7. Budget & Financial Analysis
### Cost Breakdown
Create this markdown table:
| Category | Estimated Cost |
|----------|---------------|

### ROI Analysis
Return on investment

## 8. Conclusion & Recommendations
Key findings and next steps

Use proper markdown formatting with #, ##, ###, and table syntax.
IMPORTANT: All monetary values should be in {currency}.
"""

class FallbackStructuredTemplate:
    """Fallback template for structured data"""
    
    def render(self, **kwargs):
        timestamp = kwargs.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S"))
        mode = kwargs.get('mode', 'master')
        developer_count = kwargs.get('developer_count', 1)
        project_budget = kwargs.get('project_budget', 5000.0)
        currency = kwargs.get('currency', 'USD')
        
        return f"""Based on the report above, extract and format as JSON:

{{
"title": "Extract the main project title",
"sections": {{
    "executive_summary": "Extract key points from Executive Summary",
    "project_scope": "Extract key points from Project Scope",
    "technical_requirements": {{
    "system_architecture": "Extract from System Architecture",
    "technology_stack": "Extract from Technology Stack",
    "performance_security": "Extract from Performance & Security"
    }},
    "functional_requirements": {{
    "core_features": "Extract from Core Features",
    "user_roles_ui": "Extract from User Roles & UI"
    }},
    "resource_requirements": {{
    "human_resources_table": "Extract the entire markdown table from Human Resources section",
    "technical_infrastructure": "Extract from Technical Infrastructure",
    "timeline": "Extract from Timeline"
    }},
    "implementation_plan": "Extract from Implementation Plan",
    "budget_analysis": {{
    "cost_breakdown_table": "Extract the entire markdown table from Cost Breakdown",
    "roi_analysis": "Extract from ROI Analysis"
    }},
    "conclusion": "Extract from Conclusion & Recommendations"
}},
"metadata": {{
    "generated_at": "{timestamp}",
    "mode": "{mode}",
    "developer_count": {developer_count},
    "project_budget": {project_budget},
    "currency": "{currency}"
}}
}}

Extract ALL tables and key information accurately. Preserve markdown table format in the extracted fields.
"""

# ------------------------------
# Cache Management
# ------------------------------

class CacheManager:
    """Manages caching for document processing results"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, context: str, mode: str, developer_count: int, project_budget: float, currency: str) -> str:
        """Generate a cache key based on input parameters"""
        content = f"{context}_{mode}_{developer_count}_{project_budget}_{currency}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, context: str, mode: str, developer_count: int, project_budget: float, currency: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        cache_key = self._get_cache_key(context, mode, developer_count, project_budget, currency)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
                raise ProcessingError(
                    error_type=ErrorType.CACHE_ERROR,
                    message="Error reading from cache",
                    details={"cache_key": cache_key, "error": str(e)}
                ) from e
        return None
    
    def set(self, context: str, mode: str, developer_count: int, project_budget: float, currency: str, result: Dict[str, Any]) -> None:
        """Cache a processing result"""
        cache_key = self._get_cache_key(context, mode, developer_count, project_budget, currency)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except IOError as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
            raise ProcessingError(
                error_type=ErrorType.CACHE_ERROR,
                message="Error writing to cache",
                details={"cache_key": cache_key, "error": str(e)}
            ) from e

# ------------------------------
# Error Handler
# ------------------------------

class ErrorHandler:
    """Handles different types of processing errors with specific strategies"""
    
    def handle_error(self, error: ProcessingError) -> Dict[str, Any]:
        """Handle different types of errors with specific strategies"""
        if error.error_type == ErrorType.DOCUMENT_LOADING:
            return {
                "success": False,
                "error": error,
                "fallback_data": {
                    "markdown": "# Error Loading Document\n\nWe couldn't load your document. Please try again.",
                    "structured": {
                        "title": "Error",
                        "sections": {
                            "error": "Failed to load document"
                        }
                    }
                }
            }
        elif error.error_type == ErrorType.LLM_GENERATION:
            return {
                "success": False,
                "error": error,
                "fallback_data": {
                    "markdown": "# Error Generating Report\n\nWe encountered an issue generating your report. Please try again later.",
                    "structured": {
                        "title": "Error",
                        "sections": {
                            "error": "Failed to generate report"
                        }
                    }
                }
            }
        elif error.error_type == ErrorType.JSON_PARSING:
            return {
                "success": False,
                "error": error,
                "fallback_data": {
                    "markdown": "# Error Processing Report\n\nWe encountered an issue processing your report. Please try again later.",
                    "structured": {
                        "title": "Error",
                        "sections": {
                            "error": "Failed to process report"
                        }
                    }
                }
            }
        elif error.error_type == ErrorType.CACHE_ERROR:
            return {
                "success": False,
                "error": error,
                "fallback_data": {
                    "markdown": "# Cache Error\n\nWe encountered an issue with our cache. Please try again later.",
                    "structured": {
                        "title": "Error",
                        "sections": {
                            "error": "Cache error"
                        }
                    }
                }
            }
        elif error.error_type == ErrorType.TEMPLATE_ERROR:
            return {
                "success": False,
                "error": error,
                "fallback_data": {
                    "markdown": "# Template Error\n\nWe encountered an issue with our report templates. Please try again later.",
                    "structured": {
                        "title": "Error",
                        "sections": {
                            "error": "Template error"
                        }
                    }
                }
            }
        else:
            return {
                "success": False,
                "error": error,
                "fallback_data": {
                    "markdown": "# Unknown Error\n\nWe encountered an unknown error. Please try again later.",
                    "structured": {
                        "title": "Error",
                        "sections": {
                            "error": "Unknown error"
                        }
                    }
                }
            }

# ------------------------------
# Document Processor
# ------------------------------

class DocumentProcessor:
    """Processes documents and generates dual output"""
    
    def __init__(self):
        try:
            self.template_manager = TemplateManager(TEMPLATE_DIR)
            self.cache_manager = CacheManager(CACHE_DIR)
            self.error_handler = ErrorHandler()
        except Exception as e:
            logger.warning(f"Template initialization failed, using fallback: {e}")
            self.template_manager = None
            self.cache_manager = None
            self.error_handler = ErrorHandler()
        
        # Initialize LLM
        self.chat = ChatDeepSeek(
            model=CHAT_MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.2,
        )
    
    async def load_document(self, file_path: str) -> str:
        """Load document and return text"""
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise ProcessingError(
                error_type=ErrorType.DOCUMENT_LOADING,
                message=f"Failed to load document: {file_path}",
                details={"file_path": file_path, "error": str(e)}
            ) from e
    
    async def generate_dual_output(
        self,
        context: str,
        mode: str,
        developer_count: int,
        project_budget: float,
        currency: str = "USD"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate both markdown report AND structured data from LLM
        Returns: (markdown_report, structured_data)
        """
        # Check cache first (only if cache_manager exists)
        if self.cache_manager:
            cached_result = self.cache_manager.get(context, mode, developer_count, project_budget, currency)
            if cached_result:
                logger.info("Using cached result")
                return cached_result["markdown"], cached_result["structured"]
        
        try:
            # First call: Generate markdown report
            if self.template_manager:
                markdown_prompt = self.template_manager.render_template(
                    "markdown_report.j2",
                    context=context,
                    mode=mode,
                    developer_count=developer_count,
                    project_budget=project_budget,
                    currency=currency
                )
            else:
                # Use fallback template
                markdown_prompt = f"""You are an expert Business Analyst. Create a comprehensive project report.

CONTEXT:
{context}

USER CONSTRAINTS:
- Mode: {mode}
- Available Developers: {developer_count}
- Budget: {currency} {project_budget:,.2f}

Generate a detailed report in MARKDOWN format with these sections:
# Project Report

## 1. Executive Summary
High-level overview with objectives, benefits, outcomes

## 2. Project Scope and Objectives
Boundaries, objectives, success criteria

## 3. Technical Requirements
### System Architecture
High-level architecture and components

### Technology Stack
Technologies and frameworks

### Performance & Security
Benchmarks and security measures

## 4. Functional Requirements
### Core Features
Main functionalities

### User Roles & UI
User types and interface requirements

## 5. Resource Requirements
### Human Resources
Humanresource for the project.

### Technical Infrastructure
Servers, cloud services, tools

### Timeline
Key phases and milestones

## 6. Implementation Plan
Phases, milestones, deployment

## 7. Budget & Financial Analysis
### Cost Breakdown
Create this markdown table:
| Category | Estimated Cost |
|----------|---------------|

### ROI Analysis
Return on investment

## 8. Conclusion & Recommendations
Key findings and next steps

Use proper markdown formatting with #, ##, ###, and table syntax.
IMPORTANT: All monetary values should be in {currency}.
"""
            
            logger.info("🤖 Generating markdown report...")
            markdown_response = await asyncio.to_thread(self.chat.invoke, markdown_prompt)
            markdown_report = markdown_response.content.strip()

            # Remove markdown wrapper if present
            if markdown_report.startswith('```markdown'):
                markdown_report = markdown_report.replace('```markdown', '').strip()
            if markdown_report.startswith('```'):
                markdown_report = markdown_report[3:].strip()
            if markdown_report.endswith('```'):
                markdown_report = markdown_report[:-3].strip()
            
            # Second call: Extract structured data from the same report
            if self.template_manager:
                structured_template = self.template_manager.render_template(
                    "structured_data.j2",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    mode=mode,
                    developer_count=developer_count,
                    project_budget=project_budget,
                    currency=currency
                )
            else:
                # Use fallback template
                structured_template = f"""Based on the report above, extract and format as JSON:

{{
"title": "Extract the main project title",
"sections": {{
    "executive_summary": "Extract key points from Executive Summary",
    "project_scope": "Extract key points from Project Scope",
    "technical_requirements": {{
    "system_architecture": "Extract from System Architecture",
    "technology_stack": "Extract from Technology Stack",
    "performance_security": "Extract from Performance & Security"
    }},
    "functional_requirements": {{
    "core_features": "Extract from Core Features",
    "user_roles_ui": "Extract from User Roles & UI"
    }},
    "resource_requirements": {{
    "human_resources_table": "Extract the Human Resources section",
    "technical_infrastructure": "Extract from Technical Infrastructure",
    "timeline": "Extract from Timeline"
    }},
    "implementation_plan": "Extract from Implementation Plan",
    "budget_analysis": {{
    "cost_breakdown_table": "Extract the entire markdown table from Cost Breakdown",
    "roi_analysis": "Extract from ROI Analysis"
    }},
    "conclusion": "Extract from Conclusion & Recommendations"
}},
"metadata": {{
    "generated_at": "{time.strftime("%Y-%m-%d %H:%M:%S")}",
    "mode": "{mode}",
    "developer_count": {developer_count},
    "project_budget": {project_budget},
    "currency": "{currency}"
}}
}}

Extract ALL tables and key information accurately. Preserve markdown table format in the extracted fields.
"""
            
            structured_prompt = f"""
{markdown_report}

{structured_template}
"""
            
            logger.info("🔍 Extracting structured data...")
            structured_response = await asyncio.to_thread(self.chat.invoke, structured_prompt)
            
            try:
                # Try to parse the structured response as JSON
                structured_data = json.loads(structured_response.content.strip())
                logger.info("✅ Successfully extracted structured data")
            except json.JSONDecodeError:
                # Fallback: create minimal structure
                logger.warning("⚠️ Failed to parse structured JSON, creating fallback")
                structured_data = {
                    "title": "Project Report",
                    "sections": {
                        "executive_summary": "See markdown report",
                        "error": "Failed to extract structured data"
                    },
                    "metadata": {
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": mode,
                        "developer_count": developer_count,
                        "project_budget": project_budget,
                        "currency": currency
                    }
                }
            
            # Cache the result
            result = {
                "markdown": markdown_report,
                "structured": structured_data
            }
            if self.cache_manager:
                self.cache_manager.set(context, mode, developer_count, project_budget, currency, result)
            
            return markdown_report, structured_data
            
        except Exception as e:
            logger.error(f"Error generating dual output: {e}")
            raise ProcessingError(
                error_type=ErrorType.LLM_GENERATION,
                message="Failed to generate report",
                details={"error": str(e)}
            ) from e
    
    async def process_document(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        mode: str = "master",
        developer_count: int = 1,
        project_budget: float = 5000.0,
        development_scope: str = "local",
        currency: str = "USD",
        project_type: str = "web_app",
        technical_hourly_rate: float = 50.0,
        non_technical_hourly_rate: float = 40.0,
        timeline_weeks: int = 12,
        instruction: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process document and return dual output for React + backend consumption
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"🚀 Processing document: {input_file}")
        
        # Load document
        context = await self.load_document(input_file)
        
        # Generate dual output
        markdown_report, structured_data = await self.generate_dual_output(
            context=context,
            mode=mode,
            developer_count=developer_count,
            project_budget=project_budget,
            currency=currency
        )
        
        # Generate costing if in master mode
        costing_data = None
        if mode == "master":
            logger.info("💰 Generating costing analysis...")
            try:
                costing_data = await asyncio.to_thread(
                    generate_costing,
                    context=context,
                    development_scope=development_scope,
                    currency=currency,
                    project_type=project_type,
                    developer_count=developer_count,
                    project_budget=project_budget,
                    technical_hourly_rate=technical_hourly_rate,
                    non_technical_hourly_rate=non_technical_hourly_rate,
                    timeline_weeks=timeline_weeks
                )
                
                # Add costing data as clean JSON without markdown formatting
                if costing_data and "items" in costing_data:
                    markdown_report += "\n\n## 9. Costing Analysis\n\n"
                    
                    # Create markdown table
                    markdown_report += "### Resource Requirements\n\n"
                    markdown_report += "| Role | Quantity | Hourly Rate | Duration | Subtotal |\n"
                    markdown_report += "|------|----------|-------------|----------|----------|\n"
                    
                    for item in costing_data["items"]:
                        name = item.get("role", "Unknown")
                        qty = item.get("quantity", 1)
                        rate = item.get("hourly_rate", 0)
                        duration = item.get("duration_weeks", 12)
                        subtotal = item.get("subtotal", 0)
                        markdown_report += f"| {name} | {qty} | {currency} {rate:.2f} | {duration} weeks | {currency} {subtotal:,.2f} |\n"
                    
                    # Add total
                    if "total_estimated_cost" in costing_data:
                        markdown_report += f"\n**Total Estimated Cost: {currency} {costing_data['total_estimated_cost']:,.2f}**\n"
                    
            except Exception as e:
                logger.error(f"Error generating costing: {e}")
                costing_data = {
                    "error": "Failed to generate costing",
                    "details": str(e)
                }
        
        # Prepare response
        response = {
            "success": True,
            "data": {
                # Raw markdown for immediate React display
                "markdown": markdown_report,
                
                # Structured data for backend consumption
                "structured": structured_data,
                
                # Costing data
                "costing": costing_data,
                
                # File info
                "input_file": input_file,
                "output_file": output_file or "report.pdf",
                
                # Metadata
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time": "fast",
                    "mode": mode,
                    "currency": currency
                }
            }
        }
        
        logger.info("✅ Dual output generated successfully")
        return response