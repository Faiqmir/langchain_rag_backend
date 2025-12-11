import os
import time
import json
import logging
import asyncio
import hashlib
import re
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
    timeline_weeks: Optional[int] = Field(default=None, gt=0)  # Added timeline_weeks to the request model

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

USER CONSTRAINTS (MUST FOLLOW EXACTLY):
- Mode: {{ mode }}
- Available Developers: {{ developer_count }}
- Budget: {% if project_budget is not none %}{{ currency }} {{ project_budget|round(2) }}{% else %}to be estimated based on project complexity{% endif %}
- Timeline: {% if timeline_weeks is not none %}EXACTLY {{ timeline_weeks }} weeks{% else %}to be estimated based on project complexity{% endif %} (THIS IS NOT A SUGGESTION - {% if timeline_weeks is not none %}USE THIS EXACT TIMELINE{% else %}Estimate based on project complexity{% endif %})
- Currency: {{ currency }} (ALL monetary values must be in this currency)

CRITICAL REQUIREMENTS:
1. {% if timeline_weeks is not none %}The total project duration MUST be exactly {{ timeline_weeks }} weeks, no more, no less{% else %}Estimate project timeline based on complexity{% endif %}
2. All monetary values must be in {{ currency }} currency
3. {% if project_budget is not none %}Do not use default values - use only the values provided above{% else %}Estimate budget based on project complexity and requirements{% endif %}
4. The "Limitations and Constraints" section must be separate from "Project Scope"

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
Provide a description of the required human resources without using a table format.
List each role, their experience level, count, duration in weeks, hourly rate, and total cost in paragraph format.

### Technical Infrastructure
Servers, cloud services, tools

### Timeline
IMPORTANT: Create a project timeline that {% if timeline_weeks is not none %}MUST span EXACTLY {{ timeline_weeks }} weeks{% else %}should be estimated based on project complexity{% endif %}.
- {% if timeline_weeks is not none %}The total project duration MUST be {{ timeline_weeks }} weeks, no more, no less{% else %}Estimate the total project duration based on complexity{% endif %}
- Break down the timeline into phases that fit within {% if timeline_weeks is not none %}the {{ timeline_weeks }} week timeline{% else %}the estimated timeline{% endif %}.
- Each phase should have specific week ranges that {% if timeline_weeks is not none %}add up to exactly {{ timeline_weeks }} weeks{% else %}sum to the total estimated timeline{% endif %}.
- Example format:
  - Phase 1 (Weeks 1-X): [Description]
  - Phase 2 (Weeks X+1-Y): [Description]
  - Phase 3 (Weeks Y+1-{% if timeline_weeks is not none %}{{ timeline_weeks }}{% else %}Z{% endif %}): [Description]

## 6. Implementation Plan
Phases, milestones, deployment (aligned with the {% if timeline_weeks is not none %}{{ timeline_weeks }} week{% else %}estimated{% endif %} timeline)

## 7. Budget & Financial Analysis
### Cost Breakdown
Create this markdown table with ALL values in {{ currency }}:
| Category | Estimated Cost |
|----------|---------------|

### ROI Analysis
Return on investment

## 8. Costing Analysis
(Detailed costing analysis will be added here)

## 9. Conclusion & Recommendations
Key findings and next steps

## 10. Limitations and Constraints
List items that are explicitly excluded from the project scope and any constraints on the project

Use proper markdown formatting with #, ##, ###, and table syntax.
IMPORTANT: All monetary values should be in {{ currency }}.
{% if timeline_weeks is not none %}CRITICAL: The timeline MUST be exactly {{ timeline_weeks }} weeks as specified by the user.{% else %}Estimate the timeline based on project complexity.{% endif %}
CRITICAL: Do not include any HR tables - only descriptions of HR requirements.
{% if project_budget is not none %}CRITICAL: Use the provided budget of {{ project_budget }} {{ currency }}{% else %}CRITICAL: Estimate the budget based on project complexity and requirements.{% endif %}
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
    "human_resources_description": "Extract the Human Resources description (not a table)",
    "technical_infrastructure": "Extract from Technical Infrastructure",
    "timeline": "Extract the Timeline section with the exact week ranges"
    },
    "implementation_plan": "Extract from Implementation Plan",
    "budget_analysis": {
    "cost_breakdown_table": "Extract the entire markdown table from Cost Breakdown",
    "roi_analysis": "Extract from ROI Analysis"
    },
    "costing_analysis": "Extract from Costing Analysis section",
    "conclusion": "Extract from Conclusion & Recommendations",
    "limitations_constraints": "Extract from Limitations and Constraints section"
},
"metadata": {
    "generated_at": "{{ timestamp }}",
    "mode": "{{ mode }}",
    "developer_count": {{ developer_count }},
    "project_budget": {% if project_budget is not none %}{{ project_budget }}{% else %}null{% endif %},
    "timeline_weeks": {% if timeline_weeks is not none %}{{ timeline_weeks }}{% else %}null{% endif %},
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
        project_budget = kwargs.get('project_budget', None)  # Changed default to None
        timeline_weeks = kwargs.get('timeline_weeks', None)  # Changed default to None
        currency = kwargs.get('currency', 'USD')
        
        # Create budget text based on whether budget is provided
        budget_text = f"{currency} {project_budget:,.2f}" if project_budget is not None else "to be estimated based on project complexity"
        timeline_text = f"EXACTLY {timeline_weeks} weeks" if timeline_weeks is not None else "to be estimated based on project complexity"
        
        return f"""You are an expert Business Analyst. Create a comprehensive project report.

CONTEXT:
{context}

USER CONSTRAINTS (MUST FOLLOW EXACTLY):
- Mode: {mode}
- Available Developers: {developer_count}
- Budget: {budget_text}
- Timeline: {timeline_text} ({"THIS IS NOT A SUGGESTION - USE THIS EXACT TIMELINE" if timeline_weeks is not None else "Estimate based on project complexity"})
- Currency: {currency} (ALL monetary values must be in this currency)

CRITICAL REQUIREMENTS:
1. {"The total project duration MUST be exactly " + str(timeline_weeks) + " weeks, no more, no less" if timeline_weeks is not None else "Estimate project timeline based on complexity"}
2. All monetary values must be in {currency} currency
3. {"Do not use default values - use only the values provided above" if project_budget is not None else "Estimate budget based on project complexity and requirements"}
4. The "Limitations and Constraints" section must be separate from "Project Scope"

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
Provide a description of the required human resources without using a table format.
List each role, their experience level, count, duration in weeks, hourly rate, and total cost in paragraph format.

### Technical Infrastructure
Servers, cloud services, tools

### Timeline
IMPORTANT: Create a project timeline that {"MUST span EXACTLY " + str(timeline_weeks) + " weeks" if timeline_weeks is not None else "should be estimated based on project complexity"}.
- {"The total project duration MUST be " + str(timeline_weeks) + " weeks, no more, no less" if timeline_weeks is not None else "Estimate the total project duration based on complexity"}
- Break down the timeline into phases that fit within {"the " + str(timeline_weeks) + " week timeline" if timeline_weeks is not None else "the estimated timeline"}.
- Each phase should have specific week ranges that {"add up to exactly " + str(timeline_weeks) + " weeks" if timeline_weeks is not None else "sum to the total estimated timeline"}.
- Example format:
  - Phase 1 (Weeks 1-X): [Description]
  - Phase 2 (Weeks X+1-Y): [Description]
  - Phase 3 (Weeks Y+1-{"Z" if timeline_weeks is None else timeline_weeks}): [Description]

## 6. Implementation Plan
Phases, milestones, deployment (aligned with the {"estimated" if timeline_weeks is None else str(timeline_weeks) + " week"} timeline)

## 7. Budget & Financial Analysis
### Cost Breakdown
Create this markdown table with ALL values in {currency}:
| Category | Estimated Cost |
|----------|---------------|

### ROI Analysis
Return on investment

## 8. Costing Analysis
(Detailed costing analysis will be added here)

## 9. Conclusion & Recommendations
Key findings and next steps

## 10. Limitations and Constraints
List items that are explicitly excluded from the project scope and any constraints on the project

Use proper markdown formatting with #, ##, ###, and table syntax.
IMPORTANT: All monetary values should be in {currency}.
{"CRITICAL: The timeline MUST be exactly " + str(timeline_weeks) + " weeks as specified by the user." if timeline_weeks is not None else "Estimate the timeline based on project complexity."}
CRITICAL: Do not include any HR tables - only descriptions of HR requirements.
{"CRITICAL: Use the provided budget of " + str(project_budget) + " " + currency if project_budget is not None else "CRITICAL: Estimate the budget based on project complexity and requirements."}
"""

class FallbackStructuredTemplate:
    """Fallback template for structured data"""
    
    def render(self, **kwargs):
        timestamp = kwargs.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S"))
        mode = kwargs.get('mode', 'master')
        developer_count = kwargs.get('developer_count', 1)
        project_budget = kwargs.get('project_budget', None)  # Changed default to None
        timeline_weeks = kwargs.get('timeline_weeks', None)  # Changed default to None
        currency = kwargs.get('currency', 'USD')
        
        return f"""Based on the report above, extract and format as JSON:

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
    "human_resources_description": "Extract the Human Resources description (not a table)",
    "technical_infrastructure": "Extract from Technical Infrastructure",
    "timeline": "Extract the Timeline section with the exact week ranges"
    },
    "implementation_plan": "Extract from Implementation Plan",
    "budget_analysis": {
    "cost_breakdown_table": "Extract the entire markdown table from Cost Breakdown",
    "roi_analysis": "Extract from ROI Analysis"
    },
    "costing_analysis": "Extract from Costing Analysis section",
    "conclusion": "Extract from Conclusion & Recommendations",
    "limitations_constraints": "Extract from Limitations and Constraints section"
},
"metadata": {
    "generated_at": "{timestamp}",
    "mode": "{mode}",
    "developer_count": {developer_count},
    "project_budget": {project_budget},
    "timeline_weeks": {timeline_weeks},
    "currency": "{currency}"
}
}

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
    
    def _get_cache_key(self, context: str, mode: str, developer_count: int, project_budget: Optional[float], timeline_weeks: Optional[int], currency: str) -> str:
        """Generate a cache key based on input parameters"""
        # Handle None values in cache key
        budget_str = str(project_budget) if project_budget is not None else "none"
        timeline_str = str(timeline_weeks) if timeline_weeks is not None else "none"
        content = f"{context}_{mode}_{developer_count}_{budget_str}_{timeline_str}_{currency}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, context: str, mode: str, developer_count: int, project_budget: Optional[float], timeline_weeks: Optional[int], currency: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        cache_key = self._get_cache_key(context, mode, developer_count, project_budget, timeline_weeks, currency)
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
    
    def set(self, context: str, mode: str, developer_count: int, project_budget: Optional[float], timeline_weeks: Optional[int], currency: str, result: Dict[str, Any]) -> None:
        """Cache a processing result"""
        cache_key = self._get_cache_key(context, mode, developer_count, project_budget, timeline_weeks, currency)
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
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that might contain markdown or other formatting"""
        # First, try to find JSON between triple backticks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If that fails, try to find JSON object directly in the text
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return None
        return None
    
    def _enforce_timeline_constraint(self, markdown_report: str, timeline_weeks: Optional[int]) -> str:
        """Enforce user-specified timeline in the markdown report"""
        if timeline_weeks is None:
            return markdown_report
            
        # Replace any default timeline values with the user-specified timeline
        import re
        
        # Replace patterns like "10 weeks" with the user-specified timeline
        markdown_report = re.sub(r'\d+\s+weeks', f"{timeline_weeks} weeks", markdown_report)
        markdown_report = re.sub(r'\d+\s+week', f"{timeline_weeks} week", markdown_report)
        
        # Specifically replace common default values
        markdown_report = markdown_report.replace("8 weeks", f"{timeline_weeks} weeks")
        markdown_report = markdown_report.replace("10 weeks", f"{timeline_weeks} weeks")
        markdown_report = markdown_report.replace("12 weeks", f"{timeline_weeks} weeks")
        
        return markdown_report
    
    async def generate_dual_output(
        self,
        context: str,
        mode: str,
        developer_count: int,
        project_budget: Optional[float],
        timeline_weeks: Optional[int],
        currency: str = "USD"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate both markdown report AND structured data from LLM
        Returns: (markdown_report, structured_data)
        """
        # Check cache first (only if cache_manager exists)
        if self.cache_manager:
            cached_result = self.cache_manager.get(context, mode, developer_count, project_budget, timeline_weeks, currency)
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
                    timeline_weeks=timeline_weeks,
                    currency=currency
                )
            else:
                # Use fallback template
                markdown_prompt = f"""You are an expert Business Analyst. Create a comprehensive project report.

CONTEXT:
{context}

USER CONSTRAINTS (MUST FOLLOW EXACTLY):
- Mode: {mode}
- Available Developers: {developer_count}
- Budget: {f"{currency} {project_budget:,.2f}" if project_budget is not None else "to be estimated based on project complexity"}
- Timeline: {f"EXACTLY {timeline_weeks} weeks" if timeline_weeks is not None else "to be estimated based on project complexity"} ({"THIS IS NOT A SUGGESTION - USE THIS EXACT TIMELINE" if timeline_weeks is not None else "Estimate based on project complexity"})
- Currency: {currency} (ALL monetary values must be in this currency)

CRITICAL REQUIREMENTS:
1. {"The total project duration MUST be exactly " + str(timeline_weeks) + " weeks, no more, no less" if timeline_weeks is not None else "Estimate project timeline based on complexity"}
2. All monetary values must be in {currency} currency
3. {"Do not use default values - use only the values provided above" if project_budget is not None else "Estimate budget based on project complexity and requirements"}
4. The "Limitations and Constraints" section must be separate from "Project Scope"

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
Provide a description of the required human resources without using a table format.
List each role, their experience level, count, duration in weeks, hourly rate, and total cost in paragraph format.

### Technical Infrastructure
Servers, cloud services, tools

### Timeline
IMPORTANT: Create a project timeline that {"MUST span EXACTLY " + str(timeline_weeks) + " weeks" if timeline_weeks is not None else "should be estimated based on project complexity"}.
- {"The total project duration MUST be " + str(timeline_weeks) + " weeks, no more, no less" if timeline_weeks is not None else "Estimate the total project duration based on complexity"}
- Break down the timeline into phases that fit within {"the " + str(timeline_weeks) + " week timeline" if timeline_weeks is not None else "the estimated timeline"}.
- Each phase should have specific week ranges that {"add up to exactly " + str(timeline_weeks) + " weeks" if timeline_weeks is not None else "sum to the total estimated timeline"}.
- Example format:
  - Phase 1 (Weeks 1-X): [Description]
  - Phase 2 (Weeks X+1-Y): [Description]
  - Phase 3 (Weeks Y+1-{"Z" if timeline_weeks is None else timeline_weeks}): [Description]

## 6. Implementation Plan
Phases, milestones, deployment (aligned with the {"estimated" if timeline_weeks is None else str(timeline_weeks) + " week"} timeline)

## 7. Budget & Financial Analysis
### Cost Breakdown
Create this markdown table with ALL values in {currency}:
| Category | Estimated Cost |
|----------|---------------|

### ROI Analysis
Return on investment

## 8. Costing Analysis
(Detailed costing analysis will be added here)

## 9. Conclusion & Recommendations
Key findings and next steps

## 10. Limitations and Constraints
List items that are explicitly excluded from the project scope and any constraints on the project

Use proper markdown formatting with #, ##, ###, and table syntax.
IMPORTANT: All monetary values should be in {currency}.
{"CRITICAL: The timeline MUST be exactly " + str(timeline_weeks) + " weeks as specified by the user." if timeline_weeks is not None else "Estimate the timeline based on project complexity."}
CRITICAL: Do not include any HR tables - only descriptions of HR requirements.
{"CRITICAL: Use the provided budget of " + str(project_budget) + " " + currency if project_budget is not None else "CRITICAL: Estimate the budget based on project complexity and requirements."}
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
            
            # Enforce user-specified timeline
            markdown_report = self._enforce_timeline_constraint(markdown_report, timeline_weeks)
            
            # Verify timeline is correct
            if timeline_weeks is not None and (f"{timeline_weeks} weeks" not in markdown_report and f"{timeline_weeks} week" not in markdown_report):
                logger.warning(f"⚠️ Timeline may not be set correctly to {timeline_weeks} weeks")
            
            # Second call: Extract structured data from the same report
            if self.template_manager:
                structured_template = self.template_manager.render_template(
                    "structured_data.j2",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    mode=mode,
                    developer_count=developer_count,
                    project_budget=project_budget,
                    timeline_weeks=timeline_weeks,
                    currency=currency
                )
            else:
                # Use fallback template
                structured_template = f"""Based on the report above, extract and format as JSON:

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
    "human_resources_description": "Extract the Human Resources description (not a table)",
    "technical_infrastructure": "Extract from Technical Infrastructure",
    "timeline": "Extract the Timeline section with the exact week ranges"
    },
    "implementation_plan": "Extract from Implementation Plan",
    "budget_analysis": {
    "cost_breakdown_table": "Extract the entire markdown table from Cost Breakdown",
    "roi_analysis": "Extract from ROI Analysis"
    },
    "costing_analysis": "Extract from Costing Analysis section",
    "conclusion": "Extract from Conclusion & Recommendations",
    "limitations_constraints": "Extract from Limitations and Constraints section"
},
"metadata": {
    "generated_at": "{time.strftime("%Y-%m-%d %H:%M:%S")}",
    "mode": "{mode}",
    "developer_count": {developer_count},
    "project_budget": {project_budget},
    "timeline_weeks": {timeline_weeks},
    "currency": "{currency}"
}
}

Extract ALL tables and key information accurately. Preserve markdown table format in the extracted fields.
"""
            
            structured_prompt = f"""
{markdown_report}

{structured_template}
"""
            
            logger.info("🔍 Extracting structured data...")
            structured_response = await asyncio.to_thread(self.chat.invoke, structured_prompt)
            
            # Try to parse the structured response as JSON
            try:
                structured_data = self._extract_json_from_text(structured_response.content.strip())
                if structured_data is None:
                    raise json.JSONDecodeError("Could not extract JSON from response", structured_response.content, 0)
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
                        "timeline_weeks": timeline_weeks,
                        "currency": currency
                    }
                }
            
            # Cache the result
            result = {
                "markdown": markdown_report,
                "structured": structured_data
            }
            if self.cache_manager:
                self.cache_manager.set(context, mode, developer_count, project_budget, timeline_weeks, currency, result)
            
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
        developer_count: Optional[int] = None,  # Changed to Optional
        project_budget: Optional[float] = None,  # Changed to Optional
        timeline_weeks: Optional[int] = None,  # Changed to Optional
        development_scope: str = "local",
        currency: str = "USD",
        project_type: str = "web_app",
        technical_hourly_rate: Optional[float] = None,  # Changed to Optional
        non_technical_hourly_rate: Optional[float] = None,  # Changed to Optional
        instruction: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process document and return dual output for React + backend consumption
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"🚀 Processing document: {input_file}")
        logger.info(f"📅 Using timeline: {'estimated by LLM' if timeline_weeks is None else timeline_weeks} weeks")
        logger.info(f"💰 Using currency: {currency}")
        logger.info(f"💸 Using budget: {'estimated by LLM' if project_budget is None else project_budget}")
        
        # Load document
        context = await self.load_document(input_file)
        
        # Generate dual output
        markdown_report, structured_data = await self.generate_dual_output(
            context=context,
            mode=mode,
            developer_count=developer_count or 1,  # Use 1 as fallback only for internal processing
            project_budget=project_budget,  # Pass None if not provided
            timeline_weeks=timeline_weeks,  # Pass None if not provided
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
                    developer_count=developer_count,  # Pass None if not provided
                    project_budget=project_budget,  # Pass None if not provided
                    technical_hourly_rate=technical_hourly_rate,  # Pass None if not provided
                    non_technical_hourly_rate=non_technical_hourly_rate,  # Pass None if not provided
                    timeline_weeks=timeline_weeks,  # Pass None if not provided
                )
                
                # Add costing data at section 8 (before conclusion)
                if costing_data and "items" in costing_data:
                    # Find the position to insert the costing analysis (before "## 9. Conclusion & Recommendations")
                    conclusion_pos = markdown_report.find("## 9. Conclusion & Recommendations")
                    
                    if conclusion_pos == -1:
                        # Try alternative conclusion headings
                        conclusion_pos = markdown_report.find("## Conclusion")
                    
                    if conclusion_pos == -1:
                        # Try to find any conclusion section
                        conclusion_pos = markdown_report.lower().find("## conclusion")
                    
                    # Create the costing analysis section
                    costing_section = "\n\n## 8. Costing Analysis\n\n"
                    
                    # Create markdown table with user-specific currency
                    costing_section += "### Resource Requirements\n\n"
                    costing_section += "| Role | Quantity | Hourly Rate | Duration | Subtotal |\n"
                    costing_section += "|------|----------|-------------|----------|----------|\n"
                    
                    for item in costing_data["items"]:
                        name = item.get("role", "Unknown")
                        qty = item.get("quantity", 1)
                        rate = item.get("hourly_rate", 0)
                        duration = item.get("duration_weeks", timeline_weeks or 12)
                        subtotal = item.get("subtotal", 0)
                        costing_section += f"| {name} | {qty} | {currency} {rate:.2f} | {duration} weeks | {currency} {subtotal:,.2f} |\n"
                    
                    # Add total with user-specific currency
                    if "total_estimated_cost" in costing_data:
                        costing_section += f"\n**Total Estimated Cost: {currency} {costing_data['total_estimated_cost']:,.2f}**\n"
                    
                    # Insert the costing section before the conclusion if found
                    if conclusion_pos != -1:
                        markdown_report = markdown_report[:conclusion_pos] + costing_section + "\n\n" + markdown_report[conclusion_pos:]
                    else:
                        # If conclusion section not found, append at the end
                        markdown_report += costing_section
                    
            except Exception as e:
                logger.error(f"Error generating costing: {e}")
                costing_data = {
                    "error": "Failed to generate costing",
                    "details": str(e)
                }
        
        # Verify that the timeline is correctly set in the markdown report
        if timeline_weeks is not None and (f"{timeline_weeks} weeks" not in markdown_report and f"{timeline_weeks} week" not in markdown_report):
            logger.warning(f"⚠️ Timeline may not be set correctly to {timeline_weeks} weeks")
            # Try to find and replace any default timeline values
            import re
            # Replace patterns like "10 weeks" with the user-specified timeline
            markdown_report = re.sub(r'\d+\s+weeks', f"{timeline_weeks} weeks", markdown_report)
            markdown_report = re.sub(r'\d+\s+week', f"{timeline_weeks} week", markdown_report)
        
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
                    "currency": currency,
                    "timeline_weeks": timeline_weeks,
                    "project_budget": project_budget
                }
            }
        }
        
        logger.info("✅ Dual output generated successfully")
        return response