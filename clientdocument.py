import os
import time
import json
import logging
import re
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from textwrap import wrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# LangChain modules
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate

# Text cleaning utilities
from text_cleaning import normalize_text, early_normalize_chunk
from costing_agent import generate_costing

# ------------------------------
# Setup Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------------------
# Environment Setup
# ------------------------------
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", "deepseek-chat")

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "./chroma_db"
DEFAULT_OUTPUT_PDF = "business_report4.pdf"

# ------------------------------
# Structured JSON report schema
# ------------------------------
REPORT_JSON_SCHEMA_TEMPLATE = """
{
  "title": "Project Analytical Report: [Project Name]",
  "sections": [
    {
      "section_number": 1,
      "heading": "Introduction",
      "description": "Precise, accurate introduction (2–4 sentences) conveying the project's main purpose, functionality, and scope.",
      "subsections": []
    },
    {
      "section_number": 2,
      "heading": "Project Scope, Purpose, and Deliverables",
      "description": "Overall description of scope, purpose, and deliverables.",
      "subsections": [
        {
          "label": "Purpose",
          "content": "Detailed purpose description."
        },
        {
          "label": "Scope",
          "content": "Detailed scope description."
        },
        {
          "label": "Deliverables",
          "content": "List of deliverables, one per line or bullet point."
        }
      ]
    },
    {
      "section_number": 3,
      "heading": "Functional Modules and Their Roles",
      "description": "Overview of functional modules and their responsibilities.",
      "subsections": [
        {
          "label": "Module Name",
          "content": "Description of the module's role in the system."
        }
      ]
    },
    {
      "section_number": 4,
      "heading": "Detailed Resources",
      "description": "Overview of resources required.",
      "subsections": [
        {
          "label": "Human Resources",
          "content": "Description of human resources needed.",
          "table_data": {
            "columns": ["Role", "Number of People", "Estimated Cost per Person", "Total Cost"],
            "rows": [
              {
                "Role": "Role Name",
                "Number of People": "#",
                "Estimated Cost per Person": "Cost",
                "Total Cost": "Total"
              }
            ]
          }
        },
        {
          "label": "Technical Resources",
          "content": "Description of technical resources."
        },
        {
          "label": "Data Resources",
          "content": "Description of data resources."
        }
      ]
    },
    {
      "section_number": 5,
      "heading": "Costing Insights",
      "description": "Cost analysis and insights based on explicit or inferred data.",
      "subsections": []
    },
    {
      "section_number": 6,
      "heading": "Conclusion",
      "description": "Concise, actionable conclusion summarizing key points.",
      "subsections": []
    }
  ]
}
"""

if not DEEPSEEK_API_KEY:
    raise RuntimeError("❌ Missing DEEPSEEK_API_KEY in environment or .env file.")

# ------------------------------
# Master Prompt (JSON-structured)
# ------------------------------
MASTER_TEMPLATE = """
You are an expert Business Analyst and Technical Product Manager delivering a project to a client.

Your task is to create a complete, structured, client-ready analytical report based only on the given CONTEXT. Treat this as a professional project analysis — NOT a plagiarism report.
Do not write anything like "Of course. Here is the complete, structured, client-ready analytical report based on the provided context." or anything similar.Give conclusion and the costing insights of the project.

──────────────────────────────
CONTEXT:
{context}
──────────────────────────────

OBJECTIVE:
Interpret and transform the CONTEXT into a professional business report following the JSON schema format given below.

IMPORTANT HARD RULES:
- Return ONLY a single valid JSON object.
- DO NOT include markdown code fences like ```json or ``` anywhere.
- DO NOT include prose, explanations, or any text outside the JSON object.
- All strings must be valid JSON strings (escape quotes, newlines, etc.).

USE THIS JSON FORMAT AS A GUIDE (you must adapt the values based on the CONTEXT):

{schema}
"""

USER_COMMAND_TEMPLATE = """

──────────────────────────────
USER COMMAND:
{instruction}
──────────────────────────────


"""

MASTER_PROMPT = PromptTemplate(
    template=MASTER_TEMPLATE,
    input_variables=["context", "schema"],
)

# ------------------------------
# Utility Functions
# ------------------------------
def load_document(file_path: str):
    """Load PDF or text file."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()
    logger.info(f"📄 Loaded {len(docs)} document(s) from {file_path}")
    return docs

def split_text(text: str):
    """Split raw text into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    logger.info(f"🔹 Split report into {len(chunks)} chunks for embedding.")
    return chunks

def initialize_or_load_chroma():
    """Load Chroma if exists, else initialize empty one."""
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        logger.info("📦 Loading existing Chroma vector database...")
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        logger.info("🆕 Initializing new Chroma vector database...")
        return Chroma.from_texts([""], embeddings, persist_directory=PERSIST_DIRECTORY)

def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text, if present."""
    # Try to find JSON object in the text using regex
    json_pattern = r'\{[^{}]*"objects"\s*:\s*\[.*?\]\s*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        try:
            json_str = match.group(0)
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            logger.warning("⚠️ Found JSON-like text but failed to parse it.")
            return None
    
    # Also try to find any JSON object with balanced braces
    try:
        start_idx = text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                json_data = json.loads(json_str)
                if "objects" in json_data:
                    return json_data
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def generate_report(context: str, instruction: str | None = None) -> Tuple[str, dict | None]:
    """
    Legacy text-based report generation (kept for fallback when JSON mode fails).

    Returns:
        Tuple of (report_text, json_data) where:
        - report_text: The full report text
        - json_data: Extracted JSON schema (dict) if found, None otherwise
    """
    chat = ChatDeepSeek(
        model=CHAT_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2,
    )

    # Simpler legacy prompt that asks for a narrative report
    prompt = (
        "You are an expert Business Analyst. Write a detailed analytical report based ONLY on the following CONTEXT.\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Structure the report with a clear title, numbered headings, and descriptive paragraphs. "
        "Do not include any JSON or code fences."
    )
    if instruction:
        prompt = f"{prompt}\n\nAdditional user instruction: {instruction.strip()}"

    logger.info("🤖 Generating legacy analytical report via DeepSeek...")
    response = chat.invoke(prompt)
    report_text = response.content.strip()

    # Apply early normalization for basic cleaning
    report_text = early_normalize_chunk(report_text)

    # Extract JSON (if any) from the report text
    json_data = extract_json_from_text(report_text)

    if json_data:
        logger.info("✅ Legacy JSON schema extracted successfully.")
        # Print JSON for testing purposes
        print("\n" + "=" * 80)
        print("📊 EXTRACTED LEGACY JSON DATA (for testing):")
        print("=" * 80)
        print(json.dumps(json_data, indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")
        # Remove JSON from report text for cleaner output
        json_pattern = r'\{[^{}]*"objects"\s*:\s*\[.*?\]\s*\}'
        report_text = re.sub(json_pattern, "", report_text, flags=re.DOTALL).strip()
    else:
        logger.info("ℹ️ No legacy JSON schema found in response.")
        print("\n⚠️  No legacy JSON data extracted from the response.\n")

    # Apply final text normalization for clean output
    report_text = normalize_text(report_text)
    logger.info("🧹 Applied text cleaning to legacy report output.")

    logger.info("✅ Legacy report generation complete.")
    return report_text, json_data


def generate_report_json(context: str, instruction: str | None = None) -> dict | None:
    """
    Generate a structured JSON report using the JSON schema prompt.
    Returns the parsed JSON dict, or None if parsing fails.
    """
    chat = ChatDeepSeek(
        model=CHAT_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2,
    )

    prompt = MASTER_PROMPT.format(
        context=context,
        schema=REPORT_JSON_SCHEMA_TEMPLATE,
    )
    if instruction:
        prompt = f"{prompt}\n{USER_COMMAND_TEMPLATE.format(instruction=instruction.strip())}"

    logger.info("🤖 Generating JSON-structured analytical report via DeepSeek...")
    response = chat.invoke(prompt)
    raw_text = response.content.strip()

    # Remove possible code fences
    raw_text = re.sub(r"```json\s*\n?", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"```\s*\n?", "", raw_text)
    raw_text = re.sub(r"```[^`]*?```", "", raw_text, flags=re.DOTALL)
    
    # Try to locate JSON object
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        logger.warning("⚠️ No JSON object found in model response.")
        print("\n⚠️ No JSON object found in model response.\n")
        return None

    json_str = match.group(0)
    try:
        report_json = json.loads(json_str)
        logger.info("✅ Parsed JSON report successfully.")
        print("\n" + "=" * 80)
        print("📊 EXTRACTED REPORT JSON (for testing):")
        print("=" * 80)
        print(json.dumps(report_json, indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")
        return report_json
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Failed to parse JSON report: {e}")
        print(f"\n⚠️ JSON parsing error: {e}\n")
        return None

def detect_heading(line: str) -> bool:
    """
    Detect if a line is a heading based on patterns:
    - Numbered headings like "1. Introduction" or "1.1 Section"
    - Lines that are short and end with colon (like "**Purpose:**")
    - Lines wrapped entirely in **text**
    - Lines starting with # markdown heading
    """
    line_clean = line.strip()
    
    # Numbered headings: "1. ", "1.1. ", "2. ", etc.
    if re.match(r'^\d+(\.\d+)*\.\s+[A-Z]', line_clean):
        return True
    
    # Lines wrapped entirely in **text** (likely headings)
    if re.match(r'^\*\*[^*]+\*\*\s*$', line_clean):
        return True
    
    # Lines with pattern "**Label:**" at start (heading label)
    if re.match(r'^\*\*[^*]+:\*\*\s*$', line_clean):
        return True
    
    # Markdown headings with #
    if re.match(r'^#{1,6}\s+', line_clean):
        return True
    
    # Short lines (likely headings) that are mostly uppercase or have key words
    if len(line_clean) < 100 and (line_clean.isupper() or 
                                   re.match(r'^(Project|Introduction|Scope|Purpose|Deliverables|Modules|Resources|Conclusion)', line_clean, re.IGNORECASE)):
        return True
    
    return False


def convert_markdown_to_reportlab_html(text: str) -> Tuple[str, list]:
    """
    Convert markdown formatting to reportlab HTML tags.
    Returns (cleaned_text, paragraphs_info) where paragraphs_info contains
    metadata about which paragraphs are headings, titles, etc.
    """
    if not text:
        return "", []
    
    paragraphs_info = []
    
    # Remove code fences like ```json or ``` (handle various formats)
    text = re.sub(r'```\w*\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = re.sub(r'```[^`]*?```', '', text, flags=re.DOTALL)
    
    # Split into lines for processing
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            processed_lines.append('')
            continue
        
        # Check if this line starts with a numbered heading pattern like "1. " or "1.1. "
        heading_match = re.match(r'^(\d+(\.\d+)*\.\s+)(.+)', line_stripped)
        if heading_match:
            number_part = heading_match.group(1)  # "1. " or "1.1. "
            rest_of_line = heading_match.group(3)  # Everything after the number
            
            # Check if rest starts with a heading keyword or is short (likely a heading)
            # If it's long or starts with lowercase, it might be description starting with heading
            rest_clean = rest_of_line.strip()
            
            # Split on first sentence or colon if it exists
            # For numbered headings like "1. Introduction", treat as heading
            # For lines like "1. Introduction Some text", split at reasonable point
            heading_text = None
            description_text = None
            
            # Try to detect if rest contains a clear heading followed by description
            # Pattern: Heading (short, may have colons) followed by description
            split_match = re.match(r'^([^.]{2,60}?)(?:\s{2,}|\.\s+)(.+)', rest_clean)
            if split_match and len(split_match.group(1)) < 80:
                # Likely heading + description
                heading_text = number_part + split_match.group(1).strip()
                description_text = split_match.group(2).strip()
            else:
                # Treat entire rest as heading if it's short, otherwise as description
                if len(rest_clean) < 100 and not rest_clean[0].islower():
                    heading_text = number_part + rest_clean
                else:
                    # Long text or starts with lowercase - might be description
                    # But numbered items should be headings, so split at first period or reasonable point
                    first_part = re.match(r'^(.{1,60}?)(?:\s{2,}|\.\s+)(.+)', rest_clean)
                    if first_part:
                        heading_text = number_part + first_part.group(1).strip()
                        description_text = first_part.group(2).strip()
                    else:
                        heading_text = number_part + rest_clean
            
            # Clean heading text
            if heading_text:
                heading_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', heading_text)
                heading_text = re.sub(r'\*+', '', heading_text).strip()
                
                processed_lines.append(heading_text)
                paragraphs_info.append({
                    'text': heading_text,
                    'is_heading': True,
                    'is_title': False
                })
            
            # Clean and add description if exists
            if description_text:
                cleaned_desc = re.sub(r'\*\*([^*:]+):\*\*\s*', r'<b>\1:</b> ', description_text)
                cleaned_desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_desc)
                cleaned_desc = re.sub(r'\*+', '', cleaned_desc).strip()
                
                if cleaned_desc:
                    processed_lines.append(cleaned_desc)
                    paragraphs_info.append({
                        'text': cleaned_desc,
                        'is_heading': False,
                        'is_title': False
                    })
        elif detect_heading(line_stripped):
            # Other types of headings (markdown #, wrapped in **, etc.)
            # Save previous paragraph if exists
            heading_text = line_stripped
            
            # Remove markdown heading markers
            heading_text = re.sub(r'^#{1,6}\s+', '', heading_text)
            
            # Remove ** around text
            heading_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', heading_text)
            
            # Remove any remaining asterisks
            heading_text = re.sub(r'\*+', '', heading_text).strip()
            
            processed_lines.append(heading_text)
            paragraphs_info.append({
                'text': heading_text,
                'is_heading': True,
                'is_title': False
            })
        else:
            # Regular description line - remove all markdown and asterisks
            cleaned_line = re.sub(r'\*\*([^*:]+):\*\*\s*', r'<b>\1:</b> ', line_stripped)
            cleaned_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_line)
            cleaned_line = re.sub(r'\*+', '', cleaned_line).strip()
            
            if cleaned_line:
                processed_lines.append(cleaned_line)
                paragraphs_info.append({
                    'text': cleaned_line,
                    'is_heading': False,
                    'is_title': False
                })
    
    # Join back - each paragraph on separate line
    cleaned_text = '\n\n'.join([line for line in processed_lines if line])
    
    # Detect title (first paragraph that looks like a title)
    if paragraphs_info:
        first_para = paragraphs_info[0]
        # Check if first paragraph looks like a title
        first_text = first_para['text'].lower()
        if ('report' in first_text or 'project' in first_text or 
            'analytical' in first_text or len(first_para['text']) < 150):
            paragraphs_info[0]['is_title'] = True
    
    return cleaned_text, paragraphs_info


def create_table_from_json(json_data: dict) -> Table | None:
    """Create a reportlab Table from JSON data."""
    if not json_data or "objects" not in json_data or not json_data["objects"]:
        return None
    
    objects = json_data["objects"]
    if not objects:
        return None
    
    # Extract column headers from the first object
    headers = list(objects[0].keys())
    
    # Build table data: headers + rows
    table_data = [headers]
    for obj in objects:
        row = [str(obj.get(header, "")) for header in headers]
        table_data.append(row)
    
    # Create table
    table = Table(table_data)
    
    # Style the table
    style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Data rows
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ])
    
    table.setStyle(style)
    return table


def extract_table_objects_from_json_report(report_json: dict) -> dict | None:
    """Extract human-resource-style table rows from structured JSON report."""
    if not isinstance(report_json, dict):
        return None

    for section in report_json.get("sections", []):
        subsections = section.get("subsections", []) or []
        for subsection in subsections:
            table_data = subsection.get("table_data")
            if not table_data:
                continue
            rows = table_data.get("rows") or []
            if rows:
                # Match legacy structure {"objects": [...]}
                return {"objects": rows}
    return None


def write_pdf(report_text: str, output_file: str, json_data: dict | None = None):
    """Write report text to PDF with optional JSON table."""
    doc = SimpleDocTemplate(output_file, pagesize=A4,
                           rightMargin=0.8*inch, leftMargin=0.8*inch,
                           topMargin=0.8*inch, bottomMargin=0.8*inch)
    
    # Convert markdown to reportlab HTML formatting
    cleaned_text, paragraphs_info = convert_markdown_to_reportlab_html(report_text)
    logger.info("🎨 Converted markdown formatting to reportlab HTML tags.")
    
    # Get styles
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    # Create title style (font size 18, bold)
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceAfter=18,
        alignment=1  # Center alignment
    )
    
    # Create heading style (bold, medium font)
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        fontName='Helvetica-Bold',
        spaceAfter=10,
        spaceBefore=14,
        textColor=colors.black
    )
    
    # Build story (content elements)
    story = []
    
    # Find insertion point for table (after sections mentioning resources/human resources)
    insert_table_at = -1
    for i, para_info in enumerate(paragraphs_info):
        para_lower = para_info['text'].lower()
        if 'human resources' in para_lower or 'resources' in para_lower:
            if i + 1 < len(paragraphs_info):
                insert_table_at = i + 1
                break
    
    # If not found, insert before conclusion or at the end
    if insert_table_at == -1:
        for i, para_info in enumerate(paragraphs_info):
            para_lower = para_info['text'].lower()
            if 'conclusion' in para_lower:
                insert_table_at = i
                break
        if insert_table_at == -1:
            insert_table_at = len(paragraphs_info) - 1
    
    # Add paragraphs and table
    table_inserted = False
    for i, para_info in enumerate(paragraphs_info):
        para_text = para_info['text'].strip()
        is_title = para_info.get('is_title', False)
        is_heading = para_info.get('is_heading', False)
        
        if not para_text:
            continue
        
        # Render based on paragraph type
        if is_title:
            # Title - bold, large font, on separate line
            story.append(Paragraph(f"<b>{para_text}</b>", title_style))
            story.append(Spacer(1, 18))
        elif is_heading:
            # Heading - bold, medium font
            story.append(Paragraph(f"<b>{para_text}</b>", heading_style))
            story.append(Spacer(1, 8))
        else:
            # Regular description - normal font, no bold
            story.append(Paragraph(para_text, normal_style))
            story.append(Spacer(1, 12))
        
        # Insert table after appropriate paragraph (only once)
        if json_data and i == insert_table_at and not table_inserted:
            table = create_table_from_json(json_data)
            if table:
                # Create heading style
                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading3'],
                    fontSize=14,
                    fontName='Helvetica-Bold',
                    spaceAfter=6
                )
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Human Resources Breakdown:</b>", heading_style))
                story.append(Spacer(1, 6))
                story.append(table)
                story.append(Spacer(1, 12))
                table_inserted = True
                logger.info("📊 Table inserted in PDF at position %d", i)
    
    # Build PDF
    doc.build(story)
    logger.info(f"📘 PDF saved to {output_file}")


def write_pdf_from_json(report_json: dict, output_file: str, costing_json: dict | None = None):
    """
    Generate a nicely formatted PDF from structured JSON report data.
    Uses title, sections, subsections, and optional table_data.
    If costing_json is provided, also render a "Costing Estimate" section at the end.
    """
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "JsonTitle",
        parent=styles["Title"],
        fontSize=18,
        fontName="Helvetica-Bold",
        textColor=colors.black,
        spaceAfter=18,
        alignment=1,  # center
    )

    heading_style = ParagraphStyle(
        "JsonHeading",
        parent=styles["Heading2"],
        fontSize=14,
        fontName="Helvetica-Bold",
        spaceAfter=10,
        spaceBefore=14,
        textColor=colors.black,
    )

    subsection_label_style = ParagraphStyle(
        "JsonSubLabel",
        parent=styles["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )

    normal_style = styles["Normal"]
    normal_style.fontSize = 11
    normal_style.leading = 14

    story: list = []

    # Title
    title = report_json.get("title") or "Project Analytical Report"
    story.append(Paragraph(f"<b>{title}</b>", title_style))
    story.append(Spacer(1, 18))

    sections = report_json.get("sections", [])
    for section in sections:
        sec_no = section.get("section_number")
        heading = (section.get("heading") or "").strip()
        if sec_no:
            heading_text = f"{sec_no}. {heading}"
        else:
            heading_text = heading

        if heading_text:
            story.append(Paragraph(f"<b>{heading_text}</b>", heading_style))
            story.append(Spacer(1, 8))

        desc = (section.get("description") or "").strip()
        if desc:
            story.append(Paragraph(desc, normal_style))
            story.append(Spacer(1, 12))

        for subsection in section.get("subsections", []):
            label = (subsection.get("label") or "").strip()
            content = (subsection.get("content") or "").strip()
            table_data = subsection.get("table_data")

            if label:
                story.append(Paragraph(f"<b>{label}:</b>", subsection_label_style))

            if content:
                story.append(Paragraph(content, normal_style))
                story.append(Spacer(1, 8))

            if table_data:
                cols = table_data.get("columns") or []
                rows = table_data.get("rows") or []
                if cols and rows:
                    data = [cols]
                    for row in rows:
                        data.append([str(row.get(col, "")) for col in cols])

                    table = Table(data)
                    table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 12),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                                ("FONTSIZE", (0, 1), (-1, -1), 10),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 1), (-1, -1), 6),
                                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ]
                        )
                    )
                    story.append(table)
                    story.append(Spacer(1, 12))

        story.append(Spacer(1, 12))

    # Optional costing section appended after main report
    if isinstance(costing_json, dict) and costing_json:
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Costing Estimate</b>", heading_style))
        story.append(Spacer(1, 6))

        scope_raw = str(costing_json.get("development_scope", ""))
        scope_label = "International team" if scope_raw.lower() == "international" else "Local team"
        project_type = str(costing_json.get("project_type", "")).replace("_", " ")

        scope_line = f"Scope: <b>{scope_label}</b>  |  Project type: <b>{project_type}</b>"
        story.append(Paragraph(scope_line, normal_style))
        story.append(Spacer(1, 4))

        total_cost = costing_json.get("total_estimated_cost")
        currency = str(costing_json.get("currency", ""))
        team_size = costing_json.get("developer_count")
        duration = costing_json.get("assumed_duration_months")

        if total_cost is not None:
            meta_parts: list[str] = []
            if team_size is not None:
                meta_parts.append(f"team size: {team_size}")
            if duration is not None:
                meta_parts.append(f"duration: {duration} months")
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
            total_line = f"Estimated total: <b>{total_cost} {currency}</b>{meta_suffix}"
            story.append(Paragraph(total_line, normal_style))
            story.append(Spacer(1, 6))

        summary_text = costing_json.get("natural_language_summary")
        if isinstance(summary_text, str) and summary_text.strip():
            story.append(Paragraph(summary_text.strip(), normal_style))
            story.append(Spacer(1, 8))

        items = costing_json.get("items") or []
        if isinstance(items, list) and items:
            table_data: list[list[str]] = [["Role", "Qty", "Monthly rate", "Months", "Subtotal"]]
            for item in items:
                role = str(item.get("role", ""))
                qty = str(item.get("quantity", ""))
                monthly_rate = item.get("monthly_rate")
                duration_months = item.get("duration_months")
                subtotal = item.get("subtotal")

                mr_str = f"{monthly_rate} {currency}" if monthly_rate is not None else ""
                sub_str = f"{subtotal} {currency}" if subtotal is not None else ""

                table_data.append([
                    role,
                    qty,
                    mr_str,
                    str(duration_months) if duration_months is not None else "",
                    sub_str,
                ])

            table = Table(table_data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 11),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 12))

    doc.build(story)
    logger.info(f"📘 JSON-based PDF saved to {output_file}")


# ------------------------------
# Core Processing Entry Point
# ------------------------------
def augment_context_with_constraints(
    base_context: str,
    mode: str,
    developer_count: int,
    project_budget: float,
) -> str:
    """Append user-provided knobs (mode, developer count, budget) to the context."""
    constraints = (
        "\n\n──────────────────────────────\n"
        "PROJECT CONSTRAINTS PROVIDED BY USER:\n"
        f"- Requested Mode: {mode}\n"
        f"- Available Developers: {developer_count}\n"
        f"- Budget: ${project_budget:,.2f}\n"
        "Use these constraints when estimating resources, costs, timelines, and recommendations.\n"
        "──────────────────────────────\n"
    )
    return f"{base_context}{constraints}"


def process_document(
    input_file: str,
    output_file: str | None = None,
    instruction: str | None = None,
    mode: str = "master",
    developer_count: int = 1,
    project_budget: float = 5000.0,
    development_scope: str = "local",
    currency: str = "PKR",
    project_type: str = "web_app",
) -> Tuple[str, dict | None, dict | None]:
    """
    Run the end-to-end pipeline: load requirements, generate report, persist embeddings.

    Args:
        input_file: Path to the source requirements document (PDF or text).
        output_file: Optional path for the generated PDF. Defaults to DEFAULT_OUTPUT_PDF.
        instruction: Optional custom instruction from user.

    Returns:
        Tuple of (pdf_path, json_data, costing_json) where:
        - pdf_path: The path to the generated PDF report
        - json_data: Extracted JSON schema (dict) if found, None otherwise (can be used with pdf-kit for tables)
        - costing_json: Structured costing information, if available
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path = Path(output_file or DEFAULT_OUTPUT_PDF).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"🚀 Starting processing for {input_file} -> {output_path}")

    # Step 1: Load or create Chroma DB (ensure availability)
    vector_store = initialize_or_load_chroma()

    # Step 2: Load client requirements
    docs = load_document(input_file)
    input_text = "\n".join([d.page_content for d in docs])

    # Augment context with runtime constraints
    augmented_context = augment_context_with_constraints(
        input_text,
        mode=mode,
        developer_count=developer_count,
        project_budget=project_budget,
    )

    # Step 2b: Costing Agent
    costing_json = generate_costing(
        context=augmented_context,
        development_scope=development_scope,
        currency=currency,
        project_type=project_type,
        developer_count=developer_count,
        project_budget=project_budget,
    )

    # Step 3: Try JSON-first report generation
    report_json = generate_report_json(augmented_context, instruction=instruction)

    table_json_for_response: dict | None = None

    if report_json:
        # Step 4a: Save Report as PDF using structured JSON
        write_pdf_from_json(report_json, str(output_path), costing_json=costing_json)

        # Build a flattened text representation for embeddings
        flat_parts: list[str] = []
        flat_parts.append(report_json.get("title", ""))
        for section in report_json.get("sections", []):
            flat_parts.append(
                f"{section.get('section_number', '')}. {section.get('heading', '')}"
            )
            flat_parts.append(section.get("description", ""))
            for subsection in section.get("subsections", []):
                flat_parts.append(
                    f"{subsection.get('label', '')}: {subsection.get('content', '')}"
                )
        report_text_for_embeddings = "\n\n".join(
            p for p in flat_parts if isinstance(p, str) and p.strip()
        )
        table_json_for_response = extract_table_objects_from_json_report(report_json)
    else:
        # Step 4b: Fallback to legacy text pipeline
        logger.info("↩️ Falling back to legacy text-based report generation...")
        report_text_for_embeddings, json_data_for_legacy = generate_report(
            augmented_context, instruction=instruction
        )
        write_pdf(report_text_for_embeddings, str(output_path), json_data=json_data_for_legacy)
        table_json_for_response = json_data_for_legacy

    # Step 5: Embed Report into Chroma DB
    logger.info("🧩 Embedding generated report into Chroma vector DB...")
    chunks = split_text(report_text_for_embeddings)
    vector_store.add_texts(chunks)
    logger.info("✅ Report successfully embedded and saved in Chroma DB.")

    logger.info("🎯 Process complete — Report generated and indexed.")

    # Return both the PDF path, JSON data for table generation, and costing details
    return str(output_path), table_json_for_response, costing_json


# ------------------------------
# CLI Entry Point
# ------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Business Report Generator")
    parser.add_argument("--input", "-i", required=True, help="Client requirement file (.pdf or .txt)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PDF, help="Output report filename")
    args = parser.parse_args()

    process_document(args.input, args.output)

if __name__ == "__main__":
    main()