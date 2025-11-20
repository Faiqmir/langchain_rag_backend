# app/utils/text_cleaning.py
import re
from typing import Tuple
 
# -------------------------
# Low-level helpers
# -------------------------
SSE_PREFIX_RE = re.compile(r'(?i)^\s*(data:|event:)\s*')
MULTIPLE_DATA_RE = re.compile(r'(?i)(?:\s*(data:|event:)\s*)+')
OBJECT_OBJECT_RE = re.compile(r'\[object Object\]')
 
def sanitize_sse_artifacts(chunk: str) -> str:
    """
    Remove SSE-like prefixes (data:, event:) and obvious placeholders
    that a model sometimes emits. This should be run *first* on raw chunks.
    """
    if not chunk:
        return ""
    s = str(chunk)
    # Remove repeated 'data:' or 'event:' tokens anywhere
    s = MULTIPLE_DATA_RE.sub(' ', s)
    # Remove leading single token if still present
    s = SSE_PREFIX_RE.sub('', s)
    # Remove JS placeholder
    s = OBJECT_OBJECT_RE.sub(' ', s)
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    return s
 
# -------------------------
# Word-split & token fixes
# -------------------------
def fix_generic_word_splits(text: str) -> str:
    """
    Fix common artifacts where compound words or domain names were split.
    Keep conservative: avoid aggressive merges that could corrupt normal text.
    """
    if not text:
        return ""
 
    t = text
 
    # Email handles: "user @ domain" -> "user@domain"
    t = re.sub(r'\b([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b', r'\1@\2', t)
 
    # Common tech compounds (conservative
 
    # Fix "C ++" -> "C++"
    t = re.sub(r'\b(C|F|Objective)\s*\+\+\b', r'\1++', t, flags=re.IGNORECASE)
 
    # Remove accidental spaces around punctuation caused by streams
    t = re.sub(r'\s+([.,:;!?])', r'\1', t)
 
    return t
 
# -------------------------
# Markdown repair helpers
# -------------------------
def repair_partial_markdown(text: str) -> str:
    """
    Heuristic repairs for broken markdown fragments (dangling asterisks, headings)
    This is intentionally conservative — it tries to make broken fragments look valid
    rather than invent structure.
    """
    if not text:
        return ""
 
    t = text
 
    # Fix headings missing space after hashes: "##Heading" -> "## Heading"
    t = re.sub(r'^(#{1,6})([A-Za-z0-9`*_])', r'\1 \2', t, flags=re.MULTILINE)
 
    # Remove dangling single asterisk at end of line (common mid-stream)
    t = re.sub(r'\*\s*$', '*', t, flags=re.MULTILINE)
 
    # Remove sequences like "* **" -> "* **" => ensure spacing normalized
    t = re.sub(r'\*\s+\*\*', '* **', t)
 
    # If a line starts with an asterisk followed by no space, add one: "*Item" -> "* Item"
    t = re.sub(r'^[\t ]*([\*\-])([A-Za-z0-9`*_])', r'\1 \2', t, flags=re.MULTILINE)
 
    # Trim repeated stray punctuation
    t = re.sub(r'([.,:;?!]){2,}', r'\1', t)
 
    # Normalize code fence variants like "```python" or "``` python"
    t = re.sub(r'^(```)\s+([A-Za-z0-9_-]+)\s*$', r'\1\2', t, flags=re.MULTILINE)
 
    return t
 
# -------------------------
# Early chunk normalization (call this on each raw chunk immediately)
# -------------------------
def early_normalize_chunk(chunk: str) -> str:
    """
    Run minimal, structural-preserving normalization on incoming chunk.
    Use this right away before adding the chunk to the buffer.
    Keeps spacing gentle so markdown detection still works.
    """
    if not chunk:
        return ""
 
    s = sanitize_sse_artifacts(chunk)
    s = fix_generic_word_splits(s)
    s = repair_partial_markdown(s)
 
    # Collapse repeated spaces but preserve single newlines/newline sequences
    # Replace multiple spaces with single spaces, but keep '\n' intact
    s = re.sub(r'[ \t]{2,}', ' ', s)
 
    return s
 
# -------------------------
# Markdown / structure detection
# -------------------------
def detect_markdown_boundary(buffer: str) -> Tuple[bool, dict]:
    """
    Determine whether the provided buffer contains a semantic boundary that can be flushed.
    Returns (should_flush, metadata). Metadata includes whether we are inside a code fence,
    current block type, and a short reason string.
    """
    if buffer is None:
        return False, {}
 
    b = buffer.lstrip()
 
    # Code fence start/end detection
    # If buffer contains an opening ``` without a matching closing fence yet => do not flush unless we have a full line
    fence_starts = len(re.findall(r'```', buffer))
    in_code_block = fence_starts % 2 == 1  # odd number means currently open fence
 
    # If buffer ends with blank line => paragraph boundary
    if '\n\n' in buffer:
        return True, {"type": "paragraph", "reason": "double_newline"}
 
    # If buffer begins with heading or a heading line completed (line ends with newline)
    if re.match(r'^\s*#{1,6}\s+.+\n', buffer):
        return True, {"type": "heading", "reason": "heading_line"}
 
    # If buffer contains a full bullet/numbered line
    if re.search(r'(^|\n)\s*([-*]|\d+\.)\s+.+\n', buffer):
        return True, {"type": "list_item", "reason": "list_line"}
 
    # If inside code fence, flush only on newline or closing fence
    if in_code_block:
        # flush if we have complete line to avoid partial garbage display
        if buffer.endswith('\n') or '```' in buffer:
            return True, {"type": "code_block", "reason": "in_code_block_line", "in_code_block": True}
        return False, {"type": "code_block", "reason": "incomplete"}
 
    # Sentence end (., !, ?) followed by space or newline
    if re.search(r'[\.!?]\s*($|\n)', buffer):
        return True, {"type": "sentence", "reason": "sentence_end"}
 
    # If buffer is longer than a safe chunk threshold, allow a flush at a word boundary
    if len(buffer) > 400:
        return True, {"type": "length", "reason": "max_length"}
 
    return False, {"type": "none", "reason": "none"}
 
# -------------------------
# Final cosmetic normalization (call this right before sending to client)
# -------------------------
def normalize_text(text: str) -> str:
    """
    Final cleanup for outgoing text. Conservative: do not try to invent structure.
    - Removes remaining 'data:' artifacts
    - Fixes spacing around punctuation
    - Converts loose bullets into proper newline-prefixed bullets
    - Collapses excessive blank lines
    - Capitalizes first character of the chunk if appropriate
    """
    if not text:
        return ""
 
    t = str(text)
 
    # Remove any lingering SSE artifacts
    t = re.sub(r'(?i)(?:\s*(data:|event:)\s*)+', ' ', t)
 
    # Remove placeholder artifacts
    t = OBJECT_OBJECT_RE.sub(' ', t)
 
    # Fix common split words and punctuation
    t = fix_generic_word_splits(t)
 
    # Normalize newlines and spaces
    t = t.replace('\r\n', '\n').replace('\r', '\n')
 
    # Ensure bullets have a space: "*Platform" -> "* Platform"
    t = re.sub(r'^[\t ]*([\*\-])([A-Za-z0-9`*_])', r'\1 \2', t, flags=re.MULTILINE)
 
    # Optionally normalize "*" bullets to "-" bullets
    t = re.sub(r'^[\t ]*\*\s+', r'- ', t, flags=re.MULTILINE)
 
    # Ensure single blank line separation (keep up to two newlines)
    t = re.sub(r'\n{3,}', '\n\n', t)
 
    # Ensure bullets start on their own line ONLY when they already begin a line.
    # This avoids converting inline " - " into separate bullets mid-sentence.
    t = re.sub(r'^[\t ]*([\*\-])\s+', r'\1 ', t, flags=re.MULTILINE)
 
    # Ensure heading spacing "#Heading" -> "# Heading"
    t = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', t, flags=re.MULTILINE)
 
    # For chunks that are not fenced code blocks, repair common malformed
    # emphasis / label patterns into valid markdown. This is where we clean
    # up outputs like:
    #   - **Platform Name** - **Womenpreneurship Platform (WP)*******Core Purpose* *- ...
    if "```" not in t:
        # 1) Normalize triple-asterisk emphasis ***Label*** -> **Label**
        t = re.sub(r'\*{3}([^*\n]*?\S)\*{3}', r'**\1**', t)
 
        # 2) Normalize single-asterisk emphasis *Label* -> **Label**
        t = re.sub(r'\*\s*([^*\n]*?\S)\s*\*', r'**\1**', t)
 
        # 3) Collapse accidental emphasized hyphens like "**-**" back to plain '-'
        t = re.sub(r'\*\*-\*\*', '-', t)
 
        # 4) Remove stray "* *" around punctuation such as "* *:" -> ":"
        t = re.sub(r'\*\s*\*([:;,.!?])', r'\1', t)
 
        # 5) Convert label/value patterns like '**Platform Name** - **Womenpreneurship Platform (WP)**'
        #    into '**Platform Name:** Womenpreneurship Platform (WP)'
        t = re.sub(
            r'(\*\*[^*]+?\*\*)\s*-\s*(\*\*[^*]+?\*\*)',
            lambda m: f"{m.group(1)[:-2]}:** {m.group(2)[2:]}",
            t,
        )
 
        # 6) Ensure sub-feature bullets like '- * **Label: ***' become '- **Label:**'
        #    and start on their own line.
        t = re.sub(r'\s+- \* \*\*', '\n- **', t)
        t = re.sub(r'- \*\*([^*]+?): \*\*\*', r'- **\1:**', t)
 
        # 7) Remove stray '*' before hyphens after bold labels: '**Core Purpose** *-'
        t = re.sub(r'(\*\*[^*]+?\*\*)\s*\*-\s*', r'\1 - ', t)
 
        # 8) Drop dangling '*' at end-of-line now that emphasis has been normalized
        t = re.sub(r'\*\s*$', '', t, flags=re.MULTILINE)
 
        # Remove stray asterisks immediately before label words ending with ':'
        # e.g. '- *WP:' -> '- WP:'
        t = re.sub(r'(-\s*)\*([^:\n]+):', r'\1\2:', t)
 
        # 9) As a final safety, collapse runs of 3+ asterisks to a single bold marker
        #    so sequences like '*******Core' do not leak into the UI.
        t = re.sub(r'\*{3,}', '**', t)
 
        # 10) Convert leading '-' and '*' bullets into numbered points per contiguous block
        #     so that feature lists like '- Mentorship...' or '*Mentorship...' become
        #     '1. Mentorship...'. Lines starting with '**' are treated as headings, not bullets.
        lines = t.split('\n')
        new_lines = []
        counter = 1
        in_block = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('- ') or (stripped.startswith('* ') and not stripped.startswith('**')):
                prefix_len = len(line) - len(stripped)
                prefix = line[:prefix_len]
                content = stripped[2:]
                new_lines.append(f"{prefix}{counter}. {content}")
                counter += 1
                in_block = True
            else:
                new_lines.append(line)
                in_block = False
                counter = 1
        t = "\n".join(new_lines)
 
        # 11) For ordered-list content, ensure each numbered item is on its own line
        #     and strip any remaining asterisks from the item text so no '*' markers
        #     are visible in numbered points.
        ordered_lines = []
        for line in t.split('\n'):
            # If a single line contains multiple numbered items like
            # "1. Foo 2. Bar 3. Baz", split them apart first.
            parts = re.split(r'(\d+\.\s+)', line)
            if len(parts) > 3:
                # parts looks like ['', '1. ', 'Foo ', '2. ', 'Bar ', ...]
                for i in range(1, len(parts), 2):
                    prefix = parts[i]
                    content = (parts[i + 1] if i + 1 < len(parts) else '').lstrip()
                    ordered_lines.append(prefix + content.replace('*', ''))
                continue
 
            m = re.match(r'^(\s*\d+\.\s+)(.*)$', line)
            if m:
                prefix, content = m.groups()
                ordered_lines.append(prefix + content.replace('*', ''))
            else:
                ordered_lines.append(line)
 
        t = "\n".join(ordered_lines)
 
    # Spaces around punctuation
    t = re.sub(r'\s+([.,:;!?])', r'\1', t)
    t = re.sub(r'([.,:;!?])(?=\S)', r'\1 ', t)
 
    # Trim trailing spaces on each line
    t = "\n".join([line.rstrip() for line in t.splitlines()])
 
    # Trim overall whitespace
    t = t.strip()
 
    # Capitalize first char of the chunk if likely a sentence start
    if t and t[0].islower():
        # Only force if the first word looks alphabetic and not code
        if re.match(r'[a-z]', t[0]):
            t = t[0].upper() + t[1:]
 
    return t
 
# -------------------------
# Convenience: combined processing function for rag_chat
# -------------------------
def process_incoming_chunk_for_buffer(chunk: str) -> str:
    """
    A convenience wrapper intended to be used in your streaming loop:
      - Run early_normalize_chunk immediately on raw chunk
      - Return the cleaned string ready to append to your buffer
    Use normalize_text() later right before emitting the SSE chunk to the client.
    """
    return early_normalize_chunk(chunk)
 
# -------------------------
# End of file
# -------------------------