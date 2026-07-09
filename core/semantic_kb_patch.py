"""
core/semantic_kb_patch.py

When an admin approves a user-submitted Semantic Layer correction, this
module:
  1. Locates the correct KB markdown file for the approved table/column
  2. Patches ONLY the approved column's metadata in the ## Columns section
  3. Adds a structured marker so the semantic layer parser reads 100% confidence
  4. Re-embeds the patched KB file into Qdrant (single-file, not full rebuild)

The rest of the KB file (joins, metrics, business synonyms, admin edits,
sample data) is completely untouched.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("querybot.semantic_kb_patch")

# Structured marker appended to the patched column line.
_APPROVED_SOURCE = "Admin-approved Semantic Layer edit"


# ── Public entry point ────────────────────────────────────────────────────────

def locate_kb_file_for_feedback(
    *,
    kb_dir: str,
    table_fqn: str,
    table_name: str,
    schema_name: str,
) -> str:
    """Return the KB filename that an approved feedback item will patch."""
    kb_path = Path(kb_dir) if kb_dir else None
    if not kb_path or not kb_path.exists():
        return ""
    kb_file = _find_kb_file(kb_path, table_fqn, table_name, schema_name)
    return kb_file.name if kb_file else ""


def apply_approved_feedback(
    *,
    account_id:       str,
    kb_dir:           str,
    table_fqn:        str,
    table_name:       str,
    schema_name:      str,
    column_name:      str,
    approved_meaning: str,
    approved_use_case: str,
    user_comment:     str = "",
    approved_synonyms: list[str] | None = None,
    admin_note: str = "",
    persist_override: bool = False,
    infer_synonyms: bool = True,
) -> tuple[bool, str]:
    """
    Locate the KB file for this table, patch the column metadata AND the
    Business Synonyms section, then re-embed into Qdrant.

    What gets written to the KB file:
      1. ## Columns  — column meaning updated / approval marker added
      2. ## Business Synonyms — any new synonym terms added as rows
         (extracted from approved_use_case and user_comment)

    Re-embed is called once after both patches are applied.
    Returns (success: bool, human-readable message).
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return False, f"KB directory not found: {kb_dir}"

    # ── Step 1: find the KB file ──────────────────────────────────────────────
    kb_file = _find_kb_file(kb_path, table_fqn, table_name, schema_name)
    if kb_file is None:
        return False, (
            f"Could not find a KB file for table {table_fqn}. "
            f"Expected a file like [{schema_name}__{table_name}_kb.md] "
            f"or [{table_name}_kb.md] in {kb_dir}. "
            f"Rebuild the KB first if schema discovery was recently run."
        )

    original = kb_file.read_text(encoding="utf-8", errors="replace")
    previous_synonyms: list[str] = []
    if persist_override:
        try:
            from core.field_overrides import load_field_overrides, table_overrides
            previous_fields = table_overrides(
                load_field_overrides(account_id),
                table_fqn,
                table_name,
                kb_file.stem.replace("_kb", ""),
            )
            previous = previous_fields.get(column_name.upper()) or {}
            previous_synonyms = previous.get("synonyms") or []
        except Exception:
            previous_synonyms = []

    # ── Step 2: patch the column meaning in ## Columns ────────────────────────
    patched, col_changed = _patch_column(
        content=original,
        column_name=column_name,
        approved_meaning=approved_meaning,
        approved_use_case=approved_use_case,
    )
    if not col_changed:
        log.warning("Column %s not found in %s — appending", column_name, kb_file.name)
        patched = _append_column(original, column_name, approved_meaning, approved_use_case)
    if previous_synonyms:
        patched = _remove_synonyms(patched, column_name, previous_synonyms)

    # ── Step 3: extract new synonyms from the approved texts ──────────────────
    # We mine approved_use_case and user_comment for synonym-like terms.
    # The user said "refers to nationality, country" → we extract "country"
    # as a new synonym for the Nationality column and add it to Business Synonyms.
    new_synonyms = []
    if infer_synonyms:
        new_synonyms = _extract_new_synonyms(
            column_name=column_name,
            approved_meaning=approved_meaning,
            approved_use_case=approved_use_case,
            user_comment=user_comment,
            existing_content=original,
        )
    explicit_synonyms = [
        str(term).strip()
        for term in (approved_synonyms or [])
        if str(term).strip()
    ]
    seen_synonyms = {term.lower() for term in new_synonyms}
    for term in explicit_synonyms:
        if term.lower() not in seen_synonyms:
            new_synonyms.append(term)
            seen_synonyms.add(term.lower())

    if new_synonyms:
        patched = _patch_synonyms(patched, column_name, new_synonyms)
        log.info("Synonyms added to Business Synonyms for %s.%s: %s",
                 table_name, column_name, new_synonyms)

    # ── Step 4: write the fully patched file ──────────────────────────────────
    kb_file.write_text(patched, encoding="utf-8")
    log.info("KB file patched: %s — column %s + %d synonym(s)",
             kb_file.name, column_name, len(new_synonyms))

    # ── Step 5: re-embed only this file ──────────────────────────────────────
    try:
        from core.knowledge import re_embed_file
        re_embed_file(str(kb_path), account_id, kb_file.name)
        log.info("Re-embedded %s for account %s", kb_file.name, account_id)
    except Exception as e:
        log.error("Re-embed failed for %s: %s", kb_file.name, e)
        kb_file.write_text(original, encoding="utf-8")
        return False, (
            f"KB file was updated but re-embedding failed: {e}. "
            f"File reverted. Try again or rebuild the full KB."
        )

    if persist_override:
        try:
            from core.field_overrides import save_field_override
            save_field_override(
                account_id=account_id,
                table_fqn=table_fqn,
                schema_name=schema_name,
                table_name=table_name,
                file_stem=kb_file.stem.replace("_kb", ""),
                column_name=column_name,
                meaning=approved_meaning,
                use_case=approved_use_case,
                synonyms=new_synonyms,
                admin_note=admin_note,
            )
        except Exception as e:
            log.error("Could not persist field override for %s.%s: %s",
                      table_name, column_name, e)
            kb_file.write_text(original, encoding="utf-8")
            try:
                from core.knowledge import re_embed_file
                re_embed_file(str(kb_path), account_id, kb_file.name)
            except Exception as rollback_error:
                log.error("Field override rollback re-embed failed for %s: %s",
                          kb_file.name, rollback_error)
            return False, (
                f"Field edit could not be persisted: {e}. "
                "The KB file was reverted."
            )

    semantic_model_changed = False
    model_failure = ""
    try:
        from core.semantic_model import patch_field_approval
        semantic_model_changed = patch_field_approval(
            kb_dir=str(kb_path),
            table_fqn=table_fqn,
            table_name=table_name,
            schema_name=schema_name,
            column_name=column_name,
            approved_meaning=approved_meaning,
            approved_use_case=approved_use_case,
            approved_synonyms=new_synonyms,
        )
        if semantic_model_changed:
            log.info("Structured semantic model patched for %s.%s", table_name, column_name)
        else:
            model_failure = (
                f" WARNING: the structured semantic model has no matching entry for "
                f"{table_name}.{column_name} — runtime field enforcement will NOT apply "
                f"to this approval. Rebuild the KB to regenerate the model."
            )
            log.warning("Structured semantic model patch found no match for %s.%s", table_name, column_name)
    except Exception as e:
        model_failure = (
            f" WARNING: KB text was updated but the structured semantic model patch "
            f"failed ({e}) — runtime field enforcement will NOT apply to this approval."
        )
        log.warning("Structured semantic model patch failed for %s.%s: %s", table_name, column_name, e)

    synonym_note = f" + {len(new_synonyms)} synonym(s) added" if new_synonyms else ""
    model_note = " + semantic model updated" if semantic_model_changed else ""
    return True, f"Approved and KB re-embedded ({kb_file.name}{synonym_note}{model_note}){model_failure}"


def apply_field_overrides_to_content(
    content: str,
    field_overrides: dict[str, dict],
) -> str:
    """Apply persistent admin field overrides to freshly generated KB content."""
    patched = content
    for column_key, override in (field_overrides or {}).items():
        if not isinstance(override, dict):
            continue
        column_name = override.get("column_name") or column_key
        meaning = (override.get("meaning") or "").strip()
        use_case = (override.get("use_case") or "").strip()
        if not meaning:
            continue
        patched, changed = _patch_column(
            content=patched,
            column_name=column_name,
            approved_meaning=meaning,
            approved_use_case=use_case,
        )
        if not changed:
            patched = _append_column(
                patched,
                column_name,
                meaning,
                use_case,
            )
        synonyms = [
            str(term).strip()
            for term in (override.get("synonyms") or [])
            if str(term).strip()
        ]
        if synonyms:
            patched = _patch_synonyms(patched, column_name, synonyms)
    return patched


# ── KB file discovery ─────────────────────────────────────────────────────────

def _find_kb_file(
    kb_path: Path,
    table_fqn: str,
    table_name: str,
    schema_name: str,
) -> Path | None:
    """
    Locate the _kb.md file for a given table.

    Search order (most specific to least specific):
      1. Header match — read each *_kb.md and match the # FQN header line
      2. Filename patterns:  schema__table_kb.md, table_kb.md
         (case-insensitive, both with and without schema prefix)
    """
    fqn_upper   = table_fqn.upper()
    table_upper = table_name.upper()
    schema_upper = schema_name.upper()

    kb_files = sorted(kb_path.glob("*_kb.md"))
    if not kb_files:
        return None

    # Pass 1: match by FQN header inside the file
    for f in kb_files:
        try:
            first_lines = f.read_text(encoding="utf-8", errors="replace")[:400]
        except Exception:
            continue
        for line in first_lines.splitlines()[:6]:
            header = line.strip().lstrip("#").strip().upper()
            # FQN match: DB.SCHEMA.TABLE or SCHEMA.TABLE
            if header == fqn_upper:
                return f
            # Table name match — less strict
            parts = fqn_upper.split(".")
            if header in (parts[-1], ".".join(parts[-2:])):
                return f

    # Pass 2: filename pattern match — compare UPPERCASE to UPPERCASE
    candidates = [
        f"{schema_upper}__{table_upper}_KB.MD",
        f"{table_upper}_KB.MD",
        f"{schema_upper}_{table_upper}_KB.MD",
    ]
    name_map = {f.name.upper(): f for f in kb_files}
    for candidate in candidates:
        if candidate in name_map:
            return name_map[candidate]

    # Pass 3: case-insensitive partial match on table name
    for f in kb_files:
        stem = f.stem.upper().replace("_KB", "")
        if stem == table_upper or stem.endswith(f"__{table_upper}") or stem.endswith(f"_{table_upper}"):
            return f

    return None


# ── Column patch ──────────────────────────────────────────────────────────────

def _patch_column(
    content:          str,
    column_name:      str,
    approved_meaning: str,
    approved_use_case: str,
) -> tuple[str, bool]:
    """
    Find and replace the column entry in the ## Columns section.
    Returns (patched_content, was_changed).

    Handles three common formats written by LLMs:
      A. Bullet:    - `ColName` (type): old meaning text
      B. Table row: | `ColName` | type | nullable | values |
      C. Bold:      **ColName** (type): meaning

    For bullet/bold formats, approved metadata replaces the field block. For
    table rows, approved metadata is written back into the same row so the
    Semantic Layer shows the approved value in its original position.
    """
    col_upper = column_name.strip().upper()
    lines     = content.splitlines(keepends=True)

    in_columns = False
    changed    = False
    result     = []

    i = 0
    while i < len(lines):
        line    = lines[i]
        stripped = line.rstrip()

        # Track ## Columns section entry/exit
        if re.match(r"^#{1,3}\s+(columns|column definitions|fields)", stripped, re.I):
            in_columns = True
            result.append(line)
            i += 1
            continue
        if in_columns and re.match(r"^#{1,3}\s+", stripped) and not re.match(
            r"^#{1,3}\s+(columns|column definitions|fields)", stripped, re.I
        ):
            in_columns = False

        if not in_columns:
            result.append(line)
            i += 1
            continue

        # Does this line reference the target column?
        col_match = re.search(
            r"[`*\"\[|]?" + re.escape(column_name) + r"[`*\"\]|]?",
            stripped, re.I
        )
        if not col_match:
            result.append(line)
            i += 1
            continue

        # ── Format A: bullet  - `ColName` (type): meaning ────────────────────
        bullet_match = re.match(r"^(-\s*`[^`]+`(?:\s*\([^)]+\))?)\s*:", stripped)
        if bullet_match:
            prefix = bullet_match.group(1)
            new_line = (
                f"{prefix}: {approved_meaning.strip()}\n"
                f"  Use case: {approved_use_case.strip()}\n"
                f"  Confidence: 100%.\n"
                f"  Source: {_APPROVED_SOURCE}.\n"
            )
            result.append(new_line)
            changed = True
            # Skip old continuation lines
            i += 1
            while i < len(lines):
                nx = lines[i].rstrip()
                if nx.startswith("  ") and re.match(r"\s*(Use case:|Confidence:|Source:)", nx):
                    i += 1
                else:
                    break
            continue

        # ── Format B: table row  | `ColName` | type | nullable | values | ────
        if stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and cells[0].strip("`").upper() == col_upper:
                # Write approved metadata into the same row for in-place display.
                _ensure_table_metadata_headers(result)
                cells = _approved_table_row_cells(cells, approved_meaning, approved_use_case)
                result.append("| " + " | ".join(cells) + " |\n")
                changed = True
                i += 1
                # Remove any previously injected approval comment
                while i < len(lines) and "<!-- Approved:" in lines[i]:
                    i += 1
                continue

        # ── Format C: bold  **ColName**: meaning or **ColName** (type): meaning ─
        bold_match = re.match(r"^\*\*([^*]+)\*\*", stripped)
        if bold_match and bold_match.group(1).strip().upper() == col_upper:
            type_m = re.search(r"\(([^)]+)\)", stripped)
            type_str = f" ({type_m.group(1)})" if type_m else ""
            new_line = (
                f"**{column_name}**{type_str}: {approved_meaning.strip()}\n"
                f"  Use case: {approved_use_case.strip()}\n"
                f"  Confidence: 100%.\n"
                f"  Source: {_APPROVED_SOURCE}.\n"
            )
            result.append(new_line)
            changed = True
            i += 1
            while i < len(lines):
                nx = lines[i].rstrip()
                if nx.startswith("  ") and re.match(r"\s*(Use case:|Confidence:|Source:)", nx):
                    i += 1
                else:
                    break
            continue

        result.append(line)
        i += 1

    return "".join(result), changed


def _approved_table_row_cells(
    cells: list[str],
    approved_meaning: str,
    approved_use_case: str,
) -> list[str]:
    """Return table row cells with approved metadata in fixed positions."""
    updated = list(cells)
    while len(updated) < 4:
        updated.append("")
    while len(updated) < 8:
        updated.append("")
    updated[4] = approved_meaning.strip()
    updated[5] = approved_use_case.strip()
    updated[6] = "100%"
    updated[7] = _APPROVED_SOURCE
    return updated


def _ensure_table_metadata_headers(result_lines: list[str]) -> None:
    """Extend the current markdown table header for approved metadata cells."""
    if len(result_lines) < 2:
        return
    header_idx = len(result_lines) - 2
    sep_idx = len(result_lines) - 1
    header = result_lines[header_idx].rstrip()
    separator = result_lines[sep_idx].rstrip()
    if not (header.startswith("|") and separator.startswith("|") and "---" in separator):
        return
    header_cells = [c.strip() for c in header.strip("|").split("|")]
    separator_cells = [c.strip() for c in separator.strip("|").split("|")]
    if len(header_cells) >= 8:
        return
    extra_headers = ["Meaning", "Use case", "Confidence", "Source"]
    missing = 8 - len(header_cells)
    header_cells.extend(extra_headers[-missing:] if missing < len(extra_headers) else extra_headers)
    separator_cells.extend(["---"] * (8 - len(separator_cells)))
    result_lines[header_idx] = "| " + " | ".join(header_cells) + " |\n"
    result_lines[sep_idx] = "| " + " | ".join(separator_cells[:len(header_cells)]) + " |\n"


def _append_column(
    content: str,
    column_name: str,
    approved_meaning: str,
    approved_use_case: str,
) -> str:
    """
    If the column wasn't found, append it to the ## Columns section.
    """
    new_entry = (
        f"- `{column_name}`: {approved_meaning.strip()}\n"
    )
    if approved_use_case.strip():
        new_entry += (
            f"  Use case: {approved_use_case.strip()}\n"
        )
    new_entry += (
        f"  Confidence: 100%.\n"
        f"  Source: {_APPROVED_SOURCE}.\n"
    )

    # Find the ## Columns section and append after the last bullet in it
    lines = content.splitlines(keepends=True)
    col_section_end = None
    in_columns = False

    for idx, line in enumerate(lines):
        if re.match(r"^#{1,3}\s+(columns|column definitions|fields)", line.rstrip(), re.I):
            in_columns = True
            continue
        if in_columns:
            if re.match(r"^#{1,3}\s+", line.rstrip()):
                col_section_end = idx
                break
            if line.strip().startswith("- ") or line.strip().startswith("|"):
                col_section_end = idx + 1

    if col_section_end is not None:
        lines.insert(col_section_end, new_entry)
        return "".join(lines)

    # Fallback: just append at the end
    return content + "\n" + new_entry


# ══════════════════════════════════════════════════════════════════════════════
# Synonym extraction and Business Synonyms patching
# ══════════════════════════════════════════════════════════════════════════════

def _extract_new_synonyms(
    column_name:      str,
    approved_meaning: str,
    approved_use_case: str,
    user_comment:     str,
    existing_content: str,
) -> list[str]:
    """
    Extract plain-English synonym terms from the admin-approved texts that
    should be added to the ## Business Synonyms table.

    Strategy:
    - Split approved_use_case and user_comment on commas and common delimiters
    - Strip noise words ("refers to", "when", "used for", "a question", etc.)
    - Remove terms that already exist in the Business Synonyms table for this column
    - Remove the column name itself and single-char tokens
    - Return a deduplicated list of new synonym terms (lowercase, sorted)

    Example:
      column_name = "Nationality"
      approved_use_case = "Used when a question explicitly refers to nationality, country."
      → extracts ["country"]  (nationality already exists as a synonym)
    """
    col_upper = column_name.upper()

    # Collect candidates from use_case and user_comment ONLY
    # (approved_meaning is a description, not a synonym source)
    NOISE = {
        "used", "when", "a", "an", "the", "question", "explicitly", "refers",
        "to", "for", "by", "is", "are", "of", "with", "in", "on", "at",
        "this", "field", "column", "table", "selected", "filter", "use",
        "case", "what", "it", "that", "these", "those", "how", "where",
        "which", "data", "value", "values", "only", "also", "can", "will",
        "from", "into", "based", "using", "context", "needs", "has",
        "select", "show", "get", "find", "list", "display", "called",
        "known", "as", "such", "like", "same", "refers", "see", "also",
        "employee", "employees", "record", "records",
    }

    candidates: list[str] = []
    for source in [approved_use_case, user_comment]:
        if not source:
            continue
        parts = re.split(r"[,;]|\bor\b|\band\b", source, flags=re.I)
        for part in parts:
            clean = re.sub(r"[.!?\"'()\[\]{}]", "", part).strip().lower()
            words = clean.split()
            while words and words[0] in NOISE:
                words.pop(0)
            while words and words[-1] in NOISE:
                words.pop()
            term = " ".join(words).strip()
            if len(term) >= 2:
                candidates.append(term)

    # Load existing Business Synonyms for this column from the KB file
    existing_synonyms: set[str] = set()
    in_synonyms = False
    for line in existing_content.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,3}\s+business synonyms", stripped, re.I):
            in_synonyms = True
            continue
        if in_synonyms and re.match(r"^#{1,3}\s+", stripped):
            break
        if in_synonyms and stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip().lower() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                # cell[0] = plain english terms, cell[1] = column name
                if cells[1].strip("`").upper() == col_upper:
                    for t in cells[0].split(","):
                        existing_synonyms.add(t.strip())

    # Also exclude the column name itself in various forms
    col_variants = {
        column_name.lower(),
        column_name.replace("_", " ").lower(),
        column_name.replace("_", "").lower(),
    }
    col_variants |= {part.lower() for part in re.split(r"[_\s]+", column_name) if len(part) >= 2}
    existing_synonyms |= col_variants

    # Filter: new, non-trivial, non-duplicate, short (max 4 words)
    new_terms = []
    seen: set[str] = set()
    for term in candidates:
        term_lower = term.lower()
        word_count = len(term_lower.split())
        if (
            term_lower
            and len(term_lower) >= 2
            and word_count <= 4              # synonyms should be short phrases
            and term_lower not in existing_synonyms
            and term_lower not in seen
            and term_lower not in col_variants
            and not all(w in NOISE for w in term_lower.split())
        ):
            new_terms.append(term_lower)
            seen.add(term_lower)

    return sorted(new_terms)


def _patch_synonyms(content: str, column_name: str, new_synonyms: list[str]) -> str:
    """
    Add new synonym rows to the ## Business Synonyms table in the KB file.

    If the section doesn't exist, create it before ## Key Metrics or at the end.
    If the column already has a row, append the new terms to that row.
    If no row for this column, add a new row.

    Format of each row:
      | synonym term  | ColumnName | Admin-approved synonym. |
    """
    if not new_synonyms:
        return content

    col_upper = column_name.upper()
    lines = content.splitlines(keepends=True)

    # ── Try to find an existing row for this column in Business Synonyms ──────
    in_synonyms = False
    existing_row_idx = None
    synonyms_section_end = None   # line index just after last row in the section
    synonyms_header_idx  = None   # line index of the ## Business Synonyms header

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#{1,3}\s+business synonyms", stripped, re.I):
            in_synonyms = True
            synonyms_header_idx = idx
            continue
        if in_synonyms:
            if re.match(r"^#{1,3}\s+", stripped):
                synonyms_section_end = idx
                break
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cells) >= 2 and cells[1].strip("`").upper() == col_upper:
                    existing_row_idx = idx
                synonyms_section_end = idx + 1

    if existing_row_idx is not None:
        # Append new terms to the existing row
        row = lines[existing_row_idx]
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        # cells[0] = current terms, cells[1] = column, cells[2] = notes
        current_terms = [t.strip().lower() for t in cells[0].split(",") if t.strip()]
        added = [t for t in new_synonyms if t.lower() not in current_terms]
        if added:
            cells[0] = ", ".join(current_terms + added)
            note = cells[2] if len(cells) > 2 else ""
            if "Admin-approved" not in note:
                note = (note + " | Admin-approved synonym." if note else "Admin-approved synonym.")
                if len(cells) > 2:
                    cells[2] = note
                else:
                    cells.append(note)
            new_row = "| " + " | ".join(cells) + " |\n"
            lines[existing_row_idx] = new_row
        return "".join(lines)

    # ── Build new rows for the column ────────────────────────────────────────
    new_rows = "".join(
        f"| {term} | {column_name} | Admin-approved synonym. |\n"
        for term in new_synonyms
    )

    if synonyms_header_idx is not None and synonyms_section_end is not None:
        # Insert before the section end
        lines.insert(synonyms_section_end, new_rows)
        return "".join(lines)

    if synonyms_header_idx is not None:
        # Section exists but has no rows yet — append after header/table header
        # Find the table header row (---|---|---) if present
        insert_at = synonyms_header_idx + 1
        for j in range(synonyms_header_idx + 1, len(lines)):
            if "---" in lines[j]:
                insert_at = j + 1
                break
        lines.insert(insert_at, new_rows)
        return "".join(lines)

    # ── No Business Synonyms section — create one ─────────────────────────────
    section = (
        "\n## Business Synonyms\n"
        "| Plain English | Column | Notes |\n"
        "|---|---|---|\n"
        + new_rows
    )

    # Insert before ## Key Metrics if it exists, otherwise before the last heading
    for idx, line in enumerate(lines):
        if re.match(r"^#{1,3}\s+key metrics", line.strip(), re.I):
            lines.insert(idx, section)
            return "".join(lines)

    # Append at the end
    return content + section


def _remove_synonyms(content: str, column_name: str, synonyms: list[str]) -> str:
    """Remove previously persisted admin synonyms while preserving other terms."""
    remove = {str(term).strip().lower() for term in synonyms if str(term).strip()}
    if not remove:
        return content
    column_upper = column_name.upper()
    lines = content.splitlines(keepends=True)
    in_synonyms = False
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,3}\s+business synonyms", stripped, re.I):
            in_synonyms = True
            output.append(line)
            continue
        if in_synonyms and re.match(r"^#{1,3}\s+", stripped):
            in_synonyms = False
        if in_synonyms and stripped.startswith("|") and "---" not in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) >= 2 and cells[1].strip("`").upper() == column_upper:
                kept = [
                    term.strip()
                    for term in cells[0].split(",")
                    if term.strip() and term.strip().lower() not in remove
                ]
                if not kept:
                    continue
                cells[0] = ", ".join(kept)
                line = "| " + " | ".join(cells) + " |\n"
        output.append(line)
    return "".join(output)
