#!/usr/bin/env python3
"""Check DOCX containers and optionally recover readable text as Markdown.

This is a caller-side development helper, not a native runtime dependency or
a check of document layout, legal conclusions, citations, or task completion.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import xml.etree.ElementTree as ET
import zipfile

MAX_BYTES = 64 * 1024 * 1024
WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
CONTENT_TYPES = "{http://schemas.openxmlformats.org/package/2006/content-types}"
RELATIONSHIPS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
OFFICE_DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
DOCUMENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"


def xml_member(archive: zipfile.ZipFile, name: str) -> ET.Element:
    raw = archive.read(name)
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("XML document types/entities are not admitted")
    return ET.fromstring(raw)


def inspect_document(path: Path) -> tuple[dict, str | None]:
    result = {"name": path.name, "container_valid": False, "format": "unknown"}
    text = None
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("expected a regular, non-symlink file")
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("file exceeds 64 MiB limit")
        raw = path.read_bytes()
        result.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if path.suffix.lower() != ".docx":
            raise ValueError("only .docx inputs are supported")
        if not zipfile.is_zipfile(path):
            try:
                candidate = raw.decode("utf-8-sig")
                if candidate.strip() and all(ord(c) >= 32 or c in "\n\r\t" for c in candidate):
                    text = candidate
                    result["format"] = "utf8_text_with_docx_extension"
            except UnicodeDecodeError:
                pass
            raise ValueError("not an OOXML ZIP container; renaming text does not create DOCX")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(members) > 4096 or sum(member.file_size for member in members) > MAX_BYTES:
                raise ValueError("expanded archive exceeds inspection limits")
            if len(names) != len(set(names)):
                raise ValueError("duplicate ZIP member")
            for member in members:
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1):
                    raise ValueError("unsafe or encrypted ZIP member")
            if archive.testzip() is not None:
                raise ValueError("ZIP member CRC failure")
            types = xml_member(archive, "[Content_Types].xml")
            relationships = xml_member(archive, "_rels/.rels")
            document = xml_member(archive, "word/document.xml")
            declared = any(node.get("PartName") == "/word/document.xml"
                           and node.get("ContentType") == DOCUMENT_TYPE
                           for node in types.findall(CONTENT_TYPES + "Override"))
            linked = any(node.get("Type") == OFFICE_DOCUMENT
                         and node.get("Target") in ("word/document.xml", "/word/document.xml")
                         and node.get("TargetMode", "Internal") == "Internal"
                         for node in relationships.findall(RELATIONSHIPS + "Relationship"))
            if (types.tag != CONTENT_TYPES + "Types" or not declared or not linked
                    or relationships.tag != RELATIONSHIPS + "Relationships"
                    or document.tag != WORD + "document" or document.find(WORD + "body") is None):
                raise ValueError("missing or inconsistent DOCX main-document structure")
            paragraphs = []
            for paragraph in document.iter(WORD + "p"):
                pieces = []
                for node in paragraph.iter():
                    if node.tag == WORD + "t":
                        pieces.append(node.text or "")
                    elif node.tag == WORD + "tab":
                        pieces.append("\t")
                    elif node.tag in (WORD + "br", WORD + "cr"):
                        pieces.append("\n")
                paragraphs.append("".join(pieces))
            text = "\n\n".join(paragraphs) + "\n"
            result.update(container_valid=True, format="docx_ooxml", paragraph_count=len(paragraphs))
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile, ET.ParseError) as error:
        result["error"] = str(error)
    return result, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path, nargs="+")
    parser.add_argument("--markdown-dir", type=Path)
    args = parser.parse_args(argv)
    reports = []
    for path in args.documents:
        report, text = inspect_document(path)
        if args.markdown_dir is not None and text is not None:
            try:
                args.markdown_dir.mkdir(parents=True, exist_ok=True)
                destination = args.markdown_dir / (path.stem + ".md")
                # Do not overwrite an original, an earlier conversion, or a
                # same-name document from another directory.
                with destination.open("x", encoding="utf-8") as stream:
                    stream.write(text)
                report["markdown"] = destination.name
                report["markdown_scope"] = "readable text only; layout, numbering, images and claims not validated"
            except OSError as error:
                report["conversion_error"] = str(error)
        reports.append(report)
    qualified = all(item["container_valid"] and not item.get("conversion_error") for item in reports)
    print(json.dumps({"schema": "aima.agent-document-inspection.v1", "qualified": qualified,
                      "documents": reports}, ensure_ascii=False, indent=2))
    # A successful text recovery must never turn a fake DOCX into a passing
    # artifact. Callers can feed this failure back to their repair loop.
    return 0 if qualified else 2


if __name__ == "__main__":
    sys.exit(main())
