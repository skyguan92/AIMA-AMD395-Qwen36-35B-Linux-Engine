from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("agent_documents", ROOT / "scripts/check-agent-documents.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class AgentDocumentsTest(unittest.TestCase):
    def docx(self, path, *, missing=None):
        parts = {
            "[Content_Types].xml": f'<Types xmlns="{CHECKER.CONTENT_TYPES[1:-1]}"><Override PartName="/word/document.xml" ContentType="{CHECKER.DOCUMENT_TYPE}"/></Types>',
            "_rels/.rels": f'<Relationships xmlns="{CHECKER.RELATIONSHIPS[1:-1]}"><Relationship Id="rId1" Type="{CHECKER.OFFICE_DOCUMENT}" Target="word/document.xml"/></Relationships>',
            "word/document.xml": f'<w:document xmlns:w="{CHECKER.WORD[1:-1]}"><w:body><w:p><w:r><w:t>示例报告</w:t></w:r></w:p></w:body></w:document>',
        }
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in parts.items():
                if name != missing:
                    archive.writestr(name, content)

    def test_text_docx_is_recovered_but_never_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "report.docx"
            text = "# 示例报告\n\n文本不是 Word 文件。\n"
            path.write_text(text)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = CHECKER.main([str(path), "--markdown-dir", str(root / "md")])
            self.assertEqual(status, 2)
            self.assertFalse(json.loads(output.getvalue())["qualified"])
            self.assertEqual((root / "md/report.md").read_text(), text)
            self.assertEqual(path.read_text(), text)

    def test_real_docx_container_and_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.docx"
            self.docx(path)
            report, text = CHECKER.inspect_document(path)
            self.assertTrue(report["container_valid"])
            self.assertEqual(text, "示例报告\n")

    def test_missing_main_part_or_relationship_is_not_docx(self):
        with tempfile.TemporaryDirectory() as temporary:
            for missing in ("word/document.xml", "_rels/.rels", "[Content_Types].xml"):
                with self.subTest(missing=missing):
                    path = Path(temporary) / "report.docx"
                    self.docx(path, missing=missing)
                    self.assertFalse(CHECKER.inspect_document(path)[0]["container_valid"])

    def test_missing_empty_and_corrupt_files_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(CHECKER.inspect_document(root / "missing.docx")[0]["container_valid"])
            for data in (b"", b"PK\x03\x04not a zip", b"\xff\x00"):
                path = root / "report.docx"
                path.write_bytes(data)
                self.assertFalse(CHECKER.inspect_document(path)[0]["container_valid"])

    def test_unsafe_zip_member_is_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.docx"
            self.docx(path)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("../outside", "not extracted")
            self.assertFalse(CHECKER.inspect_document(path)[0]["container_valid"])

    def test_conversion_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "report.docx"
            self.docx(path)
            (root / "report.md").write_text("keep me")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(CHECKER.main([str(path), "--markdown-dir", str(root)]), 2)
            self.assertEqual((root / "report.md").read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
