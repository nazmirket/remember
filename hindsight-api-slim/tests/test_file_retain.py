"""
End-to-end tests for file retain (upload, convert, retain) functionality.
"""

import asyncio
import io
import json
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from hindsight_api.engine.providers.openai_compatible_headers import OPENAI_COMPATIBLE_USER_AGENT
from hindsight_api.extensions import FileConvertResult, OperationValidatorExtension, ValidationResult
from hindsight_api.extensions.operation_validator import (
    RecallContext,
    RecallResult,
    ReflectContext,
    RetainContext,
    RetainResult,
)


@pytest.fixture
def sample_pdf_content():
    """Create a simple PDF-like content for testing."""
    # This is a minimal PDF that markitdown can parse
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
"""


@pytest.fixture
def sample_txt_content():
    """Create simple text content."""
    return b"This is a test document.\nIt contains some important information.\nAlice works at Google."


@pytest.mark.asyncio
async def test_file_storage_postgresql(memory_no_llm_verify, sample_txt_content):
    """Test file storage in PostgreSQL."""
    # Test that files are stored and retrieved correctly
    storage = memory_no_llm_verify._file_storage

    # Store a file
    key = "test/file1.txt"
    stored_key = await storage.store(
        file_data=sample_txt_content,
        key=key,
        metadata={"content_type": "text/plain"},
    )

    assert stored_key == key

    # Retrieve the file
    retrieved = await storage.retrieve(key)
    assert retrieved == sample_txt_content

    # Check if file exists
    exists = await storage.exists(key)
    assert exists is True

    # Delete the file
    await storage.delete(key)

    # Check file no longer exists
    exists_after = await storage.exists(key)
    assert exists_after is False


@pytest.mark.asyncio
async def test_markitdown_converter():
    """Test markitdown parser."""
    from hindsight_api.engine.parsers import MarkitdownParser

    parser = MarkitdownParser()

    # Test simple text file
    text_content = b"This is a test document.\nWith multiple lines."
    result = await parser.convert(text_content, "test.txt")

    assert isinstance(result, str)
    assert len(result) > 0
    assert "test document" in result.lower() or "multiple lines" in result.lower()


def test_markitdown_converter_does_not_enable_ocr_by_default(monkeypatch):
    """Markitdown should keep its local/default behavior unless OCR is explicitly enabled."""
    import markitdown

    from hindsight_api.engine.parsers import MarkitdownParser

    calls = []

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)

    parser = MarkitdownParser()
    # markitdown is imported and MarkItDown built on first use, not at construction,
    # to keep bs4/lxml off the startup path (#4031).
    assert calls == []
    parser._get_markitdown()

    assert calls == [{}]


@pytest.mark.asyncio
async def test_markitdown_image_without_ocr_has_actionable_error(monkeypatch):
    """Image uploads should explain that MarkItDown OCR is disabled instead of surfacing a low-level error."""
    import markitdown

    from hindsight_api.engine.parsers import MarkitdownParser

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            pass

        def convert(self, path):
            raise AssertionError("MarkItDown should not be called when image OCR is disabled")

    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)

    parser = MarkitdownParser()
    with pytest.raises(RuntimeError, match="Image OCR is not enabled for the markitdown parser"):
        await parser.convert(b"\x89PNG\r\n\x1a\n", "screenshot.png")


def test_markitdown_converter_can_enable_ocr(monkeypatch):
    """When enabled, Markitdown receives an OpenAI-compatible client, model, and OCR prompt."""
    import markitdown
    import openai

    from hindsight_api.config import DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT
    from hindsight_api.engine.parsers import MarkitdownParser

    markitdown_calls = []
    openai_calls = []

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            markitdown_calls.append(kwargs)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            openai_calls.append(kwargs)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    parser = MarkitdownParser(
        ocr_enabled=True,
        ocr_api_key="parser-key",
        ocr_base_url="https://vision.example/v1",
        ocr_model="vision-model",
    )
    # The OpenAI client is built with MarkItDown on first use, not at construction (#4031).
    assert openai_calls == []
    parser._get_markitdown()

    assert openai_calls == [
        {
            "api_key": "parser-key",
            "base_url": "https://vision.example/v1",
            "default_headers": {"User-Agent": OPENAI_COMPATIBLE_USER_AGENT},
        }
    ]
    assert markitdown_calls[0]["llm_client"].__class__ is FakeOpenAI
    assert markitdown_calls[0]["llm_model"] == "vision-model"
    assert markitdown_calls[0]["llm_prompt"] == DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT


def test_markitdown_converter_passes_default_headers_when_configured(monkeypatch):
    """Custom OCR headers are forwarded to the OpenAI client alongside the hindsight User-Agent."""
    import markitdown
    import openai

    from hindsight_api.engine.parsers import MarkitdownParser

    openai_calls = []

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            pass

    class FakeOpenAI:
        def __init__(self, **kwargs):
            openai_calls.append(kwargs)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    headers = {"X-Component-Id": "hindsight-ocr", "X-Request-Source": "markitdown"}
    parser = MarkitdownParser(
        ocr_enabled=True,
        ocr_api_key="parser-key",
        ocr_base_url="https://vision.example/v1",
        ocr_model="vision-model",
        ocr_default_headers=headers,
    )
    parser._get_markitdown()

    assert openai_calls == [
        {
            "api_key": "parser-key",
            "base_url": "https://vision.example/v1",
            "default_headers": {**headers, "User-Agent": OPENAI_COMPATIBLE_USER_AGENT},
        }
    ]


def test_markitdown_converter_requires_model_when_ocr_enabled(monkeypatch):
    """OCR should fail fast when enabled without a model."""
    import markitdown

    from hindsight_api.engine.parsers import MarkitdownParser

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)

    with pytest.raises(ValueError, match="no model"):
        MarkitdownParser(ocr_enabled=True, ocr_api_key="parser-key")


def test_markitdown_converter_requires_base_url_when_ocr_enabled(monkeypatch):
    """OCR should fail fast when enabled without a dedicated OpenAI-compatible endpoint."""
    import markitdown

    from hindsight_api.engine.parsers import MarkitdownParser

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)

    with pytest.raises(ValueError, match="no base URL"):
        MarkitdownParser(ocr_enabled=True, ocr_api_key="parser-key", ocr_model="vision-model")


def test_markitdown_converter_reports_missing_openai_when_ocr_enabled(monkeypatch):
    """Missing OpenAI SDK should not be reported as missing MarkItDown."""
    import builtins

    import markitdown

    from hindsight_api.engine.parsers import MarkitdownParser

    real_import = builtins.__import__

    class FakeMarkItDown:
        def __init__(self, **kwargs):
            pass

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(markitdown, "MarkItDown", FakeMarkItDown)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Construction validates the settings but does not import the SDK, so the
    # missing-package error surfaces on first use.
    parser = MarkitdownParser(
        ocr_enabled=True,
        ocr_api_key="parser-key",
        ocr_base_url="https://vision.example/v1",
        ocr_model="vision-model",
    )
    with pytest.raises(RuntimeError, match="openai package is required"):
        parser._get_markitdown()


@pytest.mark.asyncio
async def test_converter_registry():
    """Test file parser registry."""
    from hindsight_api.engine.parsers import FileParserRegistry, MarkitdownParser

    registry = FileParserRegistry()
    parser = MarkitdownParser()
    registry.register(parser)

    # Test get by name
    retrieved = registry.get_parser("markitdown", "test.txt")
    assert retrieved is parser

    # Test auto-detection
    auto = registry.get_parser(None, "test.pdf")
    assert auto is parser

    # Test unsupported format
    with pytest.raises(ValueError, match="No parser found"):
        registry.get_parser(None, "test.xyz")


class FileConvertTrackingValidator(OperationValidatorExtension):
    """Validator that tracks on_file_convert_complete hook calls."""

    def __init__(self):
        super().__init__({})
        self.convert_calls: list[FileConvertResult] = []

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    async def on_retain_complete(self, result: RetainResult) -> None:
        pass

    async def on_recall_complete(self, result: RecallResult) -> None:
        pass

    async def on_file_convert_complete(self, result: FileConvertResult) -> None:
        self.convert_calls.append(result)
