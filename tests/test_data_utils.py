"""Tests for data utilities."""

import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatdoctor.utils.data_utils import (
    load_json,
    save_json,
    format_medical_conversation,
    chunk_text,
)


class TestJsonUtils:
    """Tests for JSON utilities."""

    def test_save_and_load(self):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_json(data, path)

            loaded = load_json(path)
            assert loaded == data

    def test_nested_directory_creation(self):
        data = {"test": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "test.json"
            save_json(data, path)

            assert path.exists()
            loaded = load_json(path)
            assert loaded == data


class TestFormatMedicalConversation:
    """Tests for medical conversation formatting."""

    def test_format(self):
        result = format_medical_conversation(
            patient_question="I have a headache",
            doctor_response="You may have tension headache",
        )

        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert result["input"] == "I have a headache"
        assert result["output"] == "You may have tension headache"
        assert "doctor" in result["instruction"].lower()


class TestChunkText:
    """Tests for text chunking."""

    def test_basic_chunking(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, max_words=25)

        assert len(chunks) == 4
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= 25

    def test_short_text(self):
        text = "short text"
        chunks = chunk_text(text, max_words=25)

        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_empty_text(self):
        chunks = chunk_text("", max_words=25)
        assert chunks == [""]
