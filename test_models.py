"""Unit tests for models.py"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from models import Model, load, _is_cache_stale, DATA_FILE


class TestIsCacheStale(unittest.TestCase):
    def test_missing_file_is_stale(self):
        with patch("models.DATA_FILE", "/nonexistent/path.json"):
            self.assertTrue(_is_cache_stale())

    def test_fresh_file_is_not_stale(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"[]")
            path = f.name
        try:
            with patch("models.DATA_FILE", path):
                self.assertFalse(_is_cache_stale())
        finally:
            os.unlink(path)


class TestLoadModels(unittest.TestCase):
    def test_load_from_valid_json(self):
        sample = [
            {
                "name": "test-model",
                "provider": "test",
                "parameter_count": "1B",
                "parameters_raw": 1_000_000_000,
                "min_ram_gb": 2.0,
                "recommended_ram_gb": 4.0,
                "min_vram_gb": 2.0,
                "quantization": "Q4_K_M",
                "context_length": 4096,
                "use_case": "general",
                "capabilities": ["chat"],
                "architecture": "transformer",
                "hf_downloads": 500,
                "hf_likes": 50,
                "release_date": "2025-01-01",
            }
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample, f)
            path = f.name
        try:
            with patch("models.DATA_FILE", path):
                models = load()
                self.assertEqual(len(models), 1)
                self.assertEqual(models[0].name, "test-model")
                self.assertEqual(models[0].parameters_raw, 1_000_000_000)
        finally:
            os.unlink(path)

    def test_load_corrupt_json_exits(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            path = f.name
        try:
            with patch("models.DATA_FILE", path):
                with self.assertRaises(SystemExit):
                    load()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
