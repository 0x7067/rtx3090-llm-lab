"""Ensure saved prefixes are rejected after tokenizer or model changes."""
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import warm_saved_prefixes as warmer

class PrefixTests(unittest.TestCase):
    def test_gateway_effort_and_shared_block_boundary(self):
        source = Path(__file__).parent / "optimization/prefix-mtp/warm_prefix.py"
        spec = importlib.util.spec_from_file_location("prefix_capture", source)
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        payload = {"messages": [], "reasoning_effort": "none"}
        self.assertFalse(capture.tokenize_payload(payload)["chat_template_kwargs"]["enable_thinking"])
        routed = capture.tokenize_payload(payload, "low")["chat_template_kwargs"]
        self.assertTrue(routed["enable_thinking"])
        self.assertEqual(routed["reasoning_effort"], "low")
        self.assertEqual(payload["reasoning_effort"], "none")
        tokens, count = capture.shared_prefix([[1] * 900 + [2], [1] * 900 + [3]])
        self.assertEqual((len(tokens), count), (864, 900))
        with self.assertRaises(ValueError):
            capture.shared_prefix([[1] * 900])

    def test_model_tokenizer_binding_and_token_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "tokenizer.json").write_bytes(b"original tokenizer")
            bundle = {"model": str(model), "tokenizer_sha256": hashlib.sha256(b"original tokenizer").hexdigest(), "prefixes": [[1] * 864]}
            self.assertEqual(len(warmer.validate(bundle, model)[0]), 864)
            with self.assertRaises(ValueError):
                warmer.validate(bundle, model / "other")
            bundle["prefixes"] = [[1] * 863]
            with self.assertRaises(ValueError):
                warmer.validate(bundle, model)
            bundle["prefixes"] = [[True] * 864]
            with self.assertRaises(ValueError):
                warmer.validate(bundle, model)
            bundle["prefixes"] = [[1] * 864]
            (model / "tokenizer.json").write_bytes(b"replacement tokenizer")
            with self.assertRaises(ValueError):
                warmer.validate(bundle, model)

    def test_gguf_with_separate_tokenizer_keeps_model_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"model file")
            tokenizer = root / "config" / "tokenizer.json"
            tokenizer.parent.mkdir()
            tokenizer.write_bytes(b"separate tokenizer")
            bundle = {"model": str(model),
                      "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                      "prefixes": [[1] * 864]}
            self.assertEqual(warmer.validate(bundle, model, tokenizer), [[1] * 864])
            with self.assertRaises(ValueError):
                warmer.validate(bundle, root / "different.gguf", tokenizer)
            tokenizer.write_bytes(b"changed tokenizer")
            with self.assertRaises(ValueError):
                warmer.validate(bundle, model, tokenizer)

    def test_invalid_file_never_reaches_backend_and_remains_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "prefixes.json"
            saved.write_text("invalid json")
            with patch.dict(os.environ, {"SPARSEKV_WARM_PREFIX_FILE": str(saved)}), patch.object(warmer, "urlopen") as request:
                warmer.main()
                request.assert_not_called()

if __name__ == "__main__":
    unittest.main()
