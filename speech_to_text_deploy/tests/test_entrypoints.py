"""Verify import independence, prompt boundaries and CLI entry points."""

import os
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from speech_to_text_deploy.__main__ import main as cli_main
from speech_to_text_deploy.output_gathering import main as collect_main
from speech_to_text_deploy.config import ROOT, load_settings, new_run_dir
from speech_to_text_deploy.model.prompts import transcription_prompt


class EntrypointTests(unittest.TestCase):
    def test_continuous_script_removes_both_caps_and_duration_can_override_it(self):
        cases = [(collect_main, [], None, None),
                 (collect_main, ["--duration", "31"], 31, 3600),
                 (cli_main, ["run", "--continuous"], None, None),
                 (cli_main, ["run"], 120, 3600)]
        result = {"status": "success", "processed_count": 0, "rejected_segment_count": 0}
        with tempfile.TemporaryDirectory() as folder:
            for entrypoint, arguments, duration, timeout in cases:
                with self.subTest(arguments=arguments, entrypoint=entrypoint.__module__):
                    with patch("speech_to_text_deploy.__main__.gather_isolated", return_value=result) as worker:
                        with redirect_stdout(io.StringIO()):
                            entrypoint([*arguments, "--output-dir", str(Path(folder) / "new-run")])
                        self.assertEqual(worker.call_args.args[0].max_audio_seconds, duration)
                        self.assertEqual(worker.call_args.kwargs["timeout_seconds"], timeout)

    def test_script_can_run_by_absolute_path_outside_project(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "speech_to_text_deploy/output_gathering.py"), "--help"],
                                    cwd=folder, check=True, capture_output=True, text=True)
            self.assertIn("--continuous", result.stdout)
            self.assertIn("--duration", result.stdout)

    def test_shared_settings_resolve_paths_from_project_root(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"model_dir": "models/moss", "source_audio": "audio/input.wav",
                                        "output_base": "results", "max_audio_seconds": 31,
                                        "hotwords": ["Subway", "ซับเวย์"],
                                        "vad_threshold": 0.9, "vad_min_speech_duration_ms": 1000,
                                        "bounded_worker_timeout_seconds": 7200}))
            values = load_settings(path)
            self.assertEqual(values["model_dir"], str(ROOT / "models/moss"))
            self.assertEqual(values["source_audio"], str(ROOT / "audio/input.wav"))
            self.assertEqual(values["output_base"], str(ROOT / "results"))
            self.assertEqual(values["max_audio_seconds"], 31)
            self.assertEqual(values["hotwords"], ("Subway", "ซับเวย์"))
            self.assertEqual(values["vad_threshold"], 0.9)
            self.assertEqual(values["bounded_worker_timeout_seconds"], 7200)
            self.assertEqual(values["mqtt_host"], "127.0.0.1")
            self.assertEqual(values["mqtt_port"], 1883)
            self.assertEqual(values["mqtt_topic"], "ASR/v1/report")
            self.assertTrue(values["save_audio"])
            self.assertTrue(values["save_transcript"])

    def test_all_deployment_modules_import_with_original_package_blocked(self):
        code = '''
import importlib
import importlib.abc
import pkgutil
import sys
class BlockOriginal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "speech_to_text" or fullname.startswith("speech_to_text."):
            raise ImportError("Original package must not be imported: " + fullname)
sys.meta_path.insert(0, BlockOriginal())
import speech_to_text_deploy
for module in pkgutil.walk_packages(speech_to_text_deploy.__path__, "speech_to_text_deploy."):
    if ".tests" not in module.name:
        importlib.import_module(module.name)
assert not any(name == "speech_to_text" or name.startswith("speech_to_text.") for name in sys.modules)
'''
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                       env=os.environ.copy(), capture_output=True, text=True)

    def test_cli_help_runs_from_another_directory(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, "-m", "speech_to_text_deploy", "--help"],
                                cwd=Path(__file__).parent, env=environment, check=True,
                                capture_output=True, text=True)
        self.assertIn("run", result.stdout)
        self.assertIn("summary", result.stdout)

    def test_prompt_bounds_and_unique_run_names(self):
        self.assertEqual(transcription_prompt("Base.", None), "Base.")
        for duration in (30, 3.125):
            prompt = transcription_prompt("Base.", ["Thai", "English"],
                                          hotwords=["Subway", "ซัพเวย์", "ซับเวย์"],
                                          audio_duration_seconds=duration)
            self.assertIn(f"0 <= start < end <= {duration} seconds", prompt)
            self.assertIn("Thai, English", prompt)
            self.assertIn("Hotwords: Subway, ซัพเวย์, ซับเวย์.", prompt)
            self.assertIn("Focus on the keyword Subway", prompt)
            self.assertIn("without adding words not heard.", prompt)
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                transcription_prompt("Base.", None, audio_duration_seconds=duration)
        self.assertNotEqual(new_run_dir(), new_run_dir())

    def test_subway_focus_requires_a_matching_hotword(self):
        for alias in ("SUBWAY", "ซัพเวย์", "ซับเวย์"):
            self.assertIn("Focus on the keyword Subway",
                          transcription_prompt("Base.", None, hotwords=[alias]))
        self.assertNotIn("Focus on the keyword Subway", transcription_prompt("Base.", None, hotwords=["Coffee"]))
