"""Validate dynamic stateful decoding against the source Qwen implementation."""
import unittest

import numpy as np
import torch

from speech_to_text.backend import transcription_prompt
from speech_to_text.openvino_backend import causal_mask
from speech_to_text.runtime import release_model_memory


class OpenVinoTests(unittest.TestCase):
    def test_timestamp_prompt_covers_full_and_partial_audio_windows(self):
        for duration in (30, 3.125):
            prompt = transcription_prompt("Base format.", ["Thai", "English"], audio_duration_seconds=duration)
            self.assertIn(f"spans 0 to {duration} seconds", prompt)
            self.assertIn(f"0 <= start < end <= {duration} seconds", prompt)
            self.assertIn("Thai, English", prompt)
        self.assertEqual(transcription_prompt("Base format.", None), "Base format.")
        for duration in (0, -1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                transcription_prompt("Base", None, audio_duration_seconds=duration)

    def test_causal_left_padding_mask(self):
        mask = np.array([[0, 1, 1], [1, 1, 1]])
        allowed = causal_mask(mask, 3) == 0
        self.assertEqual(allowed[0, 0].tolist(), [[False, False, False],
                                                [False, True, False], [False, True, True]])
        self.assertEqual((causal_mask(mask, 1) == 0)[:, 0, 0].tolist(), mask.astype(bool).tolist())

    def test_stateful_prefill_and_decode_match_source(self):
        try:
            import openvino as ov
        except ImportError:
            self.skipTest("Install requirements-openvino.txt to test conversion.")
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from speech_to_text.openvino_export import DecoderGraph, stateful_decoder

        model = wrapper = graph = compiled = request = result = reference = past = example = None
        try:
            torch.manual_seed(10)
            config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                 vocab_size=50, attention_dropout=0.)
            config._attn_implementation = "eager"
            model = Qwen3ForCausalLM(config).eval()
            wrapper = DecoderGraph(model.model, model.lm_head, config).eval()
            example = (torch.zeros(1, 2, 32), torch.zeros(1, 1, 2, 3), torch.tensor([[1, 2]]),
                       tuple(torch.zeros(1, 1, 1, 16) for _ in range(4)))
            with torch.inference_mode():
                graph = stateful_decoder(ov.convert_model(wrapper, example_input=example), config)
                compiled = ov.Core().compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
                request = compiled.create_infer_request()
                for batch_size in (1, 2):
                    past = tuple(torch.zeros(batch_size, 1, 0, 16) for _ in range(4))
                    for state in request.query_state():
                        state.state = ov.Tensor(np.zeros((batch_size, 1, 0, 16), dtype=np.float32))
                    for step, length in enumerate((3, 1, 1)):
                        embeds = torch.randn(batch_size, length, 32)
                        total = 3 + step
                        mask = torch.from_numpy(causal_mask(np.ones((batch_size, total)), length))
                        positions = torch.arange(total - length, total).expand(batch_size, -1)
                        reference = wrapper(embeds, mask, positions, past)
                        result = request.infer({"embeds": embeds.numpy(), "mask": mask.numpy(),
                                                "positions": positions.numpy()})
                        np.testing.assert_allclose(request.get_tensor("logits").data,
                                                   reference[0].numpy(), rtol=2e-4, atol=2e-5)
                        self.assertTrue(all(list(s.state.shape) == [batch_size, 1, total, 16]
                                            for s in request.query_state()))
                        past = reference[1:]
                    request.reset_state()
        finally:
            model = wrapper = graph = compiled = request = result = reference = past = example = None
            release_model_memory(torch.device("cpu"))
