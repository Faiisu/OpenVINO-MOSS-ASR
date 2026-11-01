"""Check XPU attention routing and restoration after failed inference."""
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from speech_to_text.runtime import inference_attention


class AttentionTests(unittest.TestCase):
    def test_xpu_disables_fused_backends_and_restores_them_on_failure(self):
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.OVERRIDEABLE]):
            with self.assertRaises(KeyboardInterrupt):
                with inference_attention("xpu:0"):
                    self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                    self.assertFalse(torch._C._get_overrideable_sdp_enabled())
                    self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                    raise KeyboardInterrupt
            self.assertTrue(torch._C._get_overrideable_sdp_enabled())

    def test_other_devices_keep_existing_attention_policy(self):
        with sdpa_kernel(SDPBackend.OVERRIDEABLE):
            for device in ("cpu", "cuda:0"):
                with self.subTest(device=device), inference_attention(device):
                    self.assertTrue(torch._C._get_overrideable_sdp_enabled())
                    self.assertFalse(torch.backends.cuda.math_sdp_enabled())


if __name__ == "__main__":
    unittest.main()
