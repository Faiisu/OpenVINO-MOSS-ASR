"""Verify that long-running model diagnostics do not grow without a bound."""

import unittest

from speech_to_text_deploy.model.session import METRIC_HISTORY_LIMIT, OpenVinoSession


class ModelMetricTests(unittest.TestCase):
    def test_recent_history_stays_bounded_while_total_count_keeps_growing(self):
        # Construction does not load model weights or allocate GPU tensors.
        session = OpenVinoSession("test-model", "unused")
        for index in range(1, 10001):
            session._record_batch_metrics({"index": index})
        self.assertEqual(session.runtime_metrics["openvino_batch_count"], 10000)
        history = session.runtime_metrics["openvino_batches"]
        self.assertEqual(len(history), METRIC_HISTORY_LIMIT)
        self.assertEqual(history[0]["index"], 10000 - METRIC_HISTORY_LIMIT + 1)
        self.assertEqual(history[-1]["index"], 10000)
