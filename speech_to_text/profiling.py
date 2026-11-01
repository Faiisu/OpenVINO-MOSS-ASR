"""Opt-in synchronized stage timings and tensor-free generation observations."""
import csv
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter


def stage(profiler, name, **metadata):
    """Keep the normal inference path uninstrumented when profiling is disabled."""
    return profiler.stage(name, **metadata) if profiler is not None else nullcontext()


class StageProfiler:
    """Measure nested wall time; synchronize XPU boundaries for completed work."""

    def __init__(self, output_dir, *, trace_batch=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = None
        self.events = []
        self.batches = []
        self.stack = []
        self.trace_batch = trace_batch
        self.started = perf_counter()

    def synchronize(self):
        if self.device is not None:
            import torch
            torch.xpu.synchronize(self.device)

    @contextmanager
    def stage(self, name, *, synchronize=True, **metadata):
        if synchronize:
            self.synchronize()
        event = {"id": len(self.events), "parent_id": self.stack[-1] if self.stack else None,
                 "name": name, "start_seconds": perf_counter() - self.started, **metadata}
        self.events.append(event)
        self.stack.append(event["id"])
        started = perf_counter()
        try:
            yield
        finally:
            try:
                if synchronize:
                    self.synchronize()
            finally:
                event["seconds"] = perf_counter() - started
                self.stack.pop()

    @contextmanager
    def generation(self, model, inputs, use_cache):
        """Time top-level prefill/decode and the encoder with removable hooks."""
        import torch

        batch = {"index": len(self.batches) + 1, "use_cache": bool(use_cache),
                 "input_shapes": {key: list(value.shape) for key, value in inputs.items()
                                  if isinstance(value, torch.Tensor)},
                 "prefill_calls": 0, "decode_calls": 0, "cache_observed": False}
        self.batches.append(batch)
        handles, active = [], defaultdict(list)

        def attach(module, label, top_level=False):
            def before(_module, args, kwargs):
                name = label
                if top_level:
                    cache = kwargs.get("past_key_values")
                    length = cache.get_seq_length() if cache is not None else 0
                    batch["cache_observed"] |= bool(length > 0)
                    name = "decode_forward" if batch["prefill_calls"] else "prefill_forward"
                    batch["decode_calls" if batch["prefill_calls"] else "prefill_calls"] += 1
                context = self.stage(name, batch_index=batch["index"])
                context.__enter__()
                active[label].append(context)

            def after(_module, args, kwargs, output):
                if active[label]:
                    active[label].pop().__exit__(None, None, None)
                if top_level and trace is not None:
                    trace.step()

            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))

        trace = None
        def export_trace(trace_result):
            trace_result.export_chrome_trace(str(self.output_dir / f"batch-{batch['index']}-trace.json"))
            rows = [{"operator": item.key, "calls": item.count,
                     "self_cpu_time_us": item.self_cpu_time_total,
                     "self_device_time_us": item.self_device_time_total}
                    for item in trace_result.key_averages()]
            self.write_csv(self.output_dir / "operators.csv", rows)

        try:
            attach(model, "model", True)
            attach(model.model.whisper_encoder, "audio_encoder")
            attach(model.model.vq_adaptor, "audio_adaptor")
            attach(model.model.language_model, "language_model")
            if batch["index"] == self.trace_batch:
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
                trace = torch.profiler.profile(
                    activities=activities, record_shapes=True, profile_memory=True, with_stack=False,
                    schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1),
                    on_trace_ready=export_trace)
            with trace if trace is not None else nullcontext():
                yield batch
        finally:
            for handle in handles:
                handle.remove()
            # Hooks normally close on exceptions, too; retain no active frames.
            for contexts in active.values():
                while contexts:
                    contexts.pop().__exit__(None, None, None)
            del trace

    @staticmethod
    def write_csv(path, rows):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with Path(path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def save(self):
        """Export inclusive/exclusive stages without counting nested time twice."""
        child_seconds = defaultdict(float)
        for event in self.events:
            child_seconds[event["parent_id"]] += event.get("seconds", 0)
        summary = {}
        for event in self.events:
            event["exclusive_seconds"] = max(0, event.get("seconds", 0) - child_seconds[event["id"]])
            row = summary.setdefault(event["name"], {"stage": event["name"], "calls": 0,
                                                     "inclusive_seconds": 0., "exclusive_seconds": 0.})
            row["calls"] += 1
            row["inclusive_seconds"] += event.get("seconds", 0)
            row["exclusive_seconds"] += event["exclusive_seconds"]
        observed = perf_counter() - self.started
        measured = sum(row["exclusive_seconds"] for row in summary.values())
        payload = {"timing_scope": "Synchronized instrumented wall time, including hook/synchronization overhead. Inclusive stages contain children; rank exclusive time to avoid double counting. Device kernel time is available only in the optional operator trace.",
                   "worker_observed_seconds": observed,
                   "unattributed_seconds": max(0, observed - measured),
                   "events": self.events, "summary": list(summary.values()), "batches": self.batches}
        (self.output_dir / "profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.write_csv(self.output_dir / "stages.csv", self.events)
        self.write_csv(self.output_dir / "stage_summary.csv", payload["summary"])
