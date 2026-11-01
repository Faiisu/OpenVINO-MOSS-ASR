# MOSS queued speech collection deployment

This package contains only the application modules needed for the queued
BY-V/OpenVINO workflow from the research notebook. All its application imports,
including the isolated worker, stay inside `speech_to_text_deploy`. It does not
import the original `speech_to_text` package. Model weights, input recordings,
the project `.venv`, and third-party libraries remain external resources.

The maintained research implementations were extracted into focused modules.
One consumer loads MOSS once, processes captured windows, stores timestamped
utterance WAV/JSON pairs, and unloads after the queue drains. A producer records
audio while inference runs. Transcripts, audio players, and RTF are available
through both the notebook and a portable HTML report.

## Start here

- Edit [settings.json](settings.json) for microphone, model, language, duration,
  token limit, replay and output settings. CLI and notebook use these same defaults.
- Edit [model/prompts.py](model/prompts.py) for language/timestamp instructions
  appended to the upstream model's base prompt. The session supplies the actual
  window duration automatically.
- Run [output_gathering.py](output_gathering.py) for continuous data collection.
- Open [output_gathering.ipynb](output_gathering.ipynb) for optional interactive analysis.
- Use `uv run --no-project --python .venv/bin/python -m speech_to_text_deploy run`
  from the project root for command-line collection.

## Structure

```text
speech_to_text_deploy/
  README.md
  settings.json                 Shared editable CLI/notebook defaults
  requirements.txt              Runtime dependencies
  requirements-notebook.txt     Runtime plus notebook kernel
  output_gathering.py            Continuous collector; no notebook kernel required
  output_gathering.ipynb         Setup, collection, timing and final summary
  __init__.py
  __main__.py                   Public run/summary commands
  config.py                     Typed config, paths, CSV schema, run names
  pipeline.py                   Two workers, queue, reporting, process supervisor API
  worker.py                     Private supervised child-process entry point
  runtime.py                    Worker deadlines, environment and shared cleanup
  audio/
    __init__.py
    capture.py                  BY-V capture, file replay and partial windows
    io.py                       Native-precision audio I/O and frame copying
  model/
    __init__.py                 Tensor-free Segment schema
    prompts.py                  Editable prompt composition
    inputs.py                   CPU features and decoder left padding
    session.py                  Persistent OpenVINO encoder/decoder session
  outputs/
    __init__.py
    clips.py                    Interval validation and paired WAV/JSON storage
    report.py                   Transcript/audio/RTF HTML rendering
  tests/
    __init__.py
    test_entrypoints.py         Import independence, shared settings and prompts
    test_pipeline.py            Native samples, capture, queue and rejection recovery
    test_report.py              Timing, embedded WAVs and summary CLI
    test_runtime.py             Actual child-process timeout/interrupt cleanup
    test_model_metrics.py       Bounded recent model diagnostics for long runs
```

## Environment and model

Use the existing project-local `.venv`, managed with `uv`. Runtime installation
from the project root:

```bash
uv pip install --python .venv/bin/python -r speech_to_text_deploy/requirements.txt --torch-backend=xpu
```

For the notebook, install the notebook dependency set instead:

```bash
uv pip install --python .venv/bin/python -r speech_to_text_deploy/requirements-notebook.txt --torch-backend=xpu
```

The existing OpenVINO export must be present in `.tools/moss-openvino-fp16`, or
set `model_dir` in the settings to another compatible export. Required assets
include the manifest, encoder/decoder graphs and weights, embeddings, tokenizer
and processor files. Export is an offline preparation step; the deployment uses
the prepared model and does not export or download model weights during a run.
The existing [export notebook](../research/openvino_iGPU.ipynb) can prepare it.

The default runtime is OpenVINO on Intel GPU with CPU preprocessing. The worker
sets the OpenCL library environment before loading the model. It does not change
system drivers. Device indices refer to PortAudio input indices; the default
selects the only BY-V input. A one-second input buffer tolerates short scheduling
pauses; an actual audio overflow remains fatal to protect timestamp integrity.

## Continuous data collection

Run the Python script from the project root:

```bash
uv run --no-project --python .venv/bin/python speech_to_text_deploy/output_gathering.py
```

The module form is equivalent:

```bash
uv run --no-project --python .venv/bin/python -m speech_to_text_deploy.output_gathering
```

This script uses the shared settings but overrides `max_audio_seconds` to `None`
and disables the supervisor's elapsed-time deadline. It therefore does not stop
at the notebook's 120 seconds or the bounded CLI's one-hour deadline. A model is
loaded once and kept for the recording session. Default windows remain 30 seconds.
An explicit `--duration 60` runs a bounded 60-second trial instead; `--timeout`
can set an explicit process deadline. The existing CLI also supports
`python -m speech_to_text_deploy run --continuous`.

Press Ctrl+C once, send SIGTERM, or create an empty `STOP` file in the printed run
directory. All three request capture stop, final partial-window preservation,
queue drain and model cleanup. The drain deadline remains 600 seconds by default
and can be changed with `--drain-timeout`. File replay ends at EOF. Capture/model/
storage errors, the free-disk reserve and the system-RAM reserve still stop the
run explicitly; this script does not silently restart a failed session.

Transcript WAV/JSON pairs and CSV timing rows are written as windows finish.
`runtime.json` is written after drain/cleanup. Full CSV history stays on disk;
the collector keeps aggregate counters and only the last 100 model-batch metrics
in memory. `openvino_batch_count` records the full number of completed batches.
An inference backlog can still grow disk usage and the pending-file queue.

Continuous runs print the reports directory after stopping. Generate the full
embedded-audio HTML summary afterward with the separate `summary` command below.
It is not generated automatically at shutdown, because reading every clip from
a long-running collection could consume excessive RAM. The complete transcript,
audio and RTF dataset is retained regardless of HTML generation.

## Bounded CLI collection

The default settings record 120 seconds in four 30-second windows:

```bash
uv run --no-project --python .venv/bin/python -m speech_to_text_deploy run
```

Replay a real recording through the same capture/queue/model/storage workflow:

```bash
uv run --no-project --python .venv/bin/python -m speech_to_text_deploy run \
  --source-audio audio/standby/subway-test-02-thai_n_eng.wav \
  --duration 65 --fast-replay
```

Omit `--fast-replay` to simulate acquisition pacing. Unpaced replay deliberately
builds a backlog, so its queue-inclusive RTF is not live microphone performance.
`--config` selects another JSON settings file, `--output-dir` selects a new exact
run directory, and `--duration` overrides the capture duration. Paths inside JSON
are relative to the project root; CLI input/output path overrides are relative
to the shell's working directory. Run `run --help` for the shutdown deadlines.

Set `max_audio_seconds` to `null` in settings to record until stopped, subject to
the worker timeout (one hour by default). Press Ctrl+C once or create an empty
`STOP` file inside the printed run directory. Capture stops and its last partial
window is queued; inference then drains the queue within the drain deadline.
The supervisor terminates a stuck worker. Raw recordings remain recoverable.

## Run with the notebook

Select the project `.venv` kernel and run all cells. The setup reads the shared
settings; rerun it after editing settings. Notebook variables can override those
settings for one experiment. Model weights are loaded only in the supervised
child, with cleanup in `finally`; they are not loaded into the notebook kernel.

The final cell reads the current `run_dir`. It shows all window statuses/timings,
transcripts and speakers, effective recording/window timestamps, playable saved
clips and duration-weighted RTF. Rejected model intervals retain their text and
original model times, with no invented audio player. Set `run_dir` to an existing
run before executing that cell to inspect historical results.

## Outputs and timing

By default, outputs remain under `research_output_gathering/output/` at the
project root, as in the research workflow. Set `output_base` to change it.
Every run creates a new directory:

```text
run-<UTC>-<id>/
  config.json
  recordings/segment-000001.wav + .json
  results/segment-000001/
    raw_transcript.txt
    transcript.txt
    transcript.json
    utterance-0001.wav + .json
  reports/
    segments.csv
    runtime.json
    summary.html                Automatic for bounded runs; on request for continuous runs
```

Create or refresh the portable summary for any run without loading the model:

```bash
uv run --no-project --python .venv/bin/python -m speech_to_text_deploy summary /absolute/path/to/run
```

The summary embeds original WAV bytes for playback in Remote SSH and offline
browsers. Long runs can produce large HTML/notebook outputs. Each player starts
at zero within its saved clip; labels show the corresponding recording time.
Replay preserves the source rate, channels and PCM/float precision. Live capture
uses mono PCM16 at 48 kHz by default. UTC timestamps are host-clock estimates;
relative offsets and clipping are sample accurate. Speaker IDs are independent
per window and words spanning a window boundary can be split.

The model receives a 0-30 s instruction by default, shortened for a partial
window. This is a prompt, not a decoding constraint. Valid overlapping intervals
are exported independently; end times extending beyond the source are clamped.
Invalid intervals are recorded in `rejected_segments`. Their window has status
`partial`; the run reports `success_with_warnings` while later windows continue.
Capture, storage, model-runtime and cleanup failures remain fatal. An empty
parsed result is marked `no_segments` and retains the raw model output.

```text
latency_seconds = prepare_seconds + queue_wait_seconds
                + inference_seconds + trim_and_store_seconds
RTF = latency_seconds / actual_window_audio_seconds
duration-weighted RTF = sum(latency_seconds) / sum(audio_seconds)
```

RTF is measured per input window, including queue waiting and local storage,
excluding initial audio acquisition, model compilation, CSV writing and final
unloading. The aggregate includes completed `success` and `partial` windows;
failed/unprocessed windows have no RTF. It is not wall-clock throughput or a
per-utterance measurement. Runtime JSON separately records execution devices,
rejection counts and OpenVINO allocations after unload. PyTorch allocator
statistics cover CPU preprocessing, not OpenVINO GPU memory. The child-process
exit releases driver context memory.

## Validation

Run focused tests from the project root:

```bash
uv run --no-project --python .venv/bin/python -m unittest discover -s speech_to_text_deploy/tests
```

Import-independence tests block the original package while importing every
deployment module. Scheduling/failure unit tests use a small fake session;
real-model validation is reported separately. The same shared cleanup helper
is used for per-call tensor cleanup and model unload, including failures and
interruptions.

Validated locally on 2026-11-01 with `uv` and the project `.venv`:

- All 24 deployment unit/integration tests passed. These include import
  independence, shared settings, real worker shutdown, partial windows, native
  PCM precision, rejection recovery and the summary CLI.
- The public CLI replayed 63 seconds of retained BY-V audio through actual MOSS
  on `GPU.0` as 30/30/3-second windows. Importing the original package was blocked
  in both CLI and worker processes. Five utterance clips matched their source
  PCM exactly, one interval was explicitly rejected, the last window had no
  parsed utterances, and the portable HTML report was saved. All windows drained.
- All deployment notebook cells ran with actual BY-V input and MOSS for 31
  seconds, processing 30/1-second windows and rendering the final summary.
  The kernel blocked original-package imports; six WAV/JSON pairs were saved.
- After the final shared-model-setting update, the full deployment notebook
  replayed the original failing 30-second window. One valid clip and one
  rejection were displayed without aborting the run.
- OpenVINO reported zero device/host allocations after all real-model runs;
  PyTorch allocated bytes returned to zero. No editor diagnostics remained.

Live capture validated operation, not transcript accuracy. Unpaced replay
timings include backlog and must not be presented as live microphone throughput.

After adding the continuous script, all 29 deployment tests passed. Real BY-V
capture and MOSS inference ran without a configured capture limit or supervisor
deadline, then received SIGTERM after the first 30-second window plus 1.8 seconds.
Both windows were processed and allocations returned to zero. The test verifies
startup, continued capture and graceful termination; it is not a multi-day soak
test. Additional tests cover unlimited worker supervision with resource checks,
explicit duration overrides and a 10,000-record diagnostic-history bound.
