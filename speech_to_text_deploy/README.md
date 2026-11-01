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

- Edit [settings.json](settings.json) for all runtime values, including model,
  language, hotwords, audio, VAD, MQTT, worker timeouts and output paths. CLI and
  notebook use these same defaults; command-line flags can override them.
- Prompt formatting lives in [model/prompts.py](model/prompts.py); edit the
  `hotwords` and `languages` values in `settings.json` instead of changing code.
  The session supplies the actual audio duration automatically.
- Run [output_gathering.py](output_gathering.py) for continuous data collection.
- Each captured window passes through Silero VAD before MOSS. The defaults use
  speech probability threshold `0.9` and minimum speech duration `1000 ms`;
  windows with no qualifying speech are retained and skipped by MOSS.
- Open the [live monitor](http://localhost:18000/) while collecting. Start its
  independent server with the command under **Live output monitor**.
- Open [output_gathering.ipynb](output_gathering.ipynb) for optional interactive analysis.
- Use `uv run --no-project --python .venv/bin/python -m speech_to_text_deploy run`
  from the project root for command-line collection.

The main controls in `settings.json` are grouped by purpose here:

| Purpose | Settings |
| --- | --- |
| Model | `model_dir`, `model_id`, `ov_device`, `max_new_tokens`, `cpu_threads` |
| Prompt | `languages`, `hotwords` |
| Speech filter | `vad_threshold`, `vad_min_speech_duration_ms`, `vad_min_silence_duration_ms`, `vad_speech_pad_ms` |
| Audio | `input_device`, `samplerate`, `input_latency_seconds`, `segment_seconds`, `max_audio_seconds`, `source_audio`, `replay_realtime` |
| MQTT | `mqtt_enabled`, `mqtt_host`, `mqtt_port`, `mqtt_topic`, `mqtt_qos` and connection/publish timeouts |
| Audio clips | `greeting_preroll_keyword`, `greeting_preroll_seconds` |
| Time and runtime | `internet_time_timeout_seconds`, `live_audio_enabled`, `live_audio_url`, worker/drain timeouts, `min_free_bytes`, `output_base` |

## Structure

```text
speech_to_text_deploy/
  README.md
  settings.json                 Shared editable CLI/notebook defaults
  requirements.txt              Runtime dependencies
  requirements-notebook.txt     Runtime plus notebook kernel
  output_gathering.py            Continuous collector; no notebook kernel required
  output_gathering.ipynb         Setup, collection, timing and final summary
  run_model.sh                   Start the continuous collector with project uv/.venv
  run_frontend.sh                Start the monitor on 0.0.0.0:18000
  __init__.py
  __main__.py                   Public run/summary commands
  config.py                     Typed config, paths, CSV schema, run names
  pipeline.py                   Two workers, queue, reporting, process supervisor API
  worker.py                     Private supervised child-process entry point
  runtime.py                    Worker deadlines, environment and shared cleanup
  audio/
    __init__.py
    capture.py                  BY-V capture, file replay and partial windows
    live_audio.py               Nonblocking audio-chunk forwarding to the monitor
    io.py                       Native-precision audio I/O and frame copying
    vad.py                      Silero speech detection before MOSS inference
  model/
    __init__.py                 Tensor-free Segment schema
    prompts.py                  Editable prompt composition
    inputs.py                   CPU features and decoder left padding
    session.py                  Persistent OpenVINO encoder/decoder session
  outputs/
    __init__.py
    clips.py                    Interval validation and paired WAV/JSON storage
    report.py                   Transcript/audio/RTF HTML rendering
    mqtt.py                     JSON transcript reports over MQTT
  monitor/
    __init__.py
    data.py                     Incremental CSV index and saved-run snapshots
    live_audio.py               Bounded live PCM fan-out for browser listeners
    labels.py                   Reviewed labels and JSONL dataset export
    server.py                   HTTP/API/audio server on port 18000
    static/
      index.html                Live dashboard
      styles.css                Responsive monitor layout
      app.js                    Live polling, history, filtering and playback
  tests/
    __init__.py
    test_entrypoints.py         Import independence, shared settings and prompts
    test_pipeline.py            Native samples, capture, queue and rejection recovery
    test_report.py              Timing, embedded WAVs and summary CLI
    test_runtime.py             Actual child-process timeout/interrupt cleanup
    test_model_metrics.py       Bounded recent model diagnostics for long runs
    test_monitor.py             Incremental appends, HTTP, ranges and path isolation
  labelled_dataset/
    README.md                   Exported MOSS JSONL/audio format
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

This script uses the shared settings but overrides `max_audio_seconds` to `None`.
Its worker deadline comes from `continuous_worker_timeout_seconds` (default
`null`, so collection continues until stopped). A model is loaded once and kept
for the recording session. Default windows remain 30 seconds. An explicit
`--duration 60` runs a bounded 60-second trial instead; `--timeout` overrides the
configured deadline for that run. The existing CLI also supports
`python -m speech_to_text_deploy run --continuous`.

Press Ctrl+C once, send SIGTERM, or create an empty `STOP` file in the printed run
directory. All three request capture stop, final partial-window preservation,
queue drain and model cleanup. The drain deadline comes from
`drain_timeout_seconds` (default 600) and can be overridden with `--drain-timeout`.
File replay ends at EOF. Capture/model/
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
`continuous_worker_timeout_seconds`. Press Ctrl+C once or create an empty
`STOP` file inside the printed run directory. Capture stops and its last partial
window is queued; inference then drains the queue within the drain deadline.
The supervisor terminates a stuck worker. Raw recordings remain recoverable.

## MQTT transcript reports

When `mqtt_enabled` is true, the collector publishes one UTF-8 JSON message to
`mqtt_topic` for each accepted utterance. The default broker is `127.0.0.1:1883`
and the default topic is `ASR/v1/report`.
Delivery uses QoS 1 and waits for the broker acknowledgement before marking the
window complete. If the broker cannot be reached or a publish is not acknowledged,
the window is marked failed while its locally saved recording/transcript remains
available. Set `mqtt_enabled` to false to collect without a broker.

Each MQTT message contains only `transcript`, `start_time` and `end_time`, as
UTF-8 JSON. Windows with no accepted transcript produce no MQTT message.
Start/end are the model's utterance offsets anchored to the segment's captured
start time, in ISO 8601 UTC.

At capture startup, the worker tries Internet HTTP `Date` headers, with a bounded
timeout from `internet_time_timeout_seconds`; if Internet time is unavailable,
it anchors timestamps to the computer clock. The selected source is recorded in
the local segment metadata and run report. MQTT does not include WAV bytes.

For utterance clips whose transcript contains `greeting_preroll_keyword`
(`สวัสดี` by default), the saved WAV starts up to
`greeting_preroll_seconds` (2 seconds by default) before the model timestamp.
The clip start is clamped at the beginning of its captured window; model times
remain unchanged while effective clip and recording timestamps include the added
context.

## Live audio listening

With `live_audio_enabled` true, capture sends mono PCM16 chunks asynchronously
to the local monitor while recording continues. Open the frontend and press
**Listen live**; the browser plays those chunks as they arrive, with volume control.
This stream is independent of ASR inference and does not save another audio copy.
If the monitor is unavailable, ASR capture continues and the forwarder retries.

The live publish endpoint accepts only loopback requests. The browser stream is
available to anyone who can reach the monitor on port 18000, so an active listener
can hear the microphone audio in real time.

## Live output monitor

From the project root, start these in separate terminals:

```bash
./speech_to_text_deploy/run_model.sh
./speech_to_text_deploy/run_frontend.sh
```

The model script runs continuously until interrupted. The frontend script serves
the dashboard on port 18000 for reachable IPv4 clients. Press `Ctrl+C` in each
terminal to stop its process. The scripts locate the project root themselves and
use only the project `.venv` through `uv`.

Run the monitor independently of the collector, from the project root:

```bash
uv run --no-project --python .venv/bin/python -m speech_to_text_deploy.monitor.server \
  --host 0.0.0.0 --port 18000
```

Open `http://<server-ip>:18000/`. The default bind address is already `0.0.0.0`
and the default port is `18000`: all reachable IPv4 clients can view the dashboard
without a token or login. There is no IP allowlist in the application. Anyone
who can reach the site can edit labels and trigger dataset export; it does not
expose collector start/stop actions or arbitrary filesystem browsing. A VS Code background task named **MOSS live output monitor
· 18000** is provided for this workspace.

The server reads `output_base` from the shared settings. Use `--output-base` to
monitor another collection directory. It uses the Python standard library and
local HTML/CSS/JavaScript; no extra packages, model process, frontend build or
external CDN are needed. Starting/stopping the web server does not stop capture.

The dashboard refreshes every second. Results appear when each window finishes
inference and local storage, rather than token by token during generation. It
shows processed audio, clip/rejection counts, captured/pending windows, weighted
RTF, recent timing charts, source/device information, and text paired with its
timestamped audio. Rejections are displayed with reasons and no invented clip.
Activity status is inferred from saved files until final runtime JSON exists.

Select a run, follow the newest run automatically, search visible text/speakers,
edit transcript and speaker labels, save selections for finetuning, pause updates,
or page through older windows. History view is static until
**Back to live** is selected. Searching applies to the displayed windows.
Refresh preserves existing audio elements so playing a clip is not reset.
Audio is streamed from saved WAVs with byte-range support for seeking, without
base64-embedding every recording in each refresh.

CSV indexing reads only new complete records, tolerates partial/multiline writes,
and retains 200 recent rows plus aggregate counters per cached run. Historical
paging reads older CSV rows on demand. Index/manifest caches are bounded; the
monitor does not retain the entire collection in RAM. HTTP audio routes are
limited to generated clip paths within the selected output run.

For each accepted utterance, edit the transcript and speaker ID and select
**Include in dataset**, then choose **Save label**. Reviewed labels are stored in
`results/segment-*/labels.json`, separate from collector transcripts. **Export
selected dataset** gathers all saved selections across runs into
`speech_to_text_deploy/labelled_dataset/dataset.jsonl` and copies their WAVs to
`speech_to_text_deploy/labelled_dataset/audio/`. JSONL targets use clip-relative
timestamps, `[S01]` speaker IDs, and reviewed text. Each row keeps its source
run/utterance and original recording-time provenance in a `source` field. The
export appends only new utterances and skips source identities or byte-identical
audio already in the manifest; existing JSONL rows are kept unchanged. The audio
and JSONL files are generated locally and excluded from Git; the dataset README
documents the official MOSS conversation format and its limitations.

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

RTF is measured per input window, including VAD, queue waiting and local storage,
excluding initial audio acquisition, model compilation, CSV writing and final
unloading. The aggregate includes completed `success`, `partial` and `no_speech` windows;
failed/unprocessed windows have no RTF. It is not wall-clock throughput or a
per-utterance measurement. Runtime JSON separately records execution devices,
rejection counts and OpenVINO allocations after unload. PyTorch allocator
statistics cover CPU preprocessing, not OpenVINO GPU memory. The child-process
exit releases driver context memory.

There is no per-section processing deadline. A section may take longer than its
audio duration while the model generates tokens. The worker still has the
overall deadline configured by `bounded_worker_timeout_seconds` (one hour by
default); continuous mode uses `continuous_worker_timeout_seconds`. Use Ctrl+C or the run
directory's `STOP` file to request a graceful stop and queue drain. Captured
source WAVs and completed earlier results remain on disk if a run is interrupted.

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

After adding the monitor, all 33 deployment tests passed. HTTP was verified on
loopback, the host LAN address and its Tailscale address while the original
collector continued running. The browser displayed actual new windows, loaded
WAV metadata, fetched byte ranges with HTTP 206, preserved audio elements across
refresh, filtered transcripts and fit a 390-pixel mobile viewport. Monitor tests
also cover partial multiline CSV appends, exact audio bytes and traversal/symlink
isolation. No additional model was loaded for web monitoring.
