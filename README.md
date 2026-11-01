# Audio Transcription and Speaker Export

For the self-contained queued BY-V/OpenVINO deployment, start with
[speech_to_text_deploy/README.md](speech_to_text_deploy/README.md). Its CLI,
notebook, settings, audio/model/output modules and tests are grouped inside one
package and do not import the research `speech_to_text` package.

Select a local audio file, transcribe it with [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), and export timestamped speaker transcripts and a WAV file for each speaker.

## Queued BY-V collection

Open [research/output_gathering/output_gathering.ipynb](research/output_gathering/output_gathering.ipynb)
to capture 30-second BY-V microphone windows on one thread and process queued
windows with a persistent MOSS OpenVINO GPU session on another thread. Each
utterance is stored as a local WAV/JSON pair with text, speaker, and timestamps.
Per-window RTF includes queue wait, inference, trimming, and local storage and is
written separately to `reports/segments.csv` for pandas. See the
[collection guide](research/output_gathering/README.md) for timing definitions,
graceful stop, replay, and recovery behavior.

## USB microphone monitoring

Open [audio_stream.ipynb](audio_stream.ipynb) with the project `.venv` kernel to
monitor the remote BY-V USB microphone on the computer running your browser,
including when using VS Code Remote SSH.
Install the optional dependencies with
`uv pip install --python .venv/bin/python -r requirements-audio-stream.txt`.
The default **This computer (browser)** mode starts a loopback HTTP server on port
8765. Forward that port using VS Code's **Ports** panel, open the generated tokenized
link in your local external browser, then click **Play**. If VS Code maps to a
different local port, replace the port in the URL while preserving the token.
The browser controls volume and plays live mono PCM through its own audio output.
Capture starts only while a listener is connected; browser Stop releases the mic,
and notebook Stop also releases the HTTP port. Only one listener is supported.
Keep the SSH forwarding private. Network jitter can cause gaps in this simple
monitoring stream.

The optional **Remote machine speakers** mode retains direct PortAudio playback.
PortAudio device indices differ from ALSA card numbers. Use headphones to avoid
feedback, start with low volume, and click Stop before closing the notebook tab.
This workflow does not save audio or load an ASR model.

## Research notebook

For a direct OpenVINO comparison, open [openvino_iGPU.ipynb](openvino_iGPU.ipynb).
Install its optional dependencies with
`uv pip install --python .venv -r requirements-openvino.txt --torch-backend=xpu`.
It converts the original MOSS encoder/adaptor and Qwen3 decoder directly,
without installing Optimum or changing Transformers. Cached IR weights and
embedding storage use FP16; the OpenVINO GPU precision hint is FP16, while the
PyTorch XPU baseline remains BF16. Stateful KV tensors stay inside the OpenVINO
request and reset independently per batch. Both graphs must report GPU execution;
there is no silent CPU fallback. The same native audio chunks and exports are
used by both runtimes.

The notebook separates one-time conversion, per-worker graph compilation,
preprocessing, encoder, prefill, decode, text parsing, cleanup, and export time.
It saves comparison CSV/JSON, complete stage profiles, transcripts, speaker WAVs,
and text agreement (not reference-based accuracy). PyTorch prefill includes the
encoder; OpenVINO reports its encoder separately. Compare total processing and
workflow time before interpreting individual stage differences. OpenVINO memory
statistics are reported independently from the PyTorch allocator. Workers clear
compiled models/requests on every exit path, then exit the process.

On this machine, OpenVINO needs the system OpenCL compiler environment. Its
worker removes only the project `.tools/intel-iGPU` library directories from
`LD_LIBRARY_PATH`; PyTorch XPU workers retain their existing runtime environment.
Exports are cached under `.tools/moss-openvino-fp16`; the manifest records the
source snapshot and conversion dependency versions. Re-export after changing
the exporter or model source revision. Both conversion and inference can require
several GiB of RAM; worker time and available-memory limits remain enforced.

The executed OpenVINO notebook used the real 117.672-second, 48 kHz mono PCM16
active recording, 30-second chunks, batch size 1, one CPU thread, greedy decoding,
and KV cache. Both graphs ran on `GPU.0`. A batch-2 full baseline initially reached
the shared-RAM reserve; batch 1 completed in both runtimes. A separate real
8-second OpenVINO batch-2 smoke test also passed.

| Single instrumented trial | PyTorch BF16 XPU | OpenVINO FP16 GPU |
|---|---:|---:|
| Processing time, excluding load/compile/exports | 159.727 s | 35.926 s |
| Workflow time, including load/compile/exports | 169.687 s | 45.764 s |
| Processing RTF | 1.357 | 0.305 |
| Decode stage time | 146.757 s | 31.103 s |
| Generated tokens | 788 | 790 |
| Parsed segments | 30 | 30 |

OpenVINO compilation took 4.333 seconds in that trial; the original one-time
conversion took 39.767 seconds and was reused. Processing was 4.45 times faster
in this single run, with different precision and slightly different stage timing
scopes. Text character agreement was 94.60%, not a reference-based accuracy score;
inspect the side-by-side transcripts before selecting a runtime. PyTorch allocated
and reserved memory returned to zero. OpenVINO's own reported device/host allocations
also returned to zero after unloading. Speaker WAVs preserved 48 kHz, mono PCM16.
The new dynamic stateful decoder test compares CPU FP32 logits with the source
Qwen implementation through prefill and multiple decode steps at batch sizes 1
and 2. The complete local test suite passed 26 tests.

For detailed BF16 XPU timing, open [profile_iGPU.ipynb](profile_iGPU.ipynb).
It reuses the research workflow with opt-in synchronized stage instrumentation,
separating load/transfer/preprocessing, prefill, decode, encoder/adaptor/language
model, parsing, cleanup, worker lifecycle, and export time. Reports include
inclusive/exclusive stage CSV/JSON, per-batch shapes and tokens, observed KV-cache
use, experiment medians, and optional bounded CPU/XPU operator traces. The trace
captures the first two forwards of one selected batch. It is disabled by default.
Profiling synchronizations affect performance; these timings include measurement
overhead and must not be treated as uninstrumented latency. The notebook offers
cache/chunk/batch/thread comparisons and documents evidence needed before trying
compilation, OpenVINO, static shapes, async execution, or selective quantization;
those optimizations are not automatically enabled.

For the MOSS-only Intel iGPU experiment, open [research_iGPU.ipynb](research_iGPU.ipynb).
It loads one model per recording and processes `BATCH_SIZE = 2` independent
30-second windows together (use 1 for sequential comparison). The final batch
uses the remaining windows and actual audio durations. Weights are shared;
batched generation requires additional tensor/KV-cache memory.

Timing starts at submission to the loaded session and stops when CPU transcript
and segment outputs are available, with XPU synchronization. It includes audio
preprocessing, generation and parsing, excluding model loading, full-recording
decoding, window WAV writing, allocator cleanup and exports.
`batches` in each `experiment.json` records latency and batch RTF:
`elapsed_seconds / summed_audio_seconds`. Each chunk records its batch latency
and latency RTF (`batch_elapsed_seconds / chunk_duration_seconds`), since all
outputs return together. `chunk_processing_seconds` sums batch times once;
`chunk_processing_rtf` divides that sum by total audio duration. Arithmetic mean
chunk latency differs from total batch time divided by the chunk count.
Top-level `rtf` uses this aggregate submission-to-output measurement;
`workflow_rtf` includes all overhead and exports. `CPU_THREADS` controls
host parallelism; the model still runs on `DEVICE = "xpu:0"`.
Tensor allocations must return to the loaded-model baseline after every batch
and to the pre-model baseline after unloading, including failure and interruption.

Install research dependencies with `uv pip install --python .venv -r requirements-research.txt --torch-backend=xpu`
and select the project `.venv` kernel. This notebook requires an available Intel XPU,
uses BF16 with `QUANTIZATION = "none"`, and records workflow time and
peak/post-cleanup tensor allocations. `quantized_linear_count`, `quantization`,
and `compute_dtype` record the actual loaded configuration. INT8 is disabled by
default in the notebook and isolated worker APIs.
XPU inference selects the PyTorch math SDPA backend to avoid a reproducible
oneDNN fused-attention primitive failure (`CL_OUT_OF_HOST_MEMORY`, followed by
worker exit -11) with two 30-second windows on this Intel runtime. Computation
remains on XPU; math SDPA may use FP32 intermediates and has
different memory and timing characteristics. Each experiment records
`attention_backend = "math_sdpa"`. CPU/CUDA attention policies are unchanged.
Previous BF16 validation with the full active 117.672-second, 48 kHz PCM16 recording succeeded
using BF16, 30-second windows, batch size 2, and one CPU thread: four windows,
two batches, and 30 parsed segments. Processing took 119.064 seconds (RTF 1.012,
excluding model loading and exports). Both batches returned to the loaded-model
allocation baseline; allocated and reserved XPU memory were zero after unloading.
This validates execution and cleanup for this recording, not transcription accuracy.

INT8 validation with the same full recording, math SDPA, 30-second windows,
batch size 2, and one CPU thread also succeeded: 342 INT8 linear layers, four
windows, two batches, and 30 parsed segments. Processing took 287.492 seconds
(RTF 2.443, excluding loading and exports). Both batches returned to the loaded
model allocation baseline of 1,098,940,416 bytes. After unloading, allocation
returned to the pre-model bitsandbytes runtime baseline of 1,024 bytes; a 2 MiB
allocator reservation remained until worker exit. These are individual workflow
trials, not controlled performance or accuracy measurements.
It does not load Typhoon or fall back to CPU. Results go to `outputs/research-iGPU-batch-*`.
The original research notebook retains its CUDA comparison workflow.

Both inference windows and speaker WAV exports preserve the source sample rate,
channel count, and PCM/float precision. PCM and floating WAV samples are copied
without normalization or resampling; FLAC PCM precision is retained in WAV.
Only the model processor resamples/downmixes its inference input as required.
Lossy inputs are decoded at their native rate and channel count to floating WAV;
their codec, encoded bitrate, and container tags are not retained in WAV exports.
Window duration and frame counts necessarily change when slicing. Each experiment
records decoded source properties under `source_audio`.

Open [research.ipynb](research.ipynb) in VS Code and select the project `.venv/bin/python` kernel. It processes recordings in `audio/active/` sequentially and provides experiment settings, raw transcript inspection, a speaker segment table, speaker audio playback, and saved research notes. It reuses the existing inference and export modules.

```bash
export PATH="$PWD/.tools/bin:$PATH"
uv pip install --python .venv -r requirements-research.txt --torch-backend=auto
```

Store unused recordings in `audio/standby/` and move recordings to process into `audio/active/`. Run notebook cells from top to bottom. Only active is scanned, including subfolders; standby is excluded. Rerun discovery after moving files. Each active input is processed sequentially and exported into its own directory under `outputs/research-batch-*`, with errors recorded in `batch_summary.json`. Inputs remain in place. Use `REVIEW_INDEX` to inspect a successful result. Existing recordings were placed in standby during setup; move only the files needed for your experiment into active.

## Setup

Python 3.12 is recommended by upstream. Install Git and Python's Tkinter package using your operating system's package manager. On Ubuntu/Debian, these are typically `git`, `python3-tk`, and `python3-venv`. FFmpeg is bundled by the `imageio-ffmpeg` dependency inside the project `.venv`; the program uses it automatically when FFmpeg is absent from PATH. A graphical desktop is needed only for the file picker.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt --torch-backend=auto
```

Install `uv` before running these commands. If using the project-local bootstrap installation already provided in this workspace, run `export PATH="$PWD/.tools/bin:$PATH"` first. The existing `.venv` in this workspace uses the available Python 3.10 interpreter; upstream supports Python 3.10 or newer and recommends Python 3.12. The commands above are for a fresh Python 3.12 setup. Always run commands from the project root. All project dependencies belong in `.venv`; do not install them into global or system Python. `uv run --no-project --python .venv` uses that environment without activating it. Install a PyTorch build appropriate for your GPU if necessary; consult [PyTorch installation instructions](https://pytorch.org/get-started/locally/). CUDA is recommended for practical inference speed. CPU inference can be slow, and memory requirements increase with audio length.

The dependency is installed directly from the requested Git repository. It follows the upstream default branch; record the installed commit with `uv pip freeze --python .venv` when you need reproducibility. The first inference downloads the model from Hugging Face and requires network access and adequate disk space. The official model integration uses `trust_remote_code=True`, executing model code from that source; use a trusted local model directory with `--model` if you need to review and freeze that code.

## Usage

Open the desktop file picker:

```bash
uv run --no-project --python .venv transcribe.py
```

Supply a path directly on headless or remote machines:

```bash
uv run --no-project --python .venv transcribe.py /path/to/meeting.mp3
uv run --no-project --python .venv transcribe.py /path/to/meeting.wav --device cpu --output-dir outputs
uv run --no-project --python .venv transcribe.py /path/to/meeting.flac --device cuda --max-new-tokens 65536
uv run --no-project --python .venv transcribe.py --help
```

Cancellation exits successfully without loading the model. File paths containing spaces must be quoted.

Each successful run creates a unique folder under `outputs/` containing:

- `raw_transcript.txt`: the original MOSS response.
- `transcript.txt`: readable timestamped text with speaker labels.
- `transcript.json`: source path and segments with `start`, `end`, `speaker`, and `text` fields. Times are in seconds.
- `S01.wav`, `S02.wav`, etc.: each speaker's audio intervals concatenated in chronological order. Silence between intervals is removed; sample rate, channels, and PCM/float precision are preserved. Intervals overlapping within one speaker are merged to avoid duplicate samples.

Diarization identifies who spoke when. It does not isolate voices during simultaneous speech: overlapping voices and background sounds remain in the exported intervals. Speaker labels are model-assigned identifiers, not names, and may contain errors. Large inputs may exhaust device memory. A generation token limit can truncate the transcript; inspect the raw output and increase `--max-new-tokens` when needed. No automatic chunking is applied, so speaker identities are inferred in one pass.

## Structure and validation

- [transcribe.py](transcribe.py): script entry point.
- [speech_to_text/cli.py](speech_to_text/cli.py): file picker and orchestration.
- [speech_to_text/backend.py](speech_to_text/backend.py): official MOSS inference and parser adapter.
- [speech_to_text/outputs.py](speech_to_text/outputs.py): transcript and speaker WAV export.
- [AGENTS.md](AGENTS.md): repository agent rules.

Run lightweight tests without downloading model weights:

```bash
uv run --no-project --python .venv -m unittest discover -s tests -v
```

A real inference smoke test requires the setup above and an audio file with known speakers. Check the resulting text and listen to the exported speaker files to assess the model's accuracy on your recording.

## Language hints

The research notebook defaults to `LANGUAGES = ["Thai", "English"]`. The inference adapter appends these hints to the official MOSS prompt, requesting transcription in the original languages without translation while preserving timestamps and speaker labels. This is a prompt hint, not a strict language restriction or an accuracy guarantee. Set `LANGUAGES = None` to use the default prompt. Each notebook experiment records its language hints.

```bash
uv run --no-project --python .venv transcribe.py /path/to/meeting.wav --languages Thai English
```

## Two-model research comparison

The research notebook pins MOSS to `cuda:0` and [Typhoon Whisper](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3) to `cuda:1`, requiring two available CUDA GPUs. Run all cells to transcribe the same active inputs with both models and export `comparison.json`, `comparison.html`, and `typhoon_transcript.txt` per recording, plus a batch comparison summary. The passes run sequentially with separate device assignments. Typhoon uses 30-second overlapping chunks with timestamps and batch size 1 to limit GPU memory; it provides ASR text without speaker diarization. Typhoon defaults to Thai, matching the model card; set `TYPHOON_LANGUAGE = None` for a separate automatic-detection experiment. MOSS retains Thai/English prompt hints. Workflow timings include model loading and other processing and are not controlled speed measurements. Human reference transcripts are needed to measure accuracy.

When GPU 1 has limited free memory, `TYPHOON_CPU_OFFLOAD = True` keeps the Typhoon encoder on CPU and its decoder on GPU 1. Set this to `False` to place the full model on GPU 1 when sufficient memory is free. Actual encoder placement is included in comparison metadata.

## CPU versus GPU benchmark

For MOSS on Intel iGPU, use [benchmark_iGPU.ipynb](benchmark_iGPU.ipynb).
It runs three fresh-process XPU BF16 trials per active recording by default,
with optional CPU FP32 comparison, fixed CPU threads, and offline cached weights.
First complete a successful [iGPU research run](research_iGPU.ipynb) to populate the cache.
Outputs under `outputs/benchmark-iGPU-*` include per-trial transcripts, JSON/CSV,
and an HTML summary of median workflow time, RTF, failures, and XPU allocator memory.
Timing includes model loading and cleanup, rather than generation alone.
Unavailable XPU devices and missing cached weights are recorded as failed trials.

Open [benchmark_cpu_gpu.ipynb](benchmark_cpu_gpu.ipynb) with the project `.venv` kernel. It tests every active recording with MOSS CPU/GPU 0 and Typhoon CPU/GPU 1 using independent processes, cached weights, and a fixed CPU thread count. It exports per-trial transcripts, JSON, CSV, and an HTML summary with median workflow times, real-time factors, and GPU speedup. Timing includes model loading and transcription; it is not inference-only latency. CPU FP32 and GPU mixed precision are reported explicitly. Increase `REPEATS` to measure multiple trials.

## Model cleanup

Both inference adapters use `try/finally` and the shared runtime cleanup helper to release model/pipeline references, collect garbage, and empty the allocator cache after every call, including failures and interruptions. Scripts and notebooks reuse these adapters. CUDA context and driver memory may remain until the kernel or process exits; zero cached model allocations does not imply zero total memory reported by `nvidia-smi`.

See [model usage analysis](docs/model_usage_analysis.md) for upstream settings, explicit project deviations, and decoder-length compatibility decisions. Historical reports using 15-second chunks are retained as historical results.

## Intel iGPU runtime on UBX-330m

The inspected host has an Intel Core Ultra 5 125H, Meteor Lake iGPU (PCI
`8086:7dd5`), Ubuntu 24.04, the `i915` kernel driver, and 16 GiB shared RAM.
The original installation had OpenCL but no Level Zero loader or Intel Level Zero
GPU driver, so an XPU PyTorch wheel alone reported zero devices.

Install the project-local compute runtime and Jupyter kernel without sudo:

```bash
uv run --no-project --python .venv setup_iGPU.py
```

The setup downloads pinned official Intel/Level Zero release packages, verifies
GitHub release SHA256 digests, and extracts them into `.tools/intel-iGPU`.
It leaves system drivers unchanged. These user-mode packages target Ubuntu 24.04
x86_64 and require the existing working Intel kernel driver and render-node access.
Select **MOSS Intel iGPU (.venv)** for both iGPU notebooks and restart the kernel.
The library path must be set before the Python process starts. Benchmark child
processes receive this path automatically. For direct CLI inference:

```bash
LD_LIBRARY_PATH="$PWD/.tools/intel-iGPU/usr/lib/x86_64-linux-gnu:$PWD/.tools/intel-iGPU/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  uv run --no-project --python .venv transcribe.py audio/standby/test-audio.wav --device xpu:0
```

The local stack uses compute-runtime 25.48.36300.8, IGC 2.24.8, and Level Zero
loader 1.34.0. This combination detects one Intel Graphics device with 112 EUs.
FP16/BF16/FP32 matrix multiplication and BF16 SDPA were verified on the GPU;
post-cleanup tensor allocation returned to zero. These kernel checks alone do
not establish MOSS inference correctness or benchmark performance.

References: [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html),
[Intel compute-runtime release](https://github.com/intel/compute-runtime/releases/tag/25.48.36300.8).

Real-model validation on this host succeeded using an 8-second, 16 kHz mono
excerpt of `audio/standby/test-audio.wav`. An isolated offline benchmark with
4 CPU threads, BF16, Thai/English hints, and a 256-token limit completed in
20.54 seconds (RTF 2.567). Peak PyTorch XPU allocation was 2,056,759,808 bytes
(about 1.92 GiB); allocated and reserved bytes both returned to zero. It produced
two parsed speech segments. This is one short-input workflow trial with cached
weights and compiled kernels, not a full-recording benchmark or an accuracy score.
The first online smoke run took 54.76 seconds, including first-use overhead.
