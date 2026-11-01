# Model Usage Analysis

Reviewed upstream documentation on 2026-11-01 before revising the inference configuration.

## Sources

- [Typhoon Whisper Large v3 model card](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3)
- [MOSS-Transcribe-Diarize README](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- Installed Transformers Whisper generation and ASR pipeline implementations, inspected to verify decoder length validation and overlap behavior.

## Typhoon: documented example and implementation decision

The model card demonstrates the full Whisper Large v3 ASR model with `AutoModelForSpeechSeq2Seq`, `AutoProcessor`, and the ASR pipeline. Its example uses 30-second chunks, timestamps, Thai language, BF16 on CUDA, FP32 on CPU, a batch size of 16, and a new-token setting of 448. It is a Thai-focused model; English and mixed-language quality require evaluation. It does not supply speaker identities.

Use 30-second chunks and timestamp output as demonstrated. The installed pipeline defaults to a five-second stride on each side for this chunk size; make that overlap explicit. These are overlapping windows, not disjoint 30-second cuts. Let the pipeline merge its results. Keep batch size 1 as an explicit memory-related deviation from the example, not an architectural requirement. Preserve greedy decoding and deterministic sampling settings.

Use Thai as the default language to match the example; allow an explicit `None` for a separate automatic-detection experiment. A Whisper language token accepts one language, not a Thai/English list. Keep the requested `typhoon-ai` repository ID; the card's code still names the older `scb10x` namespace.

The installed Whisper decoder checks the sum of prompt length and new-token budget against its positional limit. Copying `max_new_tokens=448` literally can exceed that limit after special prompt tokens. Therefore use a single total-length setting, `max_length=448`, with `max_new_tokens=None`. This is a version-compatibility decision rather than a claim that the model card uses that exact configuration. Do not suppress warnings. The pipeline's experimental chunking warning can remain because the upstream example itself uses that API. Native long-form `generate()` is a distinct optional experiment, not the workflow selected here.

## MOSS: documented example and implementation decision

MOSS uses `AutoModelForCausalLM` and its official processor, message builder, generation helper, and transcript parser. Preserve the timestamped speaker format and the default prompt. Its audio frontend handles 16 kHz audio in 30-second feature chunks internally. This does not require independent application-level ASR calls per 30 seconds. Keep each recording in one MOSS request so the model can infer speaker labels across that recording.

Follow upstream BF16 on CUDA, FP32 on CPU, SDPA attention, and deterministic generation. The larger project token budget is configurable for long transcripts. Thai/English prompt hints are a project experiment appended to the official prompt, not a strict language filter. Speaker WAV export clips diarized intervals; it does not separate simultaneous sources.

## Comparison protocol

Use the same original recordings from `audio/active/`, keep standby excluded, record actual per-model configuration, and preserve independent transcripts and timestamps. MOSS provides speaker IDs while Typhoon does not. Workflow timings include loading and cleanup and are not accuracy metrics. Earlier results using 15-second Typhoon windows or FP16 represent the earlier protocol and must not be relabeled as results from this configuration. Rerun to obtain new measurements. Both adapters must use the existing shared cleanup helper in `finally`.

## Validation with the active recording

Executed every cell of the research notebook using the real models in the project `.venv`. MOSS on GPU 0 completed in 29.33 seconds; Typhoon on GPU 1 completed in 11.99 seconds. The output metadata confirms 30-second windows, five-second stride, BF16, Thai language, timestamps, and a single 448-token total decoder limit. The repeated conflicting-length warning is absent. The pipeline still reports its experimental chunking warning, and the model left at least one ending timestamp unspecified; missing timestamps are preserved as null rather than invented. These execution checks do not establish transcription accuracy. The CPU/GPU benchmark notebook was updated for the new protocol and its historical cell outputs cleared; it has not been rerun under this protocol.
