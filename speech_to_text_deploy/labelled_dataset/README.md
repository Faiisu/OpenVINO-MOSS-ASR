# Reviewed MOSS fine-tuning dataset

Use the live monitor to edit an utterance's transcript and speaker ID, select
**Include in dataset**, and save the label. Choose **Export selected dataset**
to create `dataset.jsonl` and copy the selected WAV clips into `audio/`.

Each JSONL record uses the official MOSS conversation format: one user text
prompt, one user audio path, then one assistant transcript. For example:

```json
{"conversation":[{"role":"user","message_type":"text","content":"Transcribe the audio with timestamps and speaker labels."},{"role":"user","message_type":"audio","content":"audio/example.wav"},{"role":"assistant","message_type":"text","content":"[0.00][S01]Hello[0.72]"}],"source":{"run":"run-example","segment":"segment-000001","utterance":"utterance-0001.wav"}}
```

The WAV already contains only the selected utterance, so each target begins at
`[0.00]` and ends at the duration of that clip. Speaker IDs are local to the
original audio window. The audio path is relative to `dataset.jsonl`. The
`source` object preserves provenance and is ignored by the official trainer,
which reads the `conversation` field.

This export contains one short cropped utterance per sample. It is suitable for
the official input parser and can teach reviewed wording, but it does not retain
multi-speaker context from the original 30-second window. For robust timestamp
and diarization fine-tuning, build examples from full windows with all speaker
segments in one target, and collect substantially more reviewed audio. This
merged export currently has eleven samples.

Export is repeatable and append-only. It adds newly selected utterances to
`dataset.jsonl`, skips a source utterance or byte-identical WAV already present,
and never changes existing JSONL rows. Existing WAVs are reused; only new
selected audio is added. The original collection WAVs and transcripts are never
modified. Generated dataset audio and manifests are ignored by Git.
