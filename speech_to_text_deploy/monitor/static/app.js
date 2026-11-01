"use strict";

const $ = (id) => document.getElementById(id);
const state = { run: null, paused: false, before: null, data: null, rendered: new Map(), runsKey: "" };
let mqttLiveOnly = false;
let mqttTranscriptSource = null;
const labels = { collecting: "Collecting", starting: "Starting", draining: "Draining", success: "Completed", success_with_warnings: "Completed with warnings", failed: "Failed", inactive: "Inactive", partial: "Partial", no_speech: "No speech", not_processed: "Not processed" };
const fmt = (n, digits = 2) => n == null ? "—" : Number(n).toFixed(digits);
const text = (id, value) => { $(id).textContent = value; };
const node = (tag, cls, value) => { const element = document.createElement(tag); if (cls) element.className = cls; if (value != null) element.textContent = value; return element; };
function stamp(seconds) { if (seconds == null) return "—"; const value = Math.max(0, Number(seconds)); return `${String(Math.floor(value / 60)).padStart(2, "0")}:${(value % 60).toFixed(2).padStart(5, "0")}`; }
function relative(iso) { const seconds = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000)); return seconds < 60 ? `${seconds}s ago` : `${Math.floor(seconds / 60)}m ago`; }
function badge(status) { const el = node("span", "badge", labels[status] || status); if (["partial", "success_with_warnings", "draining"].includes(status)) el.classList.add("warning"); else if (status === "failed") el.classList.add("error"); else if (!["success", "collecting"].includes(status)) el.classList.add("neutral"); return el; }
function notice(message) { $("notice").hidden = !message; text("notice", message || ""); }
async function fetchJSON(url) { const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(10000) }); if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); }

function utterance(entry, rejected = false, runId = "", segmentIndex = 0) {
  const row = node("article", `utterance${rejected ? " rejected" : ""}`);
  const displaySpeaker = entry.label_speaker || entry.speaker;
  const displayText = entry.label_text ?? entry.text;
  row.dataset.search = `${displaySpeaker} ${displayText}`.toLocaleLowerCase();
  const avatar = node("span", `speaker-avatar${Number((displaySpeaker || "").replace(/\D/g, "")) % 2 === 0 ? " speaker-alt" : ""}`, displaySpeaker || "?");
  const content = node("div", "utterance-content");
  const meta = node("div", "utterance-meta");
  meta.append(node("span", "speaker-name", displaySpeaker || "Unknown speaker"));
  if (rejected) meta.append(node("span", "", `Model time ${entry.model_start_seconds}–${entry.model_end_seconds}s · rejected`));
  else meta.append(node("span", "", `${stamp(entry.recording_start_seconds)} – ${stamp(entry.recording_end_seconds)}`));
  content.append(meta);
  if (rejected) content.append(node("div", "utterance-text", entry.text));
  else {
    const editor = node("div", "label-editor");
    const transcriptLabel = node("label", "editor-field");
    transcriptLabel.append(node("span", "editor-label", "Reviewed transcript"));
    const transcript = document.createElement("textarea");
    transcript.rows = Math.min(5, Math.max(2, displayText.split("\n").length)); transcript.maxLength = 10000;
    transcript.value = displayText; transcript.setAttribute("aria-label", `Reviewed transcript for ${entry.audio_file}`);
    transcriptLabel.append(transcript);
    const speakerLabel = node("label", "speaker-editor"); speakerLabel.append(node("span", "editor-label", "Speaker ID"));
    const speaker = document.createElement("input"); speaker.type = "text"; speaker.maxLength = 32;
    speaker.value = displaySpeaker || "S01"; speaker.setAttribute("aria-label", `Speaker ID for ${entry.audio_file}`);
    speakerLabel.append(speaker);
    const actions = node("div", "label-actions");
    const includeLabel = node("label", "dataset-check");
    const include = document.createElement("input"); include.type = "checkbox"; include.checked = Boolean(entry.selected);
    include.setAttribute("aria-label", `Include ${entry.audio_file} in the fine-tuning dataset`);
    includeLabel.append(include, node("span", "", "Include in dataset"));
    const save = node("button", "button secondary save-label", "Save label"); save.type = "button";
    save.addEventListener("click", () => saveLabel(save, runId, segmentIndex, entry.audio_file, transcript, speaker, include));
    actions.append(includeLabel, save);
    editor.append(transcriptLabel, speakerLabel, actions);
    content.append(editor);
  }
  if (rejected) content.append(node("p", "rejection-reason", `${entry.reason} No audio clip was created.`));
  else if (entry.audio_url) {
    const controls = node("div", "audio-row");
    const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none"; audio.src = entry.audio_url;
    audio.setAttribute("aria-label", `Play ${entry.speaker} at ${stamp(entry.recording_start_seconds)}`);
    audio.addEventListener("play", () => document.querySelectorAll("audio").forEach((other) => { if (other !== audio) other.pause(); }));
    controls.append(audio, node("span", "audio-note", `${fmt(entry.end_seconds - entry.start_seconds)}s clip · ${fmt(entry.start_seconds)}–${fmt(entry.end_seconds)}s in window`));
    content.append(controls);
  } else content.append(node("p", "rejection-reason", "Saved audio is currently unavailable."));
  row.append(avatar, content); return row;
}

async function saveLabel(button, runId, segmentIndex, audioFile, transcript, speaker, include) {
  button.disabled = true; text("dataset-status", "Saving reviewed label…");
  try {
    const path = `/api/runs/${encodeURIComponent(runId)}/labels/segment-${String(segmentIndex).padStart(6, "0")}/${encodeURIComponent(audioFile.replace(/\.wav$/i, ".json"))}`;
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: transcript.value, speaker: speaker.value, selected: include.checked }),
      signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    text("dataset-status", result.selected ? "Label saved and selected for export." : "Reviewed label saved.");
    button.textContent = "Saved"; setTimeout(() => { button.textContent = "Save label"; }, 1500);
  } catch (error) { text("dataset-status", `Could not save label (${error.message}).`); }
  finally { button.disabled = false; }
}

function windowCard(win, position, runId) {
  const card = node("details", "window"); card.open = position < 3;
  const heading = node("summary");
  heading.append(node("span", "window-title", `WINDOW ${String(win.index).padStart(3, "0")}`),
    node("span", "window-time", `${stamp(win.offset_seconds)} · ${fmt(win.audio_seconds, 1)}s`),
    node("span", "window-rtf", win.rtf == null ? "RTF —" : `RTF ${fmt(win.rtf, 3)}`));
  if (win.status !== "success") heading.append(badge(win.status));
  card.append(heading);
  win.utterances.forEach((entry) => card.append(utterance(entry, false, runId, win.index)));
  win.rejected.forEach((entry) => card.append(utterance(entry, true)));
  if (!win.utterances.length && !win.rejected.length) card.append(node("div", "window-empty", win.error || "No parsed speech in this window. Raw output is retained below."));
  if (win.raw_transcript) { const raw = node("details", "raw"); raw.append(node("summary", "", "Raw model output"), node("pre", "", win.raw_transcript)); card.append(raw); }
  return card;
}

function renderWindows(data) {
  const container = $("windows");
  if (!data.windows.length) { if (!state.rendered.size) container.replaceChildren(node("div", "empty", "Listening for the first completed audio window…")); return; }
  if (!state.rendered.size) container.replaceChildren();
  const keep = new Set(); let cursor = container.firstChild;
  data.windows.forEach((win, position) => {
    const key = `${data.id}:${win.index}`; const signature = JSON.stringify(win); keep.add(key);
    let existing = state.rendered.get(key);
    if (!existing || existing.signature !== signature) {
      const card = windowCard(win, position, data.id);
      if (existing) { card.open = existing.element.open; if (cursor === existing.element) cursor = card; existing.element.replaceWith(card); }
      existing = { signature, element: card }; state.rendered.set(key, existing);
    }
    if (existing.element !== cursor) container.insertBefore(existing.element, cursor);
    cursor = existing.element.nextSibling;
  });
  for (const [key, value] of state.rendered) { if (!keep.has(key)) { value.element.remove(); state.rendered.delete(key); } }
  applySearch();
}

function applySearch() {
  const query = $("search").value.trim().toLocaleLowerCase(); let matches = 0;
  document.querySelectorAll(".window").forEach((card) => {
    let visible = 0; const entries = card.querySelectorAll(".utterance");
    entries.forEach((entry) => { const match = !query || entry.dataset.search.includes(query); entry.hidden = !match; if (match) visible++; });
    card.hidden = Boolean(query && !visible); if (!card.hidden) matches++;
    if (query && visible) card.open = true;
  });
  $("no-matches").hidden = !query || matches > 0;
}

function renderChart(windows) {
  const values = [...windows].reverse().filter((win) => win.rtf != null);
  const target = $("chart"); target.replaceChildren();
  if (!values.length) { target.append(node("span", "quiet", "Waiting for timing data")); return; }
  const ns = "http://www.w3.org/2000/svg"; const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 280 110"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Recent window RTF trend");
  const max = Math.max(1.2, ...values.map((win) => win.rtf * 1.15));
  const x = (i) => 8 + i * 264 / Math.max(1, values.length - 1); const y = (value) => 96 - value / max * 86;
  const baseline = document.createElementNS(ns, "path"); baseline.setAttribute("d", `M0 ${y(1)}H280`); baseline.setAttribute("stroke", "#465365"); baseline.setAttribute("stroke-dasharray", "4 5"); svg.append(baseline);
  const path = document.createElementNS(ns, "path"); path.setAttribute("d", values.map((win, i) => `${i ? "L" : "M"}${x(i)} ${y(win.rtf)}`).join(" ")); path.setAttribute("fill", "none"); path.setAttribute("stroke", "#8ee9c1"); path.setAttribute("stroke-width", "2"); svg.append(path);
  values.forEach((win, i) => { const circle = document.createElementNS(ns, "circle"); circle.setAttribute("cx", x(i)); circle.setAttribute("cy", y(win.rtf)); circle.setAttribute("r", "3"); circle.setAttribute("fill", "#8ee9c1"); const title = document.createElementNS(ns, "title"); title.textContent = `Window ${win.index}: RTF ${fmt(win.rtf, 3)}`; circle.append(title); svg.append(circle); });
  target.append(svg);
}

function renderTimings(windows) {
  const target = $("timings"); target.replaceChildren();
  windows.filter((win) => win.latency_seconds != null).slice(0, 6).forEach((win) => {
    const row = node("div", "timing-row"), header = node("div", "timing-header");
    header.append(node("span", "", `Window ${String(win.index).padStart(3, "0")}`), node("span", "", `${fmt(win.latency_seconds)}s`));
    const track = node("div", "timing-track");
    for (const [kind, value] of [["infer", win.inference_seconds], ["queue", win.queue_wait_seconds], ["store", win.trim_and_store_seconds]]) { const span = node("span", `timing-${kind}`); span.style.width = `${Math.min(100, (value || 0) / Math.max(.00001, win.latency_seconds) * 100)}%`; span.title = `${kind}: ${fmt(value, 3)}s`; track.append(span); }
    const details = node("div", "timing-values"); details.append(node("span", "", `Inference ${fmt(win.inference_seconds)}s`), node("span", "", `Queue ${fmt(win.queue_wait_seconds)}s`));
    row.append(header, track, details); target.append(row);
  });
  if (!target.children.length) target.append(node("p", "quiet", "No completed windows yet."));
}

function render(data) {
  state.data = data; const stats = data.stats;
  text("metric-audio", stamp(stats.audio_seconds)); text("metric-windows", `${stats.processed_windows} processed / ${stats.reported_windows} reported windows`);
  text("metric-clips", stats.clip_count.toLocaleString()); text("metric-rejected", `${stats.rejected_count} rejected intervals · ${stats.partial_windows} partial windows`);
  text("metric-pending", stats.pending_windows); text("metric-captured", `${stats.captured_windows} captured windows`);
  text("metric-rtf", fmt(stats.weighted_rtf, 3)); text("source", data.source); text("device", data.device);
  text("window-size", `${data.window_seconds} seconds`); text("current-window", data.current_window == null ? "—" : `Window ${data.current_window}`);
  text("last-activity", relative(data.last_activity_at)); text("updated", `Updated ${new Date(data.updated_at).toLocaleTimeString()} · File activity is inferred from saved output`);
  const status = badge(data.status); status.id = "status"; $("status").replaceWith(status);
  text("visible-count", `${data.windows.length} windows`);
  renderWindows(data); renderChart(data.windows); renderTimings(data.windows);
  $("older").disabled = !data.has_older; $("live").hidden = state.before == null;
  notice(data.errors.length ? data.errors.map((error) => `${error.worker}: ${error.error}`).join(" · ") : (state.before != null ? "History view. Use Back to live to resume automatic updates." : ""));
}

const liveAudio = { source: null, context: null, gain: null, nextTime: 0 };
function resetLiveAudioLevel() {
  const meter = $("live-audio-level-bar").parentElement;
  $("live-audio-level-bar").style.width = "0%";
  $("live-audio-level-value").textContent = "—";
  meter.setAttribute("aria-valuenow", "0");
  meter.setAttribute("aria-valuetext", "No live audio");
}
function updateLiveAudioLevel(rms, peak) {
  const floorDb = -60;
  const rmsDb = 20 * Math.log10(Math.max(rms, 1 / 32768));
  const peakDb = 20 * Math.log10(Math.max(peak, 1 / 32768));
  const level = Math.max(0, Math.min(100, ((rmsDb - floorDb) / -floorDb) * 100));
  const meter = $("live-audio-level-bar").parentElement;
  $("live-audio-level-bar").style.width = `${level}%`;
  $("live-audio-level-value").textContent = `RMS ${rmsDb.toFixed(1)} dBFS · Peak ${peakDb.toFixed(1)} dBFS`;
  meter.setAttribute("aria-valuenow", level.toFixed(0));
  meter.setAttribute("aria-valuetext", `RMS ${rmsDb.toFixed(1)} dBFS, peak ${peakDb.toFixed(1)} dBFS`);
}
function stopLiveAudio() {
  if (liveAudio.source) liveAudio.source.close();
  if (liveAudio.context) liveAudio.context.close();
  liveAudio.source = liveAudio.context = liveAudio.gain = null;
  liveAudio.nextTime = 0;
  $("live-audio-toggle").textContent = "Listen live";
  text("live-audio-status", "Not connected");
  resetLiveAudioLevel();
}
function playLiveChunk(packet) {
  if (!liveAudio.context || !liveAudio.gain) return;
  const binary = atob(packet.audio);
  const samples = new DataView(new ArrayBuffer(binary.length));
  for (let index = 0; index < binary.length; index++) samples.setUint8(index, binary.charCodeAt(index));
  const sampleCount = Math.floor(binary.length / 2);
  const buffer = liveAudio.context.createBuffer(1, sampleCount, packet.sample_rate);
  const channel = buffer.getChannelData(0);
  let sumSquares = 0;
  let peak = 0;
  for (let index = 0; index < sampleCount; index++) {
    const sample = samples.getInt16(index * 2, true) / 32768;
    channel[index] = sample;
    sumSquares += sample * sample;
    peak = Math.max(peak, Math.abs(sample));
  }
  updateLiveAudioLevel(sampleCount ? Math.sqrt(sumSquares / sampleCount) : 0, peak);
  const source = liveAudio.context.createBufferSource(); source.buffer = buffer; source.connect(liveAudio.gain);
  const startAt = Math.max(liveAudio.nextTime, liveAudio.context.currentTime + 0.03);
  source.start(startAt); liveAudio.nextTime = startAt + buffer.duration;
}
$("live-audio-toggle").addEventListener("click", async () => {
  if (liveAudio.source) { stopLiveAudio(); return; }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass || !window.EventSource) { text("live-audio-status", "Live audio is not supported in this browser."); return; }
  try {
    liveAudio.context = new AudioContextClass(); await liveAudio.context.resume();
    liveAudio.gain = liveAudio.context.createGain();
    liveAudio.gain.gain.value = Number($("live-audio-volume").value); liveAudio.gain.connect(liveAudio.context.destination);
    liveAudio.source = new EventSource("/api/live-audio/stream");
    $("live-audio-toggle").textContent = "Stop listening";
    text("live-audio-status", "Connecting to live capture…");
    liveAudio.source.onopen = () => text("live-audio-status", "Connected · waiting for audio");
    liveAudio.source.onmessage = (event) => {
      try { playLiveChunk(JSON.parse(event.data)); text("live-audio-status", "Playing live capture"); }
      catch { text("live-audio-status", "Received an invalid audio chunk."); }
    };
    liveAudio.source.onerror = () => { text("live-audio-status", "Reconnecting to live capture…"); resetLiveAudioLevel(); };
  } catch (error) { stopLiveAudio(); text("live-audio-status", `Could not start live audio (${error.message}).`); }
});
$("live-audio-volume").addEventListener("input", () => {
  if (liveAudio.gain) liveAudio.gain.gain.value = Number($("live-audio-volume").value);
});

$("export-dataset").addEventListener("click", async () => {
  const button = $("export-dataset"); button.disabled = true;
  text("dataset-status", "Appending new selected samples; existing dataset rows will be kept…");
  try {
    const response = await fetch("/api/dataset/export", { method: "POST", signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    text("dataset-status", `${result.added} new samples appended; ${result.skipped_existing} already-exported samples skipped (${result.existing_count + result.added} total).`);
  } catch (error) { text("dataset-status", `Dataset export failed (${error.message}).`); }
  finally { button.disabled = false; }
});

function changeRun(id) { state.run = id; state.before = null; state.rendered.clear(); $("windows").replaceChildren(); }
function renderMqttTranscript(message) {
  const target = $("mqtt-live-events");
  if (target.firstElementChild?.classList.contains("empty")) target.replaceChildren();
  const event = node("article", "mqtt-live-event");
  const start = message.start_time ? new Date(message.start_time).toLocaleTimeString() : "Unknown start";
  const end = message.end_time ? new Date(message.end_time).toLocaleTimeString() : "Unknown end";
  event.append(node("time", "mqtt-live-time", `${start} – ${end}`));
  event.append(node("p", "mqtt-live-text", message.transcript));
  target.append(event);
  while (target.children.length > 100) target.firstElementChild.remove();
  text("mqtt-live-status", "Receiving MQTT transcripts");
}
function connectLiveMqtt() {
  mqttTranscriptSource = new EventSource("/api/live-transcript/stream");
  mqttTranscriptSource.onopen = () => text("mqtt-live-status", "Connected · waiting for MQTT transcripts");
  mqttTranscriptSource.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (typeof message.transcript !== "string") throw new Error("Missing transcript");
      renderMqttTranscript(message);
    } catch { text("mqtt-live-status", "Received an invalid MQTT transcript."); }
  };
  mqttTranscriptSource.onerror = () => text("mqtt-live-status", "Reconnecting to MQTT stream…");
}
async function initializeMonitor() {
  try {
    const settings = await fetchJSON("/api/settings");
    if (!settings.save_audio) $("live-audio-panel").hidden = true;
    if (!settings.save_transcript) {
      mqttLiveOnly = true;
      $("run-toolbar").hidden = true;
      $("metrics").hidden = true;
      $("workspace").hidden = true;
      $("pause").hidden = true;
      $("mqtt-live-panel").hidden = false;
      $("connection").className = "connection connected";
      $("connection").replaceChildren(node("i"), document.createTextNode("Connected"));
      if (settings.mqtt_enabled) connectLiveMqtt();
      else text("mqtt-live-status", "MQTT is disabled in settings.");
      return;
    }
  } catch (error) {
    notice(`Could not load monitor settings (${error.message}).`);
  }
  refresh();
}
async function refresh() {
  if (mqttLiveOnly) return;
  try {
    const { runs } = await fetchJSON("/api/runs");
    const key = runs.map((run) => run.id).join("|");
    if (key !== state.runsKey) { state.runsKey = key; $("run").replaceChildren(); runs.forEach((run) => $("run").add(new Option(run.id, run.id))); if (!runs.length) $("run").add(new Option("Waiting for a run…", "")); }
    if (runs.length && (!state.run || ($("follow").checked && !state.paused && state.before == null))) { if (state.run !== runs[0].id) changeRun(runs[0].id); }
    if (state.run) $("run").value = state.run;
    if (state.run && !state.paused && state.before == null) {
      const requestedRun = state.run;
      const data = await fetchJSON(`/api/runs/${encodeURIComponent(requestedRun)}`);
      if (state.run === requestedRun && !state.paused && state.before == null) render(data);
    }
    $("connection").className = "connection connected"; $("connection").replaceChildren(node("i"), document.createTextNode(state.paused ? "Updates paused" : "Connected"));
  } catch (error) { $("connection").className = "connection offline"; $("connection").replaceChildren(node("i"), document.createTextNode("Reconnecting")); notice("Could not refresh output. Your current view is preserved; retrying automatically."); }
  setTimeout(refresh, 1000);
}

$("run").addEventListener("change", async () => { $("follow").checked = false; changeRun($("run").value); if (state.run) { try { render(await fetchJSON(`/api/runs/${encodeURIComponent(state.run)}`)); } catch { notice("This run is unavailable. Select another run."); } } });
$("pause").addEventListener("click", () => { state.paused = !state.paused; $("pause").setAttribute("aria-pressed", String(state.paused)); text("pause", state.paused ? "Resume updates" : "Pause updates"); });
$("search").addEventListener("input", applySearch);
$("older").addEventListener("click", async () => { if (!state.data?.windows.length) return; const before = Math.min(...state.data.windows.map((win) => win.index)); try { const data = await fetchJSON(`/api/runs/${encodeURIComponent(state.run)}?before=${before}`); state.before = before; render(data); } catch { notice("Could not load older windows. Please retry."); } });
$("live").addEventListener("click", () => { state.before = null; state.paused = false; $("pause").setAttribute("aria-pressed", "false"); text("pause", "Pause updates"); notice(""); });
document.addEventListener("keydown", (event) => { if (event.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) { event.preventDefault(); $("search").focus(); } });
initializeMonitor();
