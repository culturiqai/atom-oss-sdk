const state = {
  selectedJobId: null,
  jobs: [],
  selectedSupersonicJobId: null,
  supersonicJobs: [],
  supersonicTelemetry: null,
  runtimeTelemetry: null,
  geometryAssets: [],
  selectedGeometryId: "",
  intentFocus: "scientific_discovery",
  diagnosticMode: "overview",
  assistantEngine: "deterministic",
  uiSettings: {
    refreshMs: 2500,
    density: "comfortable",
    showRawPanels: false,
    motion: true,
    showObstacleOverlay: true,
    showSaliencyFocus: true,
    narrativeMode: "executive",
  },
  onboardingSeen: false,
  wizardAssistantCompleted: false,
  pollingTimer: null,
  studioCatalog: null,
  studioEvents: [],
  inverseSpecTemplate: null,
  worldKwDraftByWorld: {},
};

const worldSelect = document.getElementById("world-select");
const worldKwHint = document.getElementById("world-kwargs-contract-hint");
const worldKwFields = document.getElementById("world-kwargs-fields");
const jobList = document.getElementById("job-list");
const jobReport = document.getElementById("job-report");
const form = document.getElementById("job-form");
const submitBtn = document.getElementById("submit-btn");
const geometryUploadForm = document.getElementById("geometry-upload-form");
const geometryFileInput = document.getElementById("geometry-file");
const geometryUploadBtn = document.getElementById("geometry-upload-btn");
const geometryList = document.getElementById("geometry-list");
const geometrySelect = document.getElementById("geometry-select");
const statusMatrix = document.getElementById("status-matrix");
const selectedJobLabel = document.getElementById("selected-job-label");
const supersonicForm = document.getElementById("supersonic-form");
const supersonicSubmitBtn = document.getElementById("supersonic-submit-btn");
const supersonicJobList = document.getElementById("supersonic-job-list");
const supersonicTelemetry = document.getElementById("supersonic-telemetry");
const supersonicPauseBtn = document.getElementById("supersonic-pause-btn");
const supersonicResumeBtn = document.getElementById("supersonic-resume-btn");
const supersonicCancelBtn = document.getElementById("supersonic-cancel-btn");
const supersonicBookmarkBtn = document.getElementById("supersonic-bookmark-btn");
const supersonicBookmarkNote = document.getElementById("supersonic-bookmark-note");
const supersonicTimeline = document.getElementById("supersonic-timeline");
const supersonicIncidents = document.getElementById("supersonic-incidents");
const assistantForm = document.getElementById("assistant-form");
const assistantSubmitBtn = document.getElementById("assistant-submit-btn");
const assistantResponse = document.getElementById("assistant-response");
const refreshStudioBtn = document.getElementById("refresh-studio-btn");
const buildDirectorPackBtn = document.getElementById("build-director-pack-btn");
const studioSimulators = document.getElementById("studio-simulators");
const studioDemos = document.getElementById("studio-demos");
const studioLog = document.getElementById("studio-log");
const rewardChartCanvas = document.getElementById("chart-reward");
const shockChartCanvas = document.getElementById("chart-shock");
const saliencyChartCanvas = document.getElementById("chart-saliency");
const theoryImportanceCanvas = document.getElementById("chart-theory-importance");
const liveSpeedCanvas = document.getElementById("map-live-speed");
const liveDensityCanvas = document.getElementById("map-live-density");
const liveDivergenceCanvas = document.getElementById("map-live-divergence");
const attentionOverlayCanvas = document.getElementById("map-attention-overlay");
const eyesSaliencyCanvas = document.getElementById("map-eyes2-saliency");
const brainSaliencyCanvas = document.getElementById("map-brain-saliency");
const attentionOverlaySummary = document.getElementById("attention-overlay-summary");
const runtimeTelemetry = document.getElementById("runtime-telemetry");
const runtimeDiagnosticsStatus = document.getElementById("runtime-diagnostics-status");
const intentFocusSelect = document.getElementById("intent-focus-select");
const diagnosticModeSelect = document.getElementById("diagnostic-mode-select");
const assistantIntentSelect = document.getElementById("assistant-intent");
const assistantDiagnosticModeSelect = document.getElementById("assistant-diagnostic-mode");
const assistantEngineSelect = document.getElementById("assistant-engine");
const diagnosticCards = document.querySelectorAll(".diagnostic-card");
const kpiInverseScore = document.getElementById("kpi-inverse-score");
const kpiInverseFeasible = document.getElementById("kpi-inverse-feasible");
const kpiReward = document.getElementById("kpi-reward");
const kpiShock = document.getElementById("kpi-shock");
const kpiReduction = document.getElementById("kpi-reduction");
const kpiShockTrend = document.getElementById("kpi-shock-trend");
const refreshIntervalSelect = document.getElementById("refresh-interval-select");
const densitySelect = document.getElementById("ui-density-select");
const showRawPanelsToggle = document.getElementById("toggle-raw-panels");
const motionToggle = document.getElementById("toggle-motion");
const obstacleOverlayToggle = document.getElementById("toggle-obstacle-overlay");
const saliencyFocusToggle = document.getElementById("toggle-saliency-focus");
const showOnboardingBtn = document.getElementById("show-onboarding-btn");
const onboardingPanel = document.getElementById("onboarding-panel");
const onboardingStartBtn = document.getElementById("onboarding-start-btn");
const onboardingDismissBtn = document.getElementById("onboarding-dismiss-btn");
const wizardLoadInverseBtn = document.getElementById("wizard-load-inverse-btn");
const wizardLaunchInverseBtn = document.getElementById("wizard-launch-inverse-btn");
const wizardLaunchSupersonicBtn = document.getElementById("wizard-launch-supersonic-btn");
const wizardRunCopilotBtn = document.getElementById("wizard-run-copilot-btn");
const wizardStepCatalog = document.getElementById("wizard-step-catalog");
const wizardStepWorld = document.getElementById("wizard-step-world");
const wizardStepInverse = document.getElementById("wizard-step-inverse");
const wizardStepSupersonic = document.getElementById("wizard-step-supersonic");
const wizardStepAssistant = document.getElementById("wizard-step-assistant");
const narrativeModeSelect = document.getElementById("narrative-mode-select");
const narrativeFeed = document.getElementById("narrative-feed");
const missionIntentButtons = Array.from(
  document.querySelectorAll(".mission-mode[data-intent]")
);
const ALLOWED_STATUS = new Set(["queued", "running", "succeeded", "failed", "cancelled"]);
const RAW_SURFACE_IDS = ["studio-log", "supersonic-telemetry", "runtime-telemetry", "assistant-response", "job-report"];

function normalizeStatus(value) {
  return ALLOWED_STATUS.has(value) ? value : "queued";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function formatIsoTime(iso) {
  if (!iso) {
    return "time: n/a";
  }
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) {
    return `time: ${iso}`;
  }
  return `time: ${dt.toLocaleTimeString()} ${dt.toLocaleDateString()}`;
}

function formatNum(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return "n/a";
  }
  return n.toFixed(digits);
}

function parseGridShape(raw) {
  const values = String(raw || "")
    .split(",")
    .map((v) => Number.parseInt(v.trim(), 10))
    .filter((v) => Number.isFinite(v) && v > 0);
  if (values.length !== 3) {
    throw new Error("Grid shape must be a comma-separated triplet, e.g. 32,32,16");
  }
  return values;
}

function normalizeWorldSpec(raw) {
  return String(raw || "").trim().toLowerCase();
}

function asObject(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return {};
  }
  return raw;
}

function worldKwContracts() {
  return asObject(state.inverseSpecTemplate?.world_kwargs_contracts);
}

function worldKwProfileEntry(worldSpec) {
  const contracts = worldKwContracts();
  const key = normalizeWorldSpec(worldSpec);
  const profileMap = asObject(contracts.world_spec_profile_map);
  const profileName = String(profileMap[key] || "");
  const profiles = asObject(contracts.profiles);
  const profile = asObject(profiles[profileName]);
  const fields = asObject(profile.fields);
  if (!profileName || !Object.keys(fields).length) {
    return null;
  }
  return { key, profileName, profile, fields, contracts };
}

function worldKwFieldLabel(fieldName) {
  const parts = String(fieldName || "")
    .split("_")
    .filter(Boolean);
  if (!parts.length) {
    return "Field";
  }
  return parts
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function worldKwFieldMeta(fieldSpec) {
  const spec = asObject(fieldSpec);
  const chunks = [String(spec.type || "string")];
  if (Number.isFinite(Number(spec.min))) {
    chunks.push(`min ${Number(spec.min)}`);
  }
  if (Number.isFinite(Number(spec.exclusive_min))) {
    chunks.push(`>${Number(spec.exclusive_min)}`);
  }
  if (Number.isFinite(Number(spec.max))) {
    chunks.push(`max ${Number(spec.max)}`);
  }
  if (Number.isFinite(Number(spec.exclusive_max))) {
    chunks.push(`<${Number(spec.exclusive_max)}`);
  }
  if (spec.suffix) {
    chunks.push(`suffix ${String(spec.suffix)}`);
  }
  if (spec.must_exist) {
    chunks.push("must_exist");
  }
  return chunks.join(" | ");
}

function worldKwInputType(fieldSpec) {
  const type = String(fieldSpec?.type || "string");
  if (type === "int" || type === "float") {
    return "number";
  }
  return "text";
}

function worldKwDraftForWorld(worldSpec) {
  return asObject(state.worldKwDraftByWorld[normalizeWorldSpec(worldSpec)]);
}

function setWorldKwDraft(worldSpec, values) {
  const key = normalizeWorldSpec(worldSpec);
  const draft = {};
  Object.entries(asObject(values)).forEach(([name, value]) => {
    if (value === null || value === undefined) {
      return;
    }
    draft[String(name)] = String(value);
  });
  state.worldKwDraftByWorld[key] = draft;
}

function worldKwExampleForWorld(worldSpec) {
  const contracts = worldKwContracts();
  const examples = asObject(contracts.examples);
  return asObject(examples[normalizeWorldSpec(worldSpec)]);
}

function renderWorldKwEditor(worldSpec) {
  if (!worldKwFields || !worldKwHint) {
    return;
  }
  const entry = worldKwProfileEntry(worldSpec);
  if (!entry) {
    worldKwHint.textContent = "No world-parameter contract available for this simulator.";
    worldKwFields.innerHTML = "<div class='world-kwargs-empty'>No editable world parameters.</div>";
    return;
  }

  const draft = worldKwDraftForWorld(entry.key);
  const example = worldKwExampleForWorld(entry.key);
  const fieldNames = Object.keys(entry.fields);
  const geometrySelected = Boolean(state.selectedGeometryId);
  const isCustomLbm = entry.key === "lbm:custom";
  const strictUnknown = Boolean(entry.contracts.validation_policy?.strict_unknown_keys);
  worldKwHint.textContent =
    `Profile: ${entry.profileName} | fields: ${fieldNames.length} | unknown keys: ${strictUnknown ? "rejected" : "allowed"}`;

  worldKwFields.innerHTML = "";
  fieldNames.forEach((fieldName) => {
    const spec = asObject(entry.fields[fieldName]);
    const fieldNode = document.createElement("div");
    fieldNode.className = "world-kwargs-field";

    const label = document.createElement("label");
    label.setAttribute("for", `world-kw-${fieldName}`);
    label.textContent = worldKwFieldLabel(fieldName);

    const meta = document.createElement("div");
    meta.className = "world-kwargs-field-meta";
    const left = document.createElement("span");
    left.textContent = worldKwFieldMeta(spec);
    meta.appendChild(left);
    const right = document.createElement("span");
    const requiredByContract = Boolean(spec.required);
    const requiredInUi = requiredByContract && !(isCustomLbm && fieldName === "stl_path" && geometrySelected);
    right.className = "world-kwargs-required";
    right.textContent = requiredInUi ? "required" : "optional";
    meta.appendChild(right);

    const input = document.createElement("input");
    input.id = `world-kw-${fieldName}`;
    input.type = worldKwInputType(spec);
    input.dataset.worldKwName = fieldName;
    input.dataset.worldKwType = String(spec.type || "string");
    input.dataset.required = requiredInUi ? "true" : "false";
    if (Number.isFinite(Number(spec.min))) {
      input.dataset.min = String(spec.min);
    }
    if (Number.isFinite(Number(spec.exclusive_min))) {
      input.dataset.exclusiveMin = String(spec.exclusive_min);
    }
    if (Number.isFinite(Number(spec.max))) {
      input.dataset.max = String(spec.max);
    }
    if (Number.isFinite(Number(spec.exclusive_max))) {
      input.dataset.exclusiveMax = String(spec.exclusive_max);
    }
    if (spec.suffix) {
      input.dataset.suffix = String(spec.suffix);
    }
    if (input.type === "number") {
      input.step = String(spec.type) === "int" ? "1" : "any";
    }

    const hasDraft = Object.prototype.hasOwnProperty.call(draft, fieldName);
    const initial = hasDraft ? draft[fieldName] : example[fieldName];
    if (initial !== undefined && initial !== null) {
      input.value = String(initial);
    }
    if (isCustomLbm && fieldName === "stl_path" && geometrySelected) {
      input.placeholder = "auto from selected geometry asset";
      input.disabled = true;
      input.value = "";
    }

    input.addEventListener("input", () => {
      const worldKey = normalizeWorldSpec(worldSelect?.value || entry.key);
      const current = { ...worldKwDraftForWorld(worldKey) };
      const value = String(input.value || "");
      if (!value.trim()) {
        delete current[fieldName];
      } else {
        current[fieldName] = value;
      }
      state.worldKwDraftByWorld[worldKey] = current;
    });

    fieldNode.appendChild(label);
    fieldNode.appendChild(input);
    fieldNode.appendChild(meta);
    worldKwFields.appendChild(fieldNode);
  });
}

function parseWorldKwValue(inputNode, geometrySelected) {
  const name = String(inputNode.dataset.worldKwName || "");
  const raw = String(inputNode.value || "").trim();
  const required = String(inputNode.dataset.required || "false") === "true";
  if (!raw) {
    if (required) {
      throw new Error(`World parameter '${name}' is required`);
    }
    return { include: false };
  }

  const type = String(inputNode.dataset.worldKwType || "string");
  if (type === "int") {
    const num = Number(raw);
    if (!Number.isFinite(num) || !Number.isInteger(num)) {
      throw new Error(`World parameter '${name}' must be an integer`);
    }
    if (Number.isFinite(Number(inputNode.dataset.min)) && num < Number(inputNode.dataset.min)) {
      throw new Error(`World parameter '${name}' must be >= ${inputNode.dataset.min}`);
    }
    if (
      Number.isFinite(Number(inputNode.dataset.exclusiveMin)) &&
      num <= Number(inputNode.dataset.exclusiveMin)
    ) {
      throw new Error(`World parameter '${name}' must be > ${inputNode.dataset.exclusiveMin}`);
    }
    if (Number.isFinite(Number(inputNode.dataset.max)) && num > Number(inputNode.dataset.max)) {
      throw new Error(`World parameter '${name}' must be <= ${inputNode.dataset.max}`);
    }
    if (
      Number.isFinite(Number(inputNode.dataset.exclusiveMax)) &&
      num >= Number(inputNode.dataset.exclusiveMax)
    ) {
      throw new Error(`World parameter '${name}' must be < ${inputNode.dataset.exclusiveMax}`);
    }
    return { include: true, value: num };
  }

  if (type === "float") {
    const num = Number(raw);
    if (!Number.isFinite(num)) {
      throw new Error(`World parameter '${name}' must be numeric`);
    }
    if (Number.isFinite(Number(inputNode.dataset.min)) && num < Number(inputNode.dataset.min)) {
      throw new Error(`World parameter '${name}' must be >= ${inputNode.dataset.min}`);
    }
    if (
      Number.isFinite(Number(inputNode.dataset.exclusiveMin)) &&
      num <= Number(inputNode.dataset.exclusiveMin)
    ) {
      throw new Error(`World parameter '${name}' must be > ${inputNode.dataset.exclusiveMin}`);
    }
    if (Number.isFinite(Number(inputNode.dataset.max)) && num > Number(inputNode.dataset.max)) {
      throw new Error(`World parameter '${name}' must be <= ${inputNode.dataset.max}`);
    }
    if (
      Number.isFinite(Number(inputNode.dataset.exclusiveMax)) &&
      num >= Number(inputNode.dataset.exclusiveMax)
    ) {
      throw new Error(`World parameter '${name}' must be < ${inputNode.dataset.exclusiveMax}`);
    }
    return { include: true, value: num };
  }

  if (type === "path" && inputNode.dataset.suffix) {
    const suffix = String(inputNode.dataset.suffix).toLowerCase();
    if (raw && !raw.toLowerCase().endsWith(suffix)) {
      throw new Error(`World parameter '${name}' must end with '${suffix}'`);
    }
  }
  if (
    normalizeWorldSpec(worldSelect?.value || "") === "lbm:custom" &&
    name === "stl_path" &&
    geometrySelected
  ) {
    return { include: false };
  }
  return { include: true, value: raw };
}

function collectWorldKwPayload(geometrySelected) {
  if (!worldKwFields) {
    return {};
  }
  const payload = {};
  const inputs = Array.from(worldKwFields.querySelectorAll("input[data-world-kw-name]"));
  inputs.forEach((inputNode) => {
    const result = parseWorldKwValue(inputNode, geometrySelected);
    if (result.include) {
      payload[String(inputNode.dataset.worldKwName)] = result.value;
    }
  });
  const worldKey = normalizeWorldSpec(worldSelect?.value || "");
  setWorldKwDraft(worldKey, payload);
  return payload;
}

function loadUiSettings() {
  try {
    const raw = localStorage.getItem("atom_ui_settings_v1");
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return;
    }
    const refreshMs = Number(parsed.refreshMs);
    const density = String(parsed.density || "comfortable");
    const showRawPanels = Boolean(parsed.showRawPanels);
    const motion = parsed.motion !== false;
    const showObstacleOverlay = parsed.showObstacleOverlay !== false;
    const showSaliencyFocus = parsed.showSaliencyFocus !== false;
    const narrativeMode = String(parsed.narrativeMode || "executive");
    if (Number.isFinite(refreshMs) && refreshMs >= 800 && refreshMs <= 30000) {
      state.uiSettings.refreshMs = refreshMs;
    }
    if (density === "comfortable" || density === "compact") {
      state.uiSettings.density = density;
    }
    state.uiSettings.showRawPanels = showRawPanels;
    state.uiSettings.motion = motion;
    state.uiSettings.showObstacleOverlay = showObstacleOverlay;
    state.uiSettings.showSaliencyFocus = showSaliencyFocus;
    if (["off", "executive", "technical"].includes(narrativeMode)) {
      state.uiSettings.narrativeMode = narrativeMode;
    }
  } catch (err) {
    // no-op: invalid local state should not block UI startup
  }
}

function loadOnboardingState() {
  try {
    const raw = localStorage.getItem("atom_ui_onboarding_seen_v1");
    state.onboardingSeen = raw === "1";
  } catch (err) {
    state.onboardingSeen = false;
  }
}

function persistOnboardingState() {
  try {
    localStorage.setItem("atom_ui_onboarding_seen_v1", state.onboardingSeen ? "1" : "0");
  } catch (err) {
    // ignore storage failures
  }
}

function persistUiSettings() {
  try {
    localStorage.setItem("atom_ui_settings_v1", JSON.stringify(state.uiSettings));
  } catch (err) {
    // ignore storage failures
  }
}

function applyUiSettings() {
  const body = document.body;
  if (!body) {
    return;
  }
  body.classList.toggle("density-compact", state.uiSettings.density === "compact");
  body.classList.toggle("reduced-motion", !state.uiSettings.motion);
  body.classList.toggle("hide-raw-panels", !state.uiSettings.showRawPanels);

  setSelectValue(refreshIntervalSelect, String(state.uiSettings.refreshMs));
  setSelectValue(densitySelect, state.uiSettings.density);
  if (showRawPanelsToggle) {
    showRawPanelsToggle.checked = Boolean(state.uiSettings.showRawPanels);
  }
  if (motionToggle) {
    motionToggle.checked = Boolean(state.uiSettings.motion);
  }
  if (obstacleOverlayToggle) {
    obstacleOverlayToggle.checked = Boolean(state.uiSettings.showObstacleOverlay);
  }
  if (saliencyFocusToggle) {
    saliencyFocusToggle.checked = Boolean(state.uiSettings.showSaliencyFocus);
  }
  if (narrativeModeSelect) {
    setSelectValue(narrativeModeSelect, state.uiSettings.narrativeMode);
  }

  RAW_SURFACE_IDS.forEach((id) => {
    const node = document.getElementById(id);
    if (!node) {
      return;
    }
    node.classList.add("raw-surface");
  });
}

function renderOnboardingPanel() {
  if (!onboardingPanel) {
    return;
  }
  onboardingPanel.hidden = Boolean(state.onboardingSeen);
}

function setWizardStep(node, done, active) {
  if (!node) {
    return;
  }
  node.classList.toggle("is-done", Boolean(done));
  node.classList.toggle("is-active", Boolean(active));
}

function updateWizardChecklist() {
  const catalogReady = Array.isArray(state.studioCatalog?.simulators) && state.studioCatalog.simulators.length > 0;
  const worldReady = Boolean(worldSelect && worldSelect.value);
  const inverseReady = Array.isArray(state.jobs) && state.jobs.length > 0;
  const supersonicReady = Array.isArray(state.supersonicJobs) && state.supersonicJobs.length > 0;
  const assistantReady = Boolean(state.wizardAssistantCompleted);

  setWizardStep(wizardStepCatalog, catalogReady, !catalogReady);
  setWizardStep(wizardStepWorld, worldReady, catalogReady && !worldReady);
  setWizardStep(wizardStepInverse, inverseReady, catalogReady && worldReady && !inverseReady);
  setWizardStep(
    wizardStepSupersonic,
    supersonicReady,
    catalogReady && worldReady && inverseReady && !supersonicReady
  );
  setWizardStep(
    wizardStepAssistant,
    assistantReady,
    catalogReady && worldReady && inverseReady && supersonicReady && !assistantReady
  );
}

function findDemoByKind(kind) {
  const demos = Array.isArray(state.studioCatalog?.demos) ? state.studioCatalog.demos : [];
  const available = demos.find((demo) => demo?.kind === kind && demo?.available);
  if (available) {
    return available;
  }
  return demos.find((demo) => demo?.kind === kind) || null;
}

function buildNarrativeEntries() {
  if (state.uiSettings.narrativeMode === "off") {
    return [];
  }

  const lines = [];
  const inverseCount = Array.isArray(state.jobs) ? state.jobs.length : 0;
  const supersonicCount = Array.isArray(state.supersonicJobs) ? state.supersonicJobs.length : 0;
  const selectedInverse = state.jobs.find((j) => j.job_id === state.selectedJobId) || null;
  const selectedSupersonic = state.supersonicJobs.find((j) => j.job_id === state.selectedSupersonicJobId) || null;
  const runtime = state.runtimeTelemetry || {};
  const derived = state.supersonicTelemetry?.derived || {};

  if (state.uiSettings.narrativeMode === "executive") {
    lines.push(`Pipeline posture: inverse_jobs=${inverseCount}, supersonic_jobs=${supersonicCount}.`);
    if (selectedSupersonic) {
      lines.push(
        `Control status: ${selectedSupersonic.status}. shock_mean_32=${formatNum(derived.shock_strength_mean_last_32, 4)}, trend_32=${formatNum(derived.shock_strength_slope_last_32, 6)}.`
      );
    }
    if (runtime?.available) {
      lines.push(
        `Trust surface: verified=${formatNum(runtime.trust_verified, 4)} floor=${formatNum(runtime.trust_structural_floor, 4)} step=${runtime.step ?? "n/a"}.`
      );
    }
    if (selectedInverse?.result?.candidates?.length) {
      const best = selectedInverse.result.candidates[0];
      lines.push(
        `Design throughput: top_candidate=${best?.candidate_id || "unknown"} objective=${formatNum(best?.objective_score ?? best?.raw_objective_score, 6)}.`
      );
    }
    if (!lines.length) {
      lines.push("No active evidence yet. Launch guided flow to generate mission telemetry.");
    }
    return lines;
  }

  lines.push(`Technical snapshot: diagnostic_mode=${state.diagnosticMode}, intent=${state.intentFocus}.`);
  if (state.supersonicTelemetry?.timeseries?.step?.length) {
    const points = state.supersonicTelemetry.timeseries.step.length;
    lines.push(`Supersonic telemetry window points=${points}, reward_last=${formatNum(derived.reward_last, 6)}.`);
  }
  if (runtime?.available) {
    const diag = runtime?.diagnostics || {};
    lines.push(
      `Runtime diagnostics: updated_step=${diag.updated_step ?? "n/a"} stale=${diag.stale_steps ?? "n/a"} interval=${diag.interval ?? "n/a"}.`
    );
  }
  const incidents = Array.isArray(state.supersonicTelemetry?.incidents_tail)
    ? state.supersonicTelemetry.incidents_tail.length
    : 0;
  lines.push(`Incident backlog in current window=${incidents}.`);
  return lines;
}

function renderNarrativeFeed() {
  if (!narrativeFeed) {
    return;
  }
  const lines = buildNarrativeEntries();
  if (!lines.length) {
    narrativeFeed.innerHTML = "<div class='narrative-line'>Narrative mode is off.</div>";
    return;
  }
  narrativeFeed.innerHTML = lines
    .map((line) => `<div class='narrative-line'>${escapeHtml(line)}</div>`)
    .join("");
}

function verifyVisualizationContainers() {
  const missing = [];
  const required = [
    { id: "chart-reward", node: rewardChartCanvas },
    { id: "chart-shock", node: shockChartCanvas },
    { id: "chart-saliency", node: saliencyChartCanvas },
    { id: "chart-theory-importance", node: theoryImportanceCanvas },
    { id: "map-live-speed", node: liveSpeedCanvas },
    { id: "map-live-density", node: liveDensityCanvas },
    { id: "map-live-divergence", node: liveDivergenceCanvas },
    { id: "map-attention-overlay", node: attentionOverlayCanvas },
    { id: "map-eyes2-saliency", node: eyesSaliencyCanvas },
    { id: "map-brain-saliency", node: brainSaliencyCanvas },
  ];
  required.forEach((item) => {
    if (!item.node) {
      missing.push(item.id);
    }
  });
  if (missing.length) {
    pushStudioEvent("Visualization container mismatch detected", "error", { missing });
  } else {
    pushStudioEvent("Visualization containers verified", "info", { count: required.length });
  }
}

function setKpiValue(node, value, alert = false) {
  if (!node) {
    return;
  }
  node.textContent = value;
  node.classList.toggle("alert", Boolean(alert));
}

function setSelectValue(node, value) {
  if (!node) {
    return;
  }
  const normalized = String(value || "");
  const hasOption = Array.from(node.options || []).some((opt) => opt.value === normalized);
  if (hasOption) {
    node.value = normalized;
  }
}

function preferredDiagnosticModeForIntent(intent) {
  const key = String(intent || "scientific_discovery");
  if (key === "scientific_discovery") {
    return "trust_surface";
  }
  if (key === "inverse_design") {
    return "overview";
  }
  if (key === "engineering") {
    return "live_flow";
  }
  return "brain_saliency";
}

function renderIntentButtons() {
  missionIntentButtons.forEach((btn) => {
    const intent = String(btn.dataset.intent || "");
    const active = intent === state.intentFocus;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
}

function syncIntentFocus(value, source = null, alignDiagnostic = false) {
  state.intentFocus = String(value || "scientific_discovery");
  if (source !== "observability") {
    setSelectValue(intentFocusSelect, state.intentFocus);
  }
  if (source !== "assistant") {
    setSelectValue(assistantIntentSelect, state.intentFocus);
  }
  renderIntentButtons();
  if (alignDiagnostic) {
    syncDiagnosticMode(preferredDiagnosticModeForIntent(state.intentFocus));
  }
  renderNarrativeFeed();
}

function syncDiagnosticMode(value, source = null) {
  state.diagnosticMode = String(value || "overview");
  if (source !== "observability") {
    setSelectValue(diagnosticModeSelect, state.diagnosticMode);
  }
  if (source !== "assistant") {
    setSelectValue(assistantDiagnosticModeSelect, state.diagnosticMode);
  }
  diagnosticCards.forEach((card) => {
    const focusRaw = String(card.dataset.focus || "");
    const foci = focusRaw
      .split(",")
      .map((token) => token.trim())
      .filter(Boolean);
    const visible =
      state.diagnosticMode === "overview" ||
      foci.includes(state.diagnosticMode);
    card.classList.toggle("is-hidden", !visible);
  });
}

function syncAssistantEngine(value) {
  state.assistantEngine = String(value || "deterministic");
  setSelectValue(assistantEngineSelect, state.assistantEngine);
}

function pushStudioEvent(message, level = "info", details = null) {
  if (!studioLog) {
    return;
  }
  const event = {
    timestamp: new Date().toISOString(),
    level,
    message: String(message || ""),
  };
  if (details !== null && details !== undefined) {
    event.details = details;
  }
  state.studioEvents.unshift(event);
  state.studioEvents = state.studioEvents.slice(0, 24);
  studioLog.textContent = JSON.stringify(
    {
      status: "studio_active",
      events: state.studioEvents,
    },
    null,
    2
  );
}

function setFormField(formNode, name, value) {
  if (!formNode) {
    return;
  }
  const field = formNode.elements.namedItem(name);
  if (!field) {
    return;
  }
  if (value === null || value === undefined) {
    return;
  }
  if (field instanceof RadioNodeList) {
    field.value = String(value);
    return;
  }
  field.value = String(value);
}

function applyDemoTemplate(demo) {
  if (!demo || typeof demo !== "object") {
    return;
  }
  const payload = demo.payload || {};
  if (demo.kind === "inverse_design") {
    setFormField(form, "name", payload.name || `demo_${demo.id}`);
    setFormField(form, "world_spec", payload.world_spec || demo.simulator || "");
    setFormField(form, "backend", payload.backend || "evolutionary");
    setFormField(form, "iterations", payload.iterations ?? 6);
    setFormField(form, "population", payload.population ?? 10);
    setFormField(form, "rollout_steps", payload.rollout_steps ?? 64);
    setFormField(form, "top_k", payload.top_k ?? 5);
    const grid = Array.isArray(payload.grid_shape) ? payload.grid_shape : [32, 32, 16];
    setFormField(form, "grid_shape", `${grid[0]},${grid[1]},${grid[2]}`);
    const demoWorld = normalizeWorldSpec(payload.world_spec || demo.simulator || "");
    if (demoWorld) {
      if (payload.world_kwargs && typeof payload.world_kwargs === "object") {
        setWorldKwDraft(demoWorld, payload.world_kwargs);
      }
      renderWorldKwEditor(demoWorld);
    }
    pushStudioEvent(`Loaded inverse template: ${demo.title}`, "template", { demo_id: demo.id });
    return;
  }
  if (demo.kind === "supersonic_challenge") {
    setFormField(supersonicForm, "name", payload.name || `demo_${demo.id}`);
    setFormField(supersonicForm, "steps", payload.steps ?? 192);
    pushStudioEvent(`Loaded supersonic template: ${demo.title}`, "template", { demo_id: demo.id });
  }
}

function drawLineChart(canvas, x, series) {
  if (!canvas) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(180, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, width, height);

  const padding = { top: 20, right: 16, bottom: 24, left: 36 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const allValues = [];
  series.forEach((item) => {
    (item.values || []).forEach((v) => {
      const n = Number(v);
      if (Number.isFinite(n)) {
        allValues.push(n);
      }
    });
  });

  if (!x.length || !allValues.length) {
    ctx.fillStyle = "rgba(153, 245, 194, 0.8)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText("Waiting for telemetry...", 18, 28);
    return;
  }

  let minY = Math.min(...allValues);
  let maxY = Math.max(...allValues);
  if (Math.abs(maxY - minY) < 1e-9) {
    minY -= 1;
    maxY += 1;
  }

  const toX = (idx) => {
    const denom = Math.max(1, x.length - 1);
    return padding.left + (idx / denom) * innerWidth;
  };
  const toY = (value) => {
    const ratio = (value - minY) / (maxY - minY);
    return padding.top + (1 - ratio) * innerHeight;
  };

  ctx.strokeStyle = "rgba(153, 245, 194, 0.18)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (i / 4) * innerHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(153, 245, 194, 0.45)";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  series.forEach((item) => {
    const values = item.values || [];
    ctx.strokeStyle = item.color || "#99f5c2";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    values.forEach((value, idx) => {
      const px = toX(idx);
      const py = toY(Number(value));
      if (idx === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    });
    ctx.stroke();
  });

  let legendX = padding.left;
  series.forEach((item) => {
    ctx.fillStyle = item.color || "#99f5c2";
    ctx.fillRect(legendX, 6, 10, 10);
    ctx.fillStyle = "rgba(153, 245, 194, 0.92)";
    ctx.font = "11px Menlo, Consolas, monospace";
    ctx.fillText(item.label, legendX + 14, 15);
    legendX += 100;
  });
}

function drawBarChart(canvas, values, labels) {
  if (!canvas) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(160, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, width, height);

  const vals = Array.isArray(values) ? values.map((v) => Number(v) || 0) : [];
  if (!vals.length) {
    ctx.fillStyle = "rgba(153, 245, 194, 0.8)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText("No saliency data yet...", 18, 24);
    return;
  }

  const maxVal = Math.max(1e-6, ...vals);
  const padding = { top: 16, right: 12, bottom: 24, left: 12 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const barWidth = innerWidth / vals.length;

  vals.forEach((v, i) => {
    const ratio = Math.max(0, Math.min(1, v / maxVal));
    const h = ratio * innerHeight;
    const x = padding.left + i * barWidth + barWidth * 0.08;
    const y = padding.top + innerHeight - h;
    const w = barWidth * 0.84;
    ctx.fillStyle = "rgba(126, 200, 255, 0.85)";
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = "rgba(153, 245, 194, 0.9)";
    ctx.font = "10px Menlo, Consolas, monospace";
    ctx.fillText(labels[i] || `L${i}`, x, height - 8);
  });
}

function sampleHeatColor(t, palette = "mint") {
  const clamped = Math.max(0, Math.min(1, Number(t) || 0));
  if (palette === "divergence") {
    const r = Math.round(35 + clamped * 220);
    const b = Math.round(220 - clamped * 180);
    const g = Math.round(90 + (1 - Math.abs(0.5 - clamped) * 2) * 80);
    return `rgb(${r},${g},${b})`;
  }
  if (palette === "amber") {
    const r = Math.round(55 + clamped * 200);
    const g = Math.round(45 + clamped * 170);
    const b = Math.round(18 + clamped * 90);
    return `rgb(${r},${g},${b})`;
  }
  const r = Math.round(14 + clamped * 74);
  const g = Math.round(42 + clamped * 210);
  const b = Math.round(32 + clamped * 160);
  return `rgb(${r},${g},${b})`;
}

function drawMaskOverlay(ctx, overlayMap, pad, mapWidth, mapHeight) {
  const overlayRows = Array.isArray(overlayMap?.map_xy) ? overlayMap.map_xy : [];
  const rowCount = overlayRows.length;
  const colCount = rowCount > 0 && Array.isArray(overlayRows[0]) ? overlayRows[0].length : 0;
  if (!rowCount || !colCount) {
    return false;
  }

  const threshold = 0.45;
  const cellW = mapWidth / colCount;
  const cellH = mapHeight / rowCount;
  let hasObstacle = false;

  ctx.save();
  ctx.strokeStyle = "rgba(255, 231, 146, 0.86)";
  ctx.lineWidth = 1;
  for (let y = 0; y < rowCount; y += 1) {
    for (let x = 0; x < colCount; x += 1) {
      const value = Number(overlayRows[y][x]);
      if (!Number.isFinite(value) || value <= threshold) {
        continue;
      }
      hasObstacle = true;
      const x0 = pad + x * cellW;
      const y0 = pad + y * cellH;
      const x1 = x0 + cellW;
      const y1 = y0 + cellH;

      ctx.fillStyle = "rgba(8, 14, 18, 0.28)";
      ctx.fillRect(x0, y0, cellW, cellH);

      const left = x === 0 || Number(overlayRows[y][x - 1]) <= threshold;
      const right = x === colCount - 1 || Number(overlayRows[y][x + 1]) <= threshold;
      const top = y === 0 || Number(overlayRows[y - 1][x]) <= threshold;
      const bottom = y === rowCount - 1 || Number(overlayRows[y + 1][x]) <= threshold;

      if (left) {
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0, y1);
        ctx.stroke();
      }
      if (right) {
        ctx.beginPath();
        ctx.moveTo(x1, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
      if (top) {
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y0);
        ctx.stroke();
      }
      if (bottom) {
        ctx.beginPath();
        ctx.moveTo(x0, y1);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }
  }
  ctx.restore();
  return hasObstacle;
}

function extractPeakFromMap(mapBlock) {
  const rows = Array.isArray(mapBlock?.map_xy) ? mapBlock.map_xy : [];
  const rowCount = rows.length;
  const colCount = rowCount > 0 && Array.isArray(rows[0]) ? rows[0].length : 0;
  if (!rowCount || !colCount) {
    return null;
  }

  let peakValue = Number.NEGATIVE_INFINITY;
  let peakX = 0;
  let peakY = 0;
  let found = false;
  for (let y = 0; y < rowCount; y += 1) {
    for (let x = 0; x < colCount; x += 1) {
      const value = Number(rows[y][x]);
      if (!Number.isFinite(value)) {
        continue;
      }
      if (!found || value > peakValue) {
        peakValue = value;
        peakX = x;
        peakY = y;
        found = true;
      }
    }
  }
  if (!found) {
    return null;
  }

  const xNorm = colCount > 1 ? peakX / (colCount - 1) : 0.0;
  const yNorm = rowCount > 1 ? peakY / (rowCount - 1) : 0.0;
  return {
    x: peakX,
    y: peakY,
    value: peakValue,
    x_norm: xNorm,
    y_norm: yNorm,
    cols: colCount,
    rows: rowCount,
  };
}

function sampleMapByNormalized(mapBlock, xNorm, yNorm) {
  const rows = Array.isArray(mapBlock?.map_xy) ? mapBlock.map_xy : [];
  const rowCount = rows.length;
  const colCount = rowCount > 0 && Array.isArray(rows[0]) ? rows[0].length : 0;
  if (!rowCount || !colCount) {
    return null;
  }
  const safeX = Math.max(0, Math.min(1, Number(xNorm) || 0));
  const safeY = Math.max(0, Math.min(1, Number(yNorm) || 0));
  const x = Math.round(safeX * Math.max(1, colCount - 1));
  const y = Math.round(safeY * Math.max(1, rowCount - 1));
  const value = Number(rows[y]?.[x]);
  if (!Number.isFinite(value)) {
    return null;
  }
  return { x, y, value, cols: colCount, rows: rowCount };
}

function focusDistanceNormalized(a, b) {
  if (!a || !b) {
    return null;
  }
  const ax = Number(a.x_norm);
  const ay = Number(a.y_norm);
  const bx = Number(b.x_norm);
  const by = Number(b.y_norm);
  if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) {
    return null;
  }
  const dx = ax - bx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}

function focusPeakToCanvasPoint(focusPeak, pad, mapWidth, mapHeight) {
  if (!focusPeak || !Number.isFinite(focusPeak.x) || !Number.isFinite(focusPeak.y)) {
    return null;
  }
  const cols = Math.max(1, Number(focusPeak.cols) || 1);
  const rows = Math.max(1, Number(focusPeak.rows) || 1);
  const x = pad + ((focusPeak.x + 0.5) / cols) * mapWidth;
  const y = pad + ((focusPeak.y + 0.5) / rows) * mapHeight;
  return { x, y };
}

function drawFocusConnection(ctx, fromPeak, toPeak, pad, mapWidth, mapHeight, color) {
  const from = focusPeakToCanvasPoint(fromPeak, pad, mapWidth, mapHeight);
  const to = focusPeakToCanvasPoint(toPeak, pad, mapWidth, mapHeight);
  if (!from || !to) {
    return false;
  }
  ctx.save();
  ctx.strokeStyle = color || "rgba(214, 244, 255, 0.94)";
  ctx.lineWidth = 1.3;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.restore();
  return true;
}

function drawFocusMarker(ctx, focusPeak, pad, mapWidth, mapHeight, label, color) {
  if (!focusPeak || !Number.isFinite(focusPeak.x) || !Number.isFinite(focusPeak.y)) {
    return false;
  }
  const point = focusPeakToCanvasPoint(focusPeak, pad, mapWidth, mapHeight);
  if (!point) {
    return false;
  }
  const x = point.x;
  const y = point.y;
  const markerColor = color || "rgba(255, 226, 128, 0.95)";
  const markerLabel = String(label || "").trim();

  ctx.save();
  ctx.strokeStyle = markerColor;
  ctx.fillStyle = markerColor;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x - 8, y);
  ctx.lineTo(x + 8, y);
  ctx.moveTo(x, y - 8);
  ctx.lineTo(x, y + 8);
  ctx.stroke();
  if (markerLabel) {
    ctx.font = "10px Menlo, Consolas, monospace";
    ctx.fillText(markerLabel, Math.min(x + 8, pad + mapWidth - 80), Math.max(y - 8, pad + 10));
  }
  ctx.restore();
  return true;
}

function drawHeatmap(canvas, mapBlock, options = {}) {
  if (!canvas) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(170, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, width, height);

  const rows = Array.isArray(mapBlock?.map_xy) ? mapBlock.map_xy : [];
  const rowCount = rows.length;
  const colCount = rowCount > 0 && Array.isArray(rows[0]) ? rows[0].length : 0;
  if (!rowCount || !colCount) {
    ctx.fillStyle = "rgba(153, 245, 194, 0.82)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText("No map data yet...", 18, 26);
    return;
  }

  const flat = [];
  rows.forEach((row) => {
    row.forEach((value) => {
      const n = Number(value);
      if (Number.isFinite(n)) {
        flat.push(n);
      }
    });
  });
  let min = Math.min(...flat);
  let max = Math.max(...flat);
  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-12) {
    min = 0.0;
    max = 1.0;
  }

  const pad = 10;
  const infoPad = 20;
  const mapWidth = width - pad * 2;
  const mapHeight = height - pad * 2 - infoPad;
  const cellW = mapWidth / colCount;
  const cellH = mapHeight / rowCount;
  const palette = String(options.palette || "mint");
  rows.forEach((row, y) => {
    row.forEach((value, x) => {
      const n = Number(value);
      const t = (n - min) / (max - min);
      ctx.fillStyle = sampleHeatColor(t, palette);
      ctx.fillRect(
        Math.floor(pad + x * cellW),
        Math.floor(pad + y * cellH),
        Math.ceil(cellW),
        Math.ceil(cellH)
      );
    });
  });

  ctx.strokeStyle = "rgba(153, 245, 194, 0.35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, mapWidth, mapHeight);
  const overlayEnabled = options.overlayEnabled !== false;
  const overlayDetected = overlayEnabled
    ? drawMaskOverlay(ctx, options.overlayMap, pad, mapWidth, mapHeight)
    : false;
  const normalizeFocusPeak = (rawPeak) => {
    if (!rawPeak) {
      return null;
    }
    const pxNorm = Number(rawPeak.x_norm);
    const pyNorm = Number(rawPeak.y_norm);
    if (Number.isFinite(pxNorm) && Number.isFinite(pyNorm)) {
      return {
        x: Math.round(Math.max(0, Math.min(1, pxNorm)) * Math.max(1, colCount - 1)),
        y: Math.round(Math.max(0, Math.min(1, pyNorm)) * Math.max(1, rowCount - 1)),
        cols: colCount,
        rows: rowCount,
      };
    }
    return {
      x: Number(rawPeak.x) || 0,
      y: Number(rawPeak.y) || 0,
      cols: colCount,
      rows: rowCount,
    };
  };
  const focusPeak = normalizeFocusPeak(options.focusPeak);
  const focusEnabled = options.focusEnabled !== false;
  const focusDetected = focusEnabled
    ? drawFocusMarker(
      ctx,
      focusPeak,
      pad,
      mapWidth,
      mapHeight,
      options.focusLabel,
      options.focusColor
    )
    : false;
  const extraFocusEntries = Array.isArray(options.extraFocusPeaks) ? options.extraFocusPeaks : [];
  let extraFocusDetected = false;
  let firstExtraPeak = null;
  extraFocusEntries.forEach((entry) => {
    const peak = normalizeFocusPeak(entry?.peak);
    if (!peak) {
      return;
    }
    if (!firstExtraPeak) {
      firstExtraPeak = peak;
    }
    const drawn = focusEnabled
      ? drawFocusMarker(
        ctx,
        peak,
        pad,
        mapWidth,
        mapHeight,
        entry?.label,
        entry?.color
      )
      : false;
    extraFocusDetected = extraFocusDetected || drawn;
  });
  const connectionDetected =
    focusEnabled &&
    Boolean(options.connectFocusPeaks) &&
    drawFocusConnection(
      ctx,
      focusPeak,
      firstExtraPeak,
      pad,
      mapWidth,
      mapHeight,
      options.focusLinkColor
    );
  ctx.fillStyle = "rgba(153, 245, 194, 0.88)";
  ctx.font = "11px Menlo, Consolas, monospace";
  const rawMin = Number(mapBlock?.raw_min);
  const rawMax = Number(mapBlock?.raw_max);
  const stride = Number.isFinite(Number(mapBlock?.stride)) ? Number(mapBlock.stride) : 1;
  const rawLabel = Number.isFinite(rawMin) && Number.isFinite(rawMax)
    ? `raw=[${rawMin.toFixed(4)}, ${rawMax.toFixed(4)}]`
    : `norm=[${min.toFixed(3)}, ${max.toFixed(3)}]`;
  ctx.fillText(
    `shape=${rowCount}x${colCount} stride=${stride} ${rawLabel}`,
    pad,
    height - 8
  );
  if (overlayDetected) {
    const overlayLabel = " + obstacle overlay";
    const textWidth = ctx.measureText(overlayLabel).width;
    ctx.fillText(overlayLabel, width - textWidth - pad, height - 8);
  }
  if (focusDetected) {
    const focusLabel = " + focus marker";
    const textWidth = ctx.measureText(focusLabel).width;
    const baseOffset = overlayDetected ? 118 : 0;
    ctx.fillText(focusLabel, width - textWidth - pad - baseOffset, height - 8);
  }
  if (extraFocusDetected) {
    const focusLabel = " + multi-focus";
    const textWidth = ctx.measureText(focusLabel).width;
    const baseOffset = (overlayDetected ? 118 : 0) + (focusDetected ? 96 : 0);
    ctx.fillText(focusLabel, width - textWidth - pad - baseOffset, height - 8);
  }
  if (connectionDetected) {
    const focusLabel = " + focus-link";
    const textWidth = ctx.measureText(focusLabel).width;
    const baseOffset = (overlayDetected ? 118 : 0) + (focusDetected ? 96 : 0) + (extraFocusDetected ? 88 : 0);
    ctx.fillText(focusLabel, width - textWidth - pad - baseOffset, height - 8);
  }
}

function updateInverseKpis(job) {
  const candidates = job?.result?.candidates;
  if (!Array.isArray(candidates) || !candidates.length) {
    setKpiValue(kpiInverseScore, "n/a");
    setKpiValue(kpiInverseFeasible, "n/a");
    return;
  }
  const sorted = [...candidates].sort((a, b) => {
    const rankA = Number.isFinite(a?.rank) ? a.rank : Number.MAX_SAFE_INTEGER;
    const rankB = Number.isFinite(b?.rank) ? b.rank : Number.MAX_SAFE_INTEGER;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return (a?.objective_score ?? Number.MAX_VALUE) - (b?.objective_score ?? Number.MAX_VALUE);
  });
  const best = sorted[0];
  const feasible = candidates.filter((c) => Boolean(c?.feasible)).length;
  const feasibleRate = feasible / candidates.length;
  setKpiValue(kpiInverseScore, formatNum(best?.objective_score ?? best?.raw_objective_score, 6));
  setKpiValue(kpiInverseFeasible, `${(feasibleRate * 100).toFixed(1)}%`, feasibleRate < 0.7);
}

function updateSupersonicKpis(telemetry) {
  const derived = telemetry?.derived || {};
  const reward = Number(derived.reward_mean_last_32);
  const shock = Number(derived.shock_strength_mean_last_32);
  const reduction = Number(derived.shock_reduction_mean_last_32);
  const trend = Number(derived.shock_strength_slope_last_32);
  setKpiValue(kpiReward, formatNum(reward, 5));
  setKpiValue(kpiShock, formatNum(shock, 5), shock > 0.2);
  setKpiValue(kpiReduction, formatNum(reduction, 5));
  setKpiValue(kpiShockTrend, formatNum(trend, 6), trend > 0.002);
}

function renderSupersonicCharts(telemetry) {
  const timeseries = telemetry?.timeseries || {};
  const x = Array.isArray(timeseries.step) ? timeseries.step : [];
  const reward = Array.isArray(timeseries.reward) ? timeseries.reward : [];
  const reduction = Array.isArray(timeseries.shock_reduction) ? timeseries.shock_reduction : [];
  const shock = Array.isArray(timeseries.shock_strength) ? timeseries.shock_strength : [];
  const jet = Array.isArray(timeseries.jet_power) ? timeseries.jet_power : [];

  drawLineChart(rewardChartCanvas, x, [
    { label: "reward", values: reward, color: "#85f7be" },
    { label: "shock_reduction", values: reduction, color: "#f3d27a" },
  ]);
  drawLineChart(shockChartCanvas, x, [
    { label: "shock_strength", values: shock, color: "#ff9b80" },
    { label: "jet_power", values: jet, color: "#7ec8ff" },
  ]);
}

function renderRuntimeTelemetry(snapshot) {
  if (!runtimeTelemetry) {
    return;
  }
  if (!snapshot || !snapshot.available) {
    runtimeTelemetry.textContent = JSON.stringify(
      { status: "runtime_telemetry_unavailable" },
      null,
      2
    );
    drawBarChart(saliencyChartCanvas, [], []);
    drawBarChart(theoryImportanceCanvas, [], []);
    drawHeatmap(liveSpeedCanvas, null);
    drawHeatmap(liveDensityCanvas, null);
    drawHeatmap(liveDivergenceCanvas, null);
    drawHeatmap(attentionOverlayCanvas, null);
    drawHeatmap(eyesSaliencyCanvas, null);
    drawHeatmap(brainSaliencyCanvas, null);
    if (attentionOverlaySummary) {
      attentionOverlaySummary.textContent = "attention alignment unavailable";
    }
    if (runtimeDiagnosticsStatus) {
      runtimeDiagnosticsStatus.textContent = "runtime diagnostics: unavailable";
    }
    return;
  }

  const diagnostics = snapshot.diagnostics || {};
  const liveView = diagnostics.live_view || {};
  const eyes = diagnostics.eyes2_saliency || {};
  const brain = diagnostics.brain_saliency || {};
  const obstacle = liveView.obstacle_xy || null;
  const overlayEnabled = Boolean(state.uiSettings.showObstacleOverlay);
  const focusEnabled = Boolean(state.uiSettings.showSaliencyFocus);
  const eyesPeak = extractPeakFromMap(eyes.map);
  const brainPeak = extractPeakFromMap(brain.map);
  const eyesBrainFocusDistance = focusDistanceNormalized(eyesPeak, brainPeak);
  const speedAtEyesPeak = eyesPeak
    ? sampleMapByNormalized(liveView.speed_xy, eyesPeak.x_norm, eyesPeak.y_norm)
    : null;
  const speedAtBrainPeak = brainPeak
    ? sampleMapByNormalized(liveView.speed_xy, brainPeak.x_norm, brainPeak.y_norm)
    : null;

  const latentSaliency = Array.isArray(snapshot.saliency?.jacobian_abs_normalized)
    ? snapshot.saliency.jacobian_abs_normalized
    : [];
  const latentLabels = latentSaliency.map((_, idx) => `latent_${idx}`);
  drawBarChart(saliencyChartCanvas, latentSaliency, latentLabels);

  const rawTheoryImportance = Array.isArray(brain.theory_feature_importance)
    ? brain.theory_feature_importance
    : Array.isArray(snapshot.saliency?.theory_feature_importance)
      ? snapshot.saliency.theory_feature_importance
      : [];
  const theoryImportance = rawTheoryImportance.map((value) => Number(value) || 0);
  const theoryLabelsRaw = Array.isArray(brain.theory_feature_labels)
    ? brain.theory_feature_labels
    : Array.isArray(snapshot.saliency?.theory_feature_labels)
      ? snapshot.saliency.theory_feature_labels
      : [];
  const theoryLabels =
    theoryLabelsRaw.length === theoryImportance.length
      ? theoryLabelsRaw
      : theoryImportance.map((_, idx) => `theory_${idx}`);
  drawBarChart(theoryImportanceCanvas, theoryImportance, theoryLabels);

  drawHeatmap(liveSpeedCanvas, liveView.speed_xy, {
    palette: "mint",
    overlayMap: obstacle,
    overlayEnabled,
    focusEnabled,
    focusPeak: eyesPeak,
    focusLabel: "eyes focus",
    focusColor: "rgba(255, 228, 138, 0.95)",
  });
  drawHeatmap(liveDensityCanvas, liveView.density_xy, {
    palette: "amber",
    overlayMap: obstacle,
    overlayEnabled,
    focusEnabled,
    focusPeak: eyesPeak,
    focusLabel: "eyes focus",
    focusColor: "rgba(255, 228, 138, 0.95)",
  });
  drawHeatmap(liveDivergenceCanvas, liveView.divergence_xy, {
    palette: "divergence",
    overlayMap: obstacle,
    overlayEnabled,
    focusEnabled,
    focusPeak: eyesPeak,
    focusLabel: "eyes focus",
    focusColor: "rgba(255, 228, 138, 0.95)",
  });
  drawHeatmap(attentionOverlayCanvas, liveView.speed_xy, {
    palette: "mint",
    overlayMap: obstacle,
    overlayEnabled,
    focusEnabled,
    focusPeak: eyesPeak,
    focusLabel: "eyes",
    focusColor: "rgba(255, 228, 138, 0.95)",
    extraFocusPeaks: [
      {
        peak: brainPeak,
        label: "brain",
        color: "rgba(143, 225, 255, 0.96)",
      },
    ],
    connectFocusPeaks: true,
    focusLinkColor: "rgba(196, 233, 255, 0.94)",
  });
  drawHeatmap(eyesSaliencyCanvas, eyes.map, {
    palette: "amber",
    overlayMap: obstacle,
    overlayEnabled,
    focusEnabled,
    focusPeak: eyesPeak,
    focusLabel: "peak",
    focusColor: "rgba(255, 228, 138, 0.95)",
  });
  drawHeatmap(brainSaliencyCanvas, brain.map, {
    palette: "mint",
    overlayMap: obstacle,
    overlayEnabled,
    focusEnabled,
    focusPeak: brainPeak,
    focusLabel: "peak",
    focusColor: "rgba(143, 225, 255, 0.96)",
  });
  if (attentionOverlaySummary) {
    const eyeLabel = eyesPeak
      ? `eyes=(${eyesPeak.x},${eyesPeak.y}) v=${formatNum(eyesPeak.value, 4)}`
      : "eyes=n/a";
    const brainLabel = brainPeak
      ? `brain=(${brainPeak.x},${brainPeak.y}) v=${formatNum(brainPeak.value, 4)}`
      : "brain=n/a";
    const distanceLabel = Number.isFinite(Number(eyesBrainFocusDistance))
      ? `focus_distance=${formatNum(eyesBrainFocusDistance, 4)}`
      : "focus_distance=n/a";
    const flowEyes = speedAtEyesPeak ? `flow@eyes=${formatNum(speedAtEyesPeak.value, 4)}` : "flow@eyes=n/a";
    const flowBrain = speedAtBrainPeak ? `flow@brain=${formatNum(speedAtBrainPeak.value, 4)}` : "flow@brain=n/a";
    attentionOverlaySummary.textContent = `${eyeLabel} | ${brainLabel} | ${distanceLabel} | ${flowEyes} | ${flowBrain}`;
  }

  if (runtimeDiagnosticsStatus) {
    const updatedStep = Number(diagnostics.updated_step ?? 0);
    const staleSteps = Number(diagnostics.stale_steps ?? 0);
    const interval = Number(diagnostics.interval ?? 0);
    runtimeDiagnosticsStatus.textContent =
      `runtime diagnostics: updated_step=${updatedStep} stale=${staleSteps} interval=${interval} overlay=${overlayEnabled ? "on" : "off"} focus=${focusEnabled ? "on" : "off"}`;
  }

  const payload = {
    step: snapshot.step,
    reward: snapshot.reward,
    stress: snapshot.stress,
    theory: snapshot.theory,
    trust_raw: snapshot.trust_raw,
    trust_verified: snapshot.trust_verified,
    trust_structural_floor: snapshot.trust_structural_floor,
    diagnostics: {
      enabled: diagnostics.enabled,
      updated_step: diagnostics.updated_step,
      stale_steps: diagnostics.stale_steps,
      interval: diagnostics.interval,
      live_projection: liveView.projection || null,
      live_obstacle_shape: obstacle?.shape || null,
      overlay_enabled: overlayEnabled,
      focus_markers_enabled: focusEnabled,
      eyes_objective: eyes.objective ?? null,
      eyes_target_label: eyes.target_label ?? null,
      eyes_target_index: eyes.target_index ?? null,
      eyes_peak: eyesPeak
        ? {
          x_index: eyesPeak.x,
          y_index: eyesPeak.y,
          x_norm: eyesPeak.x_norm,
          y_norm: eyesPeak.y_norm,
          value: eyesPeak.value,
        }
        : null,
      live_speed_at_eyes_peak: speedAtEyesPeak
        ? {
          x_index: speedAtEyesPeak.x,
          y_index: speedAtEyesPeak.y,
          value: speedAtEyesPeak.value,
        }
        : null,
      live_speed_at_brain_peak: speedAtBrainPeak
        ? {
          x_index: speedAtBrainPeak.x,
          y_index: speedAtBrainPeak.y,
          value: speedAtBrainPeak.value,
        }
        : null,
      eyes_brain_focus_distance: eyesBrainFocusDistance,
      brain_objective: brain.objective ?? null,
      brain_peak: brainPeak
        ? {
          x_index: brainPeak.x,
          y_index: brainPeak.y,
          x_norm: brainPeak.x_norm,
          y_norm: brainPeak.y_norm,
          value: brainPeak.value,
        }
        : null,
    },
    top_latents: snapshot.saliency?.top_latents || [],
    top_theory_features: theoryLabels
      .map((label, idx) => ({ label, importance: theoryImportance[idx] || 0 }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 4),
    hypothesis: snapshot.hypothesis_record?.equation || null,
    hypotheses_tail: snapshot.hypotheses_tail || [],
  };
  runtimeTelemetry.textContent = JSON.stringify(payload, null, 2);
}

function renderStudioSimulators(simulators) {
  if (!studioSimulators) {
    return;
  }
  if (!Array.isArray(simulators) || !simulators.length) {
    studioSimulators.innerHTML = "<div class='studio-card'>No simulators discovered.</div>";
    return;
  }
  studioSimulators.innerHTML = "";
  simulators.forEach((sim) => {
    const card = document.createElement("article");
    const family = String(sim.family || "custom");
    card.className = "studio-card";
    card.innerHTML = `
      <div class="studio-card-head">
        <span class="studio-card-title">${escapeHtml(sim.key || "unknown")}</span>
        <span class="studio-chip family-${escapeHtml(family)}">${escapeHtml(family)}</span>
      </div>
      <p>${escapeHtml(sim.description || "")}</p>
      <div class="studio-meta">
        <span class="studio-chip">${escapeHtml(sim.dimensions || "n/a")}</span>
        <span class="studio-chip">${escapeHtml(sim.control_surface || "n/a")}</span>
      </div>
    `;
    studioSimulators.appendChild(card);
  });
}

async function launchStudioDemo(demo) {
  if (!demo || !demo.available) {
    pushStudioEvent(`Demo unavailable: ${demo?.id || "unknown"}`, "warn");
    return;
  }
  const endpoint = String(demo.launch_endpoint || "");
  if (!endpoint) {
    pushStudioEvent(`Demo missing launch endpoint: ${demo.id}`, "error");
    return;
  }

  const payload = JSON.parse(JSON.stringify(demo.payload || {}));
  if (demo.kind === "inverse_design") {
    payload.device = payload.device || "cpu";
  } else if (demo.kind === "supersonic_challenge") {
    payload.headless = payload.headless !== false;
  }

  pushStudioEvent(`Launching demo: ${demo.title}`, "launch", { endpoint, payload });
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Launch failed (${res.status}): ${text}`);
  }
  const data = await res.json();
  pushStudioEvent(`Demo launched: ${demo.title}`, "success", data);
  if (demo.kind === "inverse_design") {
    state.selectedJobId = data.job_id;
    await fetchJobs();
  } else if (demo.kind === "supersonic_challenge") {
    state.selectedSupersonicJobId = data.job_id;
    await fetchSupersonicJobs();
  }
}

function renderStudioDemos(demos) {
  if (!studioDemos) {
    return;
  }
  if (!Array.isArray(demos) || !demos.length) {
    studioDemos.innerHTML = "<div class='studio-card'>No demo programs found.</div>";
    return;
  }
  studioDemos.innerHTML = "";
  demos.forEach((demo) => {
    const card = document.createElement("article");
    const available = Boolean(demo.available);
    const availabilityClass = available ? "availability-ok" : "availability-failed";
    const kindLabel = demo.kind === "supersonic_challenge" ? "supersonic" : "inverse";
    const tagChips = Array.isArray(demo.tags)
      ? demo.tags.map((tag) => `<span class="studio-chip">${escapeHtml(tag)}</span>`).join("")
      : "";
    card.className = "studio-card";
    card.innerHTML = `
      <div class="studio-card-head">
        <span class="studio-card-title">${escapeHtml(demo.title || demo.id || "demo")}</span>
        <span class="studio-chip ${availabilityClass}">${available ? "ready" : "blocked"}</span>
      </div>
      <p>${escapeHtml(demo.description || "")}</p>
      <div class="studio-meta">
        <span class="studio-chip">${escapeHtml(kindLabel)}</span>
        <span class="studio-chip">${escapeHtml(demo.simulator || "n/a")}</span>
        ${tagChips}
      </div>
      <div class="studio-actions">
        <button type="button" data-action="load">Load Template</button>
        <button type="button" data-action="launch" ${available ? "" : "disabled"}>Launch Demo</button>
      </div>
    `;
    const loadBtn = card.querySelector("button[data-action='load']");
    const launchBtn = card.querySelector("button[data-action='launch']");
    if (loadBtn) {
      loadBtn.onclick = () => applyDemoTemplate(demo);
    }
    if (launchBtn) {
      launchBtn.onclick = async () => {
        try {
          await launchStudioDemo(demo);
        } catch (err) {
          pushStudioEvent(
            `Launch error for ${demo.title || demo.id}`,
            "error",
            String(err?.message || err)
          );
        }
      };
    }
    studioDemos.appendChild(card);
  });
}

function renderGeometryAssets() {
  if (geometrySelect) {
    const previous = state.selectedGeometryId || "";
    geometrySelect.innerHTML = "<option value=''>none</option>";
    state.geometryAssets.forEach((asset) => {
      const opt = document.createElement("option");
      opt.value = String(asset.geometry_id || "");
      const ext = String(asset.extension || "").toLowerCase();
      const sizeMb = Number(asset.bytes || 0) / (1024 * 1024);
      opt.textContent = `${asset.filename || asset.geometry_id} ${ext} (${sizeMb.toFixed(2)} MB)`;
      geometrySelect.appendChild(opt);
    });
    if (previous) {
      geometrySelect.value = previous;
    }
  }
  renderWorldKwEditor(worldSelect?.value || "");

  if (!geometryList) {
    return;
  }
  if (!state.geometryAssets.length) {
    geometryList.innerHTML = "<div class='job-item'>No uploaded geometries.</div>";
    return;
  }
  geometryList.innerHTML = "";
  state.geometryAssets.forEach((asset) => {
    const item = document.createElement("div");
    item.className = "job-item";
    const ext = String(asset.extension || "").toLowerCase();
    const quality = asset.is_stl ? "stl-ready" : "cad";
    const sizeMb = Number(asset.bytes || 0) / (1024 * 1024);
    item.innerHTML = `
      <div class="job-head">
        <strong>${escapeHtml(asset.filename || asset.geometry_id || "geometry")}</strong>
        <span class="status ${quality}">${escapeHtml(quality)}</span>
      </div>
      <div class="job-id">${escapeHtml(asset.geometry_id || "unknown")}</div>
      <div class="job-time">${escapeHtml(formatIsoTime(asset.created_at))}</div>
      <small>${escapeHtml(ext)} • ${sizeMb.toFixed(2)} MB</small>
    `;
    item.onclick = () => {
      state.selectedGeometryId = String(asset.geometry_id || "");
      if (geometrySelect) {
        geometrySelect.value = state.selectedGeometryId;
      }
      renderWorldKwEditor(worldSelect?.value || "");
      pushStudioEvent(`Selected geometry asset: ${asset.geometry_id}`, "selection", asset);
    };
    geometryList.appendChild(item);
  });
}

async function fetchGeometries() {
  const res = await fetch("/api/v1/geometries");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Unable to load geometries (${res.status}): ${text}`);
  }
  const payload = await res.json();
  state.geometryAssets = Array.isArray(payload.assets) ? payload.assets : [];
  if (state.selectedGeometryId) {
    const found = state.geometryAssets.some(
      (asset) => String(asset.geometry_id) === String(state.selectedGeometryId)
    );
    if (!found) {
      state.selectedGeometryId = "";
    }
  }
  renderGeometryAssets();
}

async function uploadGeometry(evt) {
  evt.preventDefault();
  if (!geometryUploadForm || !geometryUploadBtn || !geometryFileInput) {
    return;
  }
  const file = geometryFileInput.files && geometryFileInput.files[0];
  if (!file) {
    throw new Error("Select a geometry file first.");
  }
  geometryUploadBtn.disabled = true;
  try {
    const res = await fetch("/api/v1/geometries/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "x-atom-filename": file.name,
      },
      body: file,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Geometry upload failed (${res.status}): ${text}`);
    }
    const payload = await res.json();
    const asset = payload?.asset || {};
    state.selectedGeometryId = String(asset.geometry_id || "");
    geometryFileInput.value = "";
    await fetchGeometries();
    pushStudioEvent(`Uploaded geometry: ${asset.filename || asset.geometry_id}`, "success", asset);
  } finally {
    geometryUploadBtn.disabled = false;
  }
}

async function fetchInverseSpecTemplate() {
  const res = await fetch("/api/v1/inverse-design/spec-template");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Unable to load inverse contract template (${res.status}): ${text}`);
  }
  const payload = await res.json();
  state.inverseSpecTemplate = payload;
  const worldKey = normalizeWorldSpec(worldSelect?.value || "");
  if (worldKey) {
    renderWorldKwEditor(worldKey);
  }
  pushStudioEvent("Inverse contract template loaded", "info", {
    supported_worlds: Array.isArray(payload.supported_world_specs)
      ? payload.supported_world_specs.length
      : 0,
  });
}

async function fetchStudioCatalog() {
  const res = await fetch("/api/v1/studio/catalog");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Unable to load studio catalog (${res.status}): ${text}`);
  }
  const payload = await res.json();
  state.studioCatalog = payload;
  renderStudioSimulators(payload.simulators || []);
  renderStudioDemos(payload.demos || []);
  pushStudioEvent("Studio catalog refreshed", "info", {
    simulators: Array.isArray(payload.simulators) ? payload.simulators.length : 0,
    demos: Array.isArray(payload.demos) ? payload.demos.length : 0,
    challenge_available: payload.challenge_available,
  });
  updateWizardChecklist();
}

async function buildDirectorPack() {
  const payload = {
    tag: `ui_${state.intentFocus || "director"}`,
    run_reliability: true,
    run_supersonic_validation: true,
    run_release_evidence: true,
    allow_missing: true,
    reliability_steps: 20,
    reliability_seed: 123,
    supersonic_steps: 16,
    supersonic_nx: 64,
    supersonic_ny: 32,
    supersonic_seed: 123,
  };
  pushStudioEvent("Building director pack", "launch", payload);
  const res = await fetch("/api/v1/studio/director-pack", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Director pack failed (${res.status}): ${text}`);
  }
  const data = await res.json();
  pushStudioEvent("Director pack complete", data.ok ? "success" : "warn", data);
  if (jobReport) {
    jobReport.textContent = JSON.stringify(
      {
        director_pack: {
          ok: Boolean(data.ok),
          pack_id: data.pack_id,
          output_dir: data.output_dir,
          release_manifest: data.release_packet_manifest,
          operations: data.operations,
        },
      },
      null,
      2
    );
  }
}

async function fetchWorlds() {
  if (!worldSelect) {
    return;
  }
  const previous = normalizeWorldSpec(worldSelect.value);
  const res = await fetch("/api/v1/worlds");
  if (!res.ok) {
    throw new Error(`Unable to load worlds (${res.status})`);
  }
  const payload = await res.json();
  const worlds = payload.worlds || {};
  worldSelect.innerHTML = "";
  Object.entries(worlds).forEach(([name, desc]) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name} — ${desc}`;
    worldSelect.appendChild(option);
  });
  if (previous) {
    const option = Array.from(worldSelect.options).find(
      (node) => normalizeWorldSpec(node.value) === previous
    );
    if (option) {
      worldSelect.value = option.value;
    }
  }
  if (!worldSelect.value && worldSelect.options.length) {
    worldSelect.value = worldSelect.options[0].value;
  }
  renderWorldKwEditor(worldSelect.value);
  updateWizardChecklist();
}

function renderStatusMatrix() {
  if (!statusMatrix) {
    return;
  }
  const counts = { queued: 0, running: 0, succeeded: 0, failed: 0, cancelled: 0 };
  state.jobs.forEach((job) => {
    if (Object.prototype.hasOwnProperty.call(counts, job.status)) {
      counts[job.status] += 1;
    }
  });
  statusMatrix.innerHTML = `
    <span class="matrix-chip queued">queued: ${counts.queued}</span>
    <span class="matrix-chip running">running: ${counts.running}</span>
    <span class="matrix-chip succeeded">succeeded: ${counts.succeeded}</span>
    <span class="matrix-chip failed">failed: ${counts.failed}</span>
    <span class="matrix-chip cancelled">cancelled: ${counts.cancelled}</span>
  `;
}

function renderJobs() {
  if (!jobList) {
    return;
  }
  jobList.innerHTML = "";
  if (!state.jobs.length) {
    jobList.innerHTML = "<div class='job-item'>No jobs submitted yet.</div>";
    renderStatusMatrix();
    updateInverseKpis(null);
    return;
  }

  state.jobs.forEach((job) => {
    const status = normalizeStatus(job.status);
    const title = escapeHtml(job.request?.name || job.job_id);
    const world = escapeHtml(job.request?.world_spec || "unknown_world");
    const jobId = escapeHtml(job.job_id);
    const updatedAt = escapeHtml(formatIsoTime(job.updated_at));
    const el = document.createElement("div");
    el.className = `job-item ${state.selectedJobId === job.job_id ? "active" : ""}`;
    el.innerHTML = `
      <div class="job-head">
        <strong>${title}</strong>
        <span class="status ${status}">${status}</span>
      </div>
      <div class="job-id">${jobId}</div>
      <div class="job-time">${updatedAt}</div>
      <small>${world}</small>
    `;
    el.onclick = () => {
      state.selectedJobId = job.job_id;
      renderJobs();
      renderReport(job);
    };
    jobList.appendChild(el);
  });
  renderStatusMatrix();
  updateWizardChecklist();
  renderNarrativeFeed();
}

function renderReport(job) {
  if (jobReport) {
    jobReport.textContent = JSON.stringify(job, null, 2);
  }
  if (selectedJobLabel) {
    selectedJobLabel.textContent = `${job.status || "unknown"} :: ${job.job_id}`;
  }
  updateInverseKpis(job);
}

function renderSupersonicMissionFeed(telemetry) {
  const timeline = Array.isArray(telemetry?.timeline_tail) ? telemetry.timeline_tail : [];
  const incidents = Array.isArray(telemetry?.incidents_tail) ? telemetry.incidents_tail : [];

  if (supersonicTimeline) {
    if (!timeline.length) {
      supersonicTimeline.innerHTML = "<div class='job-item'>No timeline events yet.</div>";
    } else {
      supersonicTimeline.innerHTML = timeline
        .slice(-24)
        .reverse()
        .map((event) => {
          const ts = escapeHtml(formatIsoTime(event?.ts));
          const type = escapeHtml(event?.type || "event");
          const message = escapeHtml(event?.message || "");
          return `<div class='timeline-item'><strong>${type}</strong><span>${ts}</span><p>${message}</p></div>`;
        })
        .join("");
    }
  }

  if (supersonicIncidents) {
    if (!incidents.length) {
      supersonicIncidents.innerHTML = "<div class='job-item'>No incidents in current window.</div>";
    } else {
      supersonicIncidents.innerHTML = incidents
        .slice(-16)
        .reverse()
        .map((incident) => {
          const severity = String(incident?.severity || "info");
          const code = escapeHtml(incident?.code || "incident");
          const step = Number.isFinite(Number(incident?.step)) ? Number(incident.step) : "n/a";
          const message = escapeHtml(incident?.message || "");
          return `
            <div class='incident-item severity-${escapeHtml(severity)}'>
              <div class='job-head'>
                <strong>${code}</strong>
                <span class='status ${escapeHtml(severity)}'>${escapeHtml(severity)}</span>
              </div>
              <div class='job-time'>step=${step}</div>
              <p>${message}</p>
            </div>
          `;
        })
        .join("");
    }
  }
}

function updateSupersonicControlButtons(job, telemetry) {
  const running = String(job?.status || "") === "running";
  const paused = Boolean(telemetry?.control_state?.paused || job?.paused);
  if (supersonicPauseBtn) {
    supersonicPauseBtn.disabled = !running || paused;
  }
  if (supersonicResumeBtn) {
    supersonicResumeBtn.disabled = !running || !paused;
  }
  if (supersonicCancelBtn) {
    supersonicCancelBtn.disabled = !running;
  }
  if (supersonicBookmarkBtn) {
    supersonicBookmarkBtn.disabled = !job;
  }
}

function renderSupersonicTelemetry(job, telemetry) {
  if (!supersonicTelemetry) {
    return;
  }
  if (!job) {
    supersonicTelemetry.textContent = JSON.stringify({ status: "no_supersonic_job" }, null, 2);
    renderSupersonicMissionFeed(null);
    updateSupersonicControlButtons(null, null);
    return;
  }
  const payload = {
    job_id: job.job_id,
    status: job.status,
    latest_telemetry: telemetry?.latest_telemetry || job.latest_telemetry || null,
    summary: telemetry?.summary || job.result?.summary || null,
    derived: telemetry?.derived || null,
    law_trace: telemetry?.law_trace || [],
    timeline_tail: telemetry?.timeline_tail || [],
    incidents_tail: telemetry?.incidents_tail || [],
    bookmarks: telemetry?.bookmarks || [],
    control_state: telemetry?.control_state || {
      paused: Boolean(job?.paused),
      cancel_requested: Boolean(job?.cancel_requested),
    },
    updated_at: telemetry?.updated_at || job.updated_at,
  };
  supersonicTelemetry.textContent = JSON.stringify(payload, null, 2);
  renderSupersonicMissionFeed(telemetry);
  updateSupersonicControlButtons(job, telemetry);
  renderNarrativeFeed();
}

function renderSupersonicJobs() {
  if (!supersonicJobList) {
    return;
  }
  supersonicJobList.innerHTML = "";
  if (!state.supersonicJobs.length) {
    supersonicJobList.innerHTML = "<div class='job-item'>No challenge jobs yet.</div>";
    renderSupersonicTelemetry(null, null);
    renderSupersonicCharts(null);
    updateSupersonicKpis(null);
    updateWizardChecklist();
    renderNarrativeFeed();
    return;
  }
  state.supersonicJobs.forEach((job) => {
    const status = normalizeStatus(job.status);
    const title = escapeHtml(job.request?.name || job.job_id);
    const updatedAt = escapeHtml(formatIsoTime(job.updated_at));
    const el = document.createElement("div");
    el.className = `job-item ${state.selectedSupersonicJobId === job.job_id ? "active" : ""}`;
    el.innerHTML = `
      <div class="job-head">
        <strong>${title}</strong>
        <span class="status ${status}">${status}</span>
      </div>
      <div class="job-id">${escapeHtml(job.job_id)}</div>
      <div class="job-time">${updatedAt}</div>
      <small>steps: ${escapeHtml(job.request?.steps || "n/a")}</small>
    `;
    el.onclick = async () => {
      state.selectedSupersonicJobId = job.job_id;
      renderSupersonicJobs();
      await syncSelectedSupersonicTelemetry();
    };
    supersonicJobList.appendChild(el);
  });
  updateWizardChecklist();
}

async function fetchJobs() {
  const res = await fetch("/api/v1/inverse-design/jobs");
  if (!res.ok) {
    throw new Error(`Unable to load jobs (${res.status})`);
  }
  const payload = await res.json();
  state.jobs = payload.jobs || [];
  if (!state.selectedJobId && state.jobs.length) {
    state.selectedJobId = state.jobs[0].job_id;
  }
  renderJobs();

  const selected = state.jobs.find((j) => j.job_id === state.selectedJobId);
  if (selected) {
    renderReport(selected);
  } else if (selectedJobLabel) {
    selectedJobLabel.textContent = "no job selected";
    updateInverseKpis(null);
  }
}

async function fetchSupersonicJobs() {
  const res = await fetch("/api/v1/challenges/supersonic/jobs");
  if (!res.ok) {
    throw new Error(`Unable to load supersonic jobs (${res.status})`);
  }
  const payload = await res.json();
  state.supersonicJobs = payload.jobs || [];
  const hasSelected = state.supersonicJobs.some(
    (job) => String(job.job_id) === String(state.selectedSupersonicJobId)
  );
  if ((!state.selectedSupersonicJobId || !hasSelected) && state.supersonicJobs.length) {
    state.selectedSupersonicJobId = state.supersonicJobs[0].job_id;
  }
  renderSupersonicJobs();
  await syncSelectedSupersonicTelemetry();
}

async function fetchSupersonicTelemetry(jobId, limit = 480) {
  const path = `/api/v1/challenges/supersonic/jobs/${encodeURIComponent(jobId)}/telemetry?limit=${limit}`;
  const res = await fetch(path);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Telemetry fetch failed (${res.status}): ${text}`);
  }
  return res.json();
}

async function sendSupersonicControl(action, note = "") {
  const jobId = state.selectedSupersonicJobId;
  if (!jobId) {
    throw new Error("Select a supersonic job first.");
  }
  const res = await fetch(
    `/api/v1/challenges/supersonic/jobs/${encodeURIComponent(jobId)}/control`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        note: String(note || "").trim() || null,
      }),
    }
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Control request failed (${res.status}): ${text}`);
  }
  return res.json();
}

async function fetchRuntimeTelemetry() {
  const res = await fetch("/api/v1/runtime/telemetry");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Runtime telemetry fetch failed (${res.status}): ${text}`);
  }
  state.runtimeTelemetry = await res.json();
  renderRuntimeTelemetry(state.runtimeTelemetry);
  renderNarrativeFeed();
}

async function pollData() {
  fetchJobs().catch((err) => {
    if (jobReport) {
      jobReport.textContent = JSON.stringify({ error: String(err) }, null, 2);
    }
  });
  fetchSupersonicJobs().catch((err) => {
    if (supersonicTelemetry) {
      supersonicTelemetry.textContent = JSON.stringify({ error: String(err) }, null, 2);
    }
  });
  fetchRuntimeTelemetry().catch((err) => {
    if (runtimeTelemetry) {
      runtimeTelemetry.textContent = JSON.stringify({ error: String(err) }, null, 2);
    }
    if (runtimeDiagnosticsStatus) {
      runtimeDiagnosticsStatus.textContent = `runtime diagnostics: error (${String(err)})`;
    }
  });
}

function restartPollingLoop() {
  if (state.pollingTimer !== null) {
    window.clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
  const interval = Number(state.uiSettings.refreshMs);
  state.pollingTimer = window.setInterval(pollData, interval);
}

async function syncSelectedSupersonicTelemetry() {
  const selected = state.supersonicJobs.find((j) => j.job_id === state.selectedSupersonicJobId);
  if (!selected) {
    state.supersonicTelemetry = null;
    renderSupersonicTelemetry(null, null);
    renderSupersonicCharts(null);
    updateSupersonicKpis(null);
    return;
  }
  try {
    const telemetry = await fetchSupersonicTelemetry(selected.job_id);
    state.supersonicTelemetry = telemetry;
    renderSupersonicTelemetry(selected, telemetry);
    renderSupersonicCharts(telemetry);
    updateSupersonicKpis(telemetry);
  } catch (err) {
    state.supersonicTelemetry = null;
    renderSupersonicTelemetry(selected, null);
    renderSupersonicCharts(null);
    updateSupersonicKpis(null);
    if (supersonicTelemetry) {
      supersonicTelemetry.textContent = JSON.stringify(
        { error: String(err?.message || err) },
        null,
        2
      );
    }
  }
}

async function submitJob(evt) {
  evt.preventDefault();
  if (!submitBtn || !form) {
    return;
  }
  submitBtn.disabled = true;
  try {
    const fd = new FormData(form);
    const geometryId = String(fd.get("geometry_id") || state.selectedGeometryId || "").trim();
    const worldKw = collectWorldKwPayload(Boolean(geometryId));
    const payload = {
      name: fd.get("name"),
      world_spec: fd.get("world_spec"),
      backend: fd.get("backend"),
      iterations: Number.parseInt(fd.get("iterations"), 10),
      population: Number.parseInt(fd.get("population"), 10),
      rollout_steps: Number.parseInt(fd.get("rollout_steps"), 10),
      top_k: Number.parseInt(fd.get("top_k"), 10),
      grid_shape: parseGridShape(fd.get("grid_shape")),
      world_kwargs: worldKw,
      device: "cpu",
    };
    if (geometryId) {
      payload.geometry_id = geometryId;
      state.selectedGeometryId = geometryId;
    }
    const res = await fetch("/api/v1/inverse-design/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Request failed (${res.status}): ${text}`);
    }
    const data = await res.json();
    state.selectedJobId = data.job_id;
    await fetchJobs();
  } catch (err) {
    if (jobReport) {
      jobReport.textContent = JSON.stringify({ error: String(err?.message || err) }, null, 2);
    }
  } finally {
    submitBtn.disabled = false;
  }
}

async function submitSupersonicJob(evt) {
  evt.preventDefault();
  if (!supersonicForm || !supersonicSubmitBtn) {
    return;
  }
  supersonicSubmitBtn.disabled = true;
  try {
    const fd = new FormData(supersonicForm);
    const payload = {
      name: fd.get("name"),
      steps: Number.parseInt(fd.get("steps"), 10),
      headless: true,
    };
    const res = await fetch("/api/v1/challenges/supersonic/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Request failed (${res.status}): ${text}`);
    }
    const data = await res.json();
    state.selectedSupersonicJobId = data.job_id;
    await fetchSupersonicJobs();
  } catch (err) {
    if (supersonicTelemetry) {
      supersonicTelemetry.textContent = JSON.stringify(
        { error: String(err?.message || err) },
        null,
        2
      );
    }
  } finally {
    supersonicSubmitBtn.disabled = false;
  }
}

async function triggerSupersonicControl(action) {
  const note = String(supersonicBookmarkNote?.value || "").trim();
  const buttons = [supersonicPauseBtn, supersonicResumeBtn, supersonicCancelBtn, supersonicBookmarkBtn];
  buttons.forEach((btn) => {
    if (btn) {
      btn.disabled = true;
    }
  });
  try {
    const response = await sendSupersonicControl(action, note);
    if (action === "bookmark" && supersonicBookmarkNote) {
      supersonicBookmarkNote.value = "";
    }
    pushStudioEvent(`Supersonic control action: ${action}`, "control", response);
    await fetchSupersonicJobs();
  } catch (err) {
    const message = String(err?.message || err);
    pushStudioEvent(`Supersonic control failed: ${action}`, "error", message);
    if (supersonicTelemetry) {
      supersonicTelemetry.textContent = JSON.stringify({ error: message }, null, 2);
    }
  } finally {
    try {
      await syncSelectedSupersonicTelemetry();
    } catch (err) {
      pushStudioEvent("Supersonic telemetry refresh failed", "error", String(err?.message || err));
    }
  }
}

async function runCopilotBrief() {
  const payload = {
    question: "Provide an operator brief with state, risk posture, and next actions.",
    mode: "next_actions",
    intent: state.intentFocus || "control",
    diagnostic_mode: state.diagnosticMode || "overview",
    engine: state.assistantEngine || "deterministic",
    inverse_job_id: state.selectedJobId || null,
    supersonic_job_id: state.selectedSupersonicJobId || null,
    telemetry_window: 96,
  };
  const res = await fetch("/api/v1/assistant/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Copilot brief failed (${res.status}): ${text}`);
  }
  const data = await res.json();
  if (assistantResponse) {
    assistantResponse.textContent = JSON.stringify(data, null, 2);
  }
  state.wizardAssistantCompleted = true;
  updateWizardChecklist();
  return data;
}

async function wizardLoadInverseTemplate() {
  const demo = findDemoByKind("inverse_design");
  if (!demo) {
    throw new Error("No inverse-design template available in catalog.");
  }
  applyDemoTemplate(demo);
}

async function wizardLaunchInverse() {
  const demo = findDemoByKind("inverse_design");
  if (demo && demo.available) {
    await launchStudioDemo(demo);
    return;
  }
  if (form && typeof form.requestSubmit === "function") {
    form.requestSubmit();
    return;
  }
  throw new Error("Unable to launch inverse flow from current UI state.");
}

async function wizardLaunchSupersonic() {
  const demo = findDemoByKind("supersonic_challenge");
  if (demo && demo.available) {
    await launchStudioDemo(demo);
    return;
  }
  if (supersonicForm && typeof supersonicForm.requestSubmit === "function") {
    supersonicForm.requestSubmit();
    return;
  }
  throw new Error("Unable to launch supersonic flow from current UI state.");
}

async function startGuidedLaunch() {
  if (!onboardingStartBtn) {
    return;
  }
  onboardingStartBtn.disabled = true;
  try {
    await wizardLoadInverseTemplate();
    pushStudioEvent("Guided launch: inverse template loaded", "guide");
    await wizardLaunchInverse();
    pushStudioEvent("Guided launch: inverse run launched", "guide");
    await wizardLaunchSupersonic();
    pushStudioEvent("Guided launch: supersonic run launched", "guide");
    await runCopilotBrief();
    pushStudioEvent("Guided launch: copilot brief generated", "guide");
    state.onboardingSeen = true;
    persistOnboardingState();
    renderOnboardingPanel();
    renderNarrativeFeed();
  } finally {
    onboardingStartBtn.disabled = false;
  }
}

async function submitAssistantQuery(evt) {
  evt.preventDefault();
  if (!assistantForm || !assistantSubmitBtn) {
    return;
  }
  assistantSubmitBtn.disabled = true;
  try {
    const fd = new FormData(assistantForm);
    const payload = {
      question: String(fd.get("question") || "").trim(),
      mode: fd.get("mode"),
      intent: String(fd.get("intent") || state.intentFocus || "scientific_discovery"),
      diagnostic_mode: String(fd.get("diagnostic_mode") || state.diagnosticMode || "overview"),
      engine: String(fd.get("engine") || state.assistantEngine || "deterministic"),
      inverse_job_id: state.selectedJobId,
      supersonic_job_id: state.selectedSupersonicJobId,
      telemetry_window: 96,
    };
    const res = await fetch("/api/v1/assistant/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Assistant request failed (${res.status}): ${text}`);
    }
    const data = await res.json();
    if (assistantResponse) {
      assistantResponse.textContent = JSON.stringify(data, null, 2);
    }
    state.wizardAssistantCompleted = true;
    updateWizardChecklist();
  } catch (err) {
    if (assistantResponse) {
      assistantResponse.textContent = JSON.stringify(
        { error: String(err?.message || err) },
        null,
        2
      );
    }
  } finally {
    assistantSubmitBtn.disabled = false;
  }
}

if (form) {
  form.addEventListener("submit", submitJob);
}
if (geometryUploadForm) {
  geometryUploadForm.addEventListener("submit", (evt) => {
    uploadGeometry(evt).catch((err) => {
      pushStudioEvent("Geometry upload failed", "error", String(err?.message || err));
    });
  });
}
if (geometrySelect) {
  geometrySelect.addEventListener("change", (evt) => {
    state.selectedGeometryId = String(evt.target.value || "");
    renderWorldKwEditor(worldSelect?.value || "");
    updateWizardChecklist();
  });
}
if (worldSelect) {
  worldSelect.addEventListener("change", () => {
    renderWorldKwEditor(worldSelect.value);
    updateWizardChecklist();
  });
}
if (supersonicForm) {
  supersonicForm.addEventListener("submit", submitSupersonicJob);
}
if (supersonicPauseBtn) {
  supersonicPauseBtn.addEventListener("click", () => {
    triggerSupersonicControl("pause");
  });
}
if (supersonicResumeBtn) {
  supersonicResumeBtn.addEventListener("click", () => {
    triggerSupersonicControl("resume");
  });
}
if (supersonicCancelBtn) {
  supersonicCancelBtn.addEventListener("click", () => {
    triggerSupersonicControl("cancel");
  });
}
if (supersonicBookmarkBtn) {
  supersonicBookmarkBtn.addEventListener("click", () => {
    triggerSupersonicControl("bookmark");
  });
}
if (assistantForm) {
  assistantForm.addEventListener("submit", submitAssistantQuery);
}
if (refreshStudioBtn) {
  refreshStudioBtn.addEventListener("click", () => {
    fetchStudioCatalog().catch((err) => {
      pushStudioEvent(
        "Studio catalog refresh failed",
        "error",
        String(err?.message || err)
      );
    });
  });
}
if (buildDirectorPackBtn) {
  buildDirectorPackBtn.addEventListener("click", () => {
    buildDirectorPack().catch((err) => {
      pushStudioEvent(
        "Director pack build failed",
        "error",
        String(err?.message || err)
      );
    });
  });
}
missionIntentButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const intent = String(btn.dataset.intent || "").trim();
    if (!intent) {
      return;
    }
    syncIntentFocus(intent, "mission", true);
    renderNarrativeFeed();
    pushStudioEvent("Mission intent updated", "intent", {
      intent,
      diagnostic_mode: state.diagnosticMode,
    });
  });
});
if (intentFocusSelect) {
  intentFocusSelect.addEventListener("change", (evt) => {
    syncIntentFocus(evt.target.value, "observability");
  });
}
if (diagnosticModeSelect) {
  diagnosticModeSelect.addEventListener("change", (evt) => {
    syncDiagnosticMode(evt.target.value, "observability");
  });
}
if (assistantIntentSelect) {
  assistantIntentSelect.addEventListener("change", (evt) => {
    syncIntentFocus(evt.target.value, "assistant");
  });
}
if (assistantDiagnosticModeSelect) {
  assistantDiagnosticModeSelect.addEventListener("change", (evt) => {
    syncDiagnosticMode(evt.target.value, "assistant");
  });
}
if (assistantEngineSelect) {
  assistantEngineSelect.addEventListener("change", (evt) => {
    syncAssistantEngine(evt.target.value);
  });
}
if (refreshIntervalSelect) {
  refreshIntervalSelect.addEventListener("change", (evt) => {
    const value = Number(evt.target.value);
    if (Number.isFinite(value) && value >= 800 && value <= 30000) {
      state.uiSettings.refreshMs = value;
      persistUiSettings();
      restartPollingLoop();
      pushStudioEvent("Polling interval updated", "settings", { refreshMs: value });
    }
  });
}
if (densitySelect) {
  densitySelect.addEventListener("change", (evt) => {
    const value = String(evt.target.value || "comfortable");
    state.uiSettings.density = value === "compact" ? "compact" : "comfortable";
    applyUiSettings();
    persistUiSettings();
  });
}
if (showRawPanelsToggle) {
  showRawPanelsToggle.addEventListener("change", (evt) => {
    state.uiSettings.showRawPanels = Boolean(evt.target.checked);
    applyUiSettings();
    persistUiSettings();
  });
}
if (motionToggle) {
  motionToggle.addEventListener("change", (evt) => {
    state.uiSettings.motion = Boolean(evt.target.checked);
    applyUiSettings();
    persistUiSettings();
  });
}
if (obstacleOverlayToggle) {
  obstacleOverlayToggle.addEventListener("change", (evt) => {
    state.uiSettings.showObstacleOverlay = Boolean(evt.target.checked);
    applyUiSettings();
    persistUiSettings();
    if (state.runtimeTelemetry) {
      renderRuntimeTelemetry(state.runtimeTelemetry);
    }
  });
}
if (saliencyFocusToggle) {
  saliencyFocusToggle.addEventListener("change", (evt) => {
    state.uiSettings.showSaliencyFocus = Boolean(evt.target.checked);
    applyUiSettings();
    persistUiSettings();
    if (state.runtimeTelemetry) {
      renderRuntimeTelemetry(state.runtimeTelemetry);
    }
  });
}
if (showOnboardingBtn) {
  showOnboardingBtn.addEventListener("click", () => {
    state.onboardingSeen = false;
    persistOnboardingState();
    renderOnboardingPanel();
  });
}
if (onboardingDismissBtn) {
  onboardingDismissBtn.addEventListener("click", () => {
    state.onboardingSeen = true;
    persistOnboardingState();
    renderOnboardingPanel();
  });
}
if (onboardingStartBtn) {
  onboardingStartBtn.addEventListener("click", () => {
    startGuidedLaunch().catch((err) => {
      pushStudioEvent("Guided launch failed", "error", String(err?.message || err));
    });
  });
}
if (wizardLoadInverseBtn) {
  wizardLoadInverseBtn.addEventListener("click", () => {
    wizardLoadInverseTemplate().catch((err) => {
      pushStudioEvent("Load inverse template failed", "error", String(err?.message || err));
    });
  });
}
if (wizardLaunchInverseBtn) {
  wizardLaunchInverseBtn.addEventListener("click", () => {
    wizardLaunchInverse().catch((err) => {
      pushStudioEvent("Launch inverse failed", "error", String(err?.message || err));
    });
  });
}
if (wizardLaunchSupersonicBtn) {
  wizardLaunchSupersonicBtn.addEventListener("click", () => {
    wizardLaunchSupersonic().catch((err) => {
      pushStudioEvent("Launch supersonic failed", "error", String(err?.message || err));
    });
  });
}
if (wizardRunCopilotBtn) {
  wizardRunCopilotBtn.addEventListener("click", () => {
    runCopilotBrief().catch((err) => {
      pushStudioEvent("Copilot brief failed", "error", String(err?.message || err));
    });
  });
}
if (narrativeModeSelect) {
  narrativeModeSelect.addEventListener("change", (evt) => {
    const mode = String(evt.target.value || "executive");
    state.uiSettings.narrativeMode = ["off", "executive", "technical"].includes(mode)
      ? mode
      : "executive";
    persistUiSettings();
    renderNarrativeFeed();
  });
}

async function bootstrap() {
  loadUiSettings();
  loadOnboardingState();
  applyUiSettings();
  renderOnboardingPanel();
  syncIntentFocus(state.intentFocus);
  syncDiagnosticMode(state.diagnosticMode);
  syncAssistantEngine(state.assistantEngine);
  verifyVisualizationContainers();
  await fetchStudioCatalog();
  await fetchInverseSpecTemplate();
  await fetchGeometries();
  await fetchWorlds();
  await fetchJobs();
  await fetchSupersonicJobs();
  await fetchRuntimeTelemetry();
  renderSupersonicCharts(state.supersonicTelemetry);
  updateWizardChecklist();
  renderNarrativeFeed();
  restartPollingLoop();
}

bootstrap().catch((err) => {
  if (jobReport) {
    jobReport.textContent = JSON.stringify({ error: String(err) }, null, 2);
  }
});
