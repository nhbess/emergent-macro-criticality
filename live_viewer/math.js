// Shared math helpers for the live firefly / reservoir simulation

function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  // Sampled from matplotlib inferno LUT (t=0 → near-black, t=1 → pale yellow)
  const stops = [
    [0,   0,   4  ],  // 0.000
    [20,  11,  52 ],  // 0.125
    [58,  15,  97 ],  // 0.250
    [106, 15,  110],  // 0.375
    [153, 37,  91 ],  // 0.500
    [195, 71,  61 ],  // 0.625
    [231, 116, 36 ],  // 0.750
    [249, 169, 10 ],  // 0.875
    [252, 255, 164],  // 1.000
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const f = seg - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
}

// MLE power-law exponent (discrete, Clauset et al. 2009)
// α̂ = 1 + n / Σ ln(x_i / (x_min − 0.5)),  x_min = 2 (excludes single-agent events)
function computeAlpha(data, cnt, xMin = 2) {
  let sumLog = 0, n = 0;
  for (let k = 0; k < cnt; k++) {
    if (data[k] >= xMin) { sumLog += Math.log(data[k] / (xMin - 0.5)); n++; }
  }
  return (n >= 10 && sumLog > 0) ? (1 + n / sumLog) : null;
}

// Avalanche math
const AV_NBINS = 22;
const AV_MIN   = 30;

function logHist(data, cnt) {
  let minV = Infinity, maxV = 0;
  for (let k = 0; k < cnt; k++) { if (data[k] < minV) minV = data[k]; if (data[k] > maxV) maxV = data[k]; }
  if (minV <= 0 || maxV <= minV) return null;
  const logMin = Math.log10(minV), logMax = Math.log10(maxV);
  if (logMax - logMin < 0.1) return null;
  const dlog = (logMax - logMin) / AV_NBINS;
  const counts = new Array(AV_NBINS).fill(0);
  for (let k = 0; k < cnt; k++) {
    const idx = Math.min(Math.floor((Math.log10(data[k]) - logMin) / dlog), AV_NBINS - 1);
    counts[idx]++;
  }
  const centers = [], logProbs = [];
  for (let k = 0; k < AV_NBINS; k++) {
    if (counts[k] < 2) continue;
    const binW = Math.pow(10, logMin + (k + 1) * dlog) - Math.pow(10, logMin + k * dlog);
    centers.push(logMin + (k + 0.5) * dlog);
    logProbs.push(Math.log10(counts[k] / (cnt * binW)));
  }
  return centers.length >= 3 ? { centers, logProbs, logMin, logMax } : null;
}

// ── Core reservoir helpers (shared micro-level rule) ────────────────────────

function countActiveRange(state, start, len) {
  let nAct = 0;
  const end = start + len;
  for (let i = start; i < end; i++) if (state[i]) nAct++;
  return nAct;
}

function reservoirStepRange(current, next, start, len, p, powTable = null) {
  const nAct = countActiveRange(current, start, len);
  const end = start + len;
  if (nAct === 0) {
    for (let j = start; j < end; j++) next[j] = 0;
    return 0;
  }
  for (let j = start; j < end; j++) {
    const src = current[j] ? nAct - 1 : nAct;
    const pj = powTable ? (1.0 - powTable[src]) : (1.0 - Math.pow(1.0 - p, src));
    next[j] = (pj > 0 && Math.random() < pj) ? 1 : 0;
  }
  return nAct;
}

// ── Simulation-level helpers (agents + global reservoir arrays) ─────────────

function curVD() { return VD_VALUES[VI]; }
function curP()  { return P_VALUES[PI];  }

function buildPowers() {
  pow1mP[0] = 1.0;
  const q = 1.0 - curP();
  for (let k = 1; k <= N_NEURONS; k++) pow1mP[k] = pow1mP[k-1] * q;
}

function resetAvalanches() {
  avCount = 0; avActive = false; avSz = 0; avDur = 0; repStep = 0; repCount = 0;
  simStartTime = null; darkSteps = 0;
}

// Soft-reinit: fresh random positions + reservoir reset; KEEPS avalanche statistics.
// Called automatically every STEPS_PER_REP steps or when a supercritical cascade is detected.
function softReinit() {
  for (let i = 0; i < N_AGENTS; i++) {
    pos[2*i]   = Math.random() * MAP_W;
    pos[2*i+1] = Math.random() * MAP_H;
    ang[i]     = Math.random() * 2 * Math.PI;
  }
  lights.fill(0); lights[0] = 1;
  res.fill(0); nxt.fill(0); obs.fill(0);
  prevOn = 1; sigmaLive = NaN; warmupCnt = 0;
  avActive = false; avSz = 0; avDur = 0;   // end any open avalanche without recording
  repStep = 0; repCount++;
  repFlash = 6;     // brief border flash to signal new adjacency
  buildPowers(); rebuildAdj(); computeObs();
}

function init() {
  for (let i = 0; i < N_AGENTS; i++) {
    pos[2*i]   = Math.random() * MAP_W;
    pos[2*i+1] = Math.random() * MAP_H;
    ang[i]     = Math.random() * 2 * Math.PI;
  }
  lights.fill(0); lights[0] = 1;
  res.fill(0);    nxt.fill(0);    obs.fill(0);
  macT = 0;  prevOn = 1;  sigmaLive = NaN;  warmupCnt = 0;
  resetAvalanches();
  buildPowers();
  rebuildAdj();
  computeObs();
}

function rebuildAdj() {
  const vd = curVD(), vd2 = vd * vd, hw = MAP_W / 2, hh = MAP_H / 2;
  const maxE = (N_AGENTS * (N_AGENTS - 1)) >> 1;
  const tIJ  = new Int16Array(maxE * 2);
  const tDXY = new Float32Array(maxE * 2);
  let ei = 0;
  for (let i = 0; i < N_AGENTS; i++) {
    const xi = pos[2*i], yi = pos[2*i+1];
    for (let j = i + 1; j < N_AGENTS; j++) {
      let dx = pos[2*j] - xi, dy = pos[2*j+1] - yi;
      if      (dx >  hw) dx -= MAP_W; else if (dx < -hw) dx += MAP_W;
      if      (dy >  hh) dy -= MAP_H; else if (dy < -hh) dy += MAP_H;
      if (dx*dx + dy*dy <= vd2) {
        tIJ[2*ei] = i; tIJ[2*ei+1] = j;
        tDXY[2*ei] = dx; tDXY[2*ei+1] = dy;
        ei++;
      }
    }
  }
  NE = ei;
  adjIJ  = tIJ.slice(0, ei * 2);
  adjDXY = tDXY.slice(0, ei * 2);
}

function computeObs() {
  obs.fill(0);
  const TWO_PI = 2 * Math.PI, BW = TWO_PI / KE;
  for (let e = 0; e < NE; e++) {
    const i = adjIJ[2*e], j = adjIJ[2*e+1];
    const dx = adjDXY[2*e], dy = adjDXY[2*e+1];
    if (lights[j]) {
      let phi = Math.atan2(dy, dx) - ang[i];
      obs[i * KE + ((((phi % TWO_PI) + TWO_PI) % TWO_PI) / BW | 0)] = 1;
    }
    if (lights[i]) {
      let phi = Math.atan2(-dy, -dx) - ang[j];
      obs[j * KE + ((((phi % TWO_PI) + TWO_PI) % TWO_PI) / BW | 0)] = 1;
    }
  }
}

// Beggs reservoir step  (aggregated O(N·NN) formulation)
// P(j activated) = 1 − (1−p)^{n_active_sources_for_j}
function stepRes() {
  nxt.fill(0);
  const p = curP();
  for (let a = 0; a < N_AGENTS; a++) {
    const base = a * N_NEURONS;
    reservoirStepRange(res, nxt, base, N_NEURONS, p, pow1mP);
  }
}

// Macro step  (mirrors _run_environment_p)
function macroStep() {
  if (simStartTime === null) simStartTime = performance.now();
  computeObs();
  for (let m = 0; m < MICRO; m++) {
    stepRes();
    if (m === 0) {
      for (let a = 0; a < N_AGENTS; a++) {
        const rb = a * N_NEURONS, ob = a * KE;
        for (let k = 0; k < KE; k++) nxt[rb + k] = obs[ob + k];
      }
    }
    res.set(nxt);
  }
  let nOn = 0;
  for (let a = 0; a < N_AGENTS; a++) {
    lights[a] = res[a * N_NEURONS + N_NEURONS - 1];
    if (lights[a]) nOn++;
  }
  if (prevOn > 0 && warmupCnt >= WARMUP) {
    const s = nOn / prevOn;
    sigmaLive = isNaN(sigmaLive) ? s : sigmaLive * (1 - SALPHA) + s * SALPHA;
  }
  // Track avalanche (only after warmup; end before reseeding)
  if (warmupCnt >= WARMUP) {
    if (nOn > 0) {
      if (!avActive) { avActive = true; avSz = 0; avDur = 0; }
      avSz += nOn; avDur++;
    } else {
      if (avActive) {
        if (avCount < AV_CAP) { avSizes[avCount] = avSz; avDurs[avCount] = avDur; avCount++; }
        avActive = false;
      }
    }
  }
  prevOn = nOn; macT++;
  if (warmupCnt < WARMUP) warmupCnt++;
  if (nOn === 0) {
    darkSteps++;
    if (darkSteps >= 3) {
      let anyRes = false;
      for (let k = 0, len = res.length; k < len; k++) { if (res[k]) { anyRes = true; break; } }
      if (!anyRes) {
        darkSteps = 0;
        const a = Math.floor(Math.random() * N_AGENTS);
        lights[a] = 1; prevOn = 1;
      }
    }
  } else {
    darkSteps = 0;
  }
  // Multi-realization cycling: open cascades at window end are censored
  repStep++;
  if (repStep >= STEPS_PER_REP) softReinit();
}

// ── Reservoir demo math (About modal raster) ────────────────────────────────

function rdP() { return rdSigma / Math.max(rdN - 1, 1); }

function rdStep() {
  if (countActiveRange(rdState, 0, rdN) === 0) rdState[Math.floor(Math.random() * rdN)] = 1;
  const nxt = new Uint8Array(rdN);
  reservoirStepRange(rdState, nxt, 0, rdN, rdP());
  rdState = nxt;
  rdHist.push(rdState.slice());
  if (rdHist.length > DEMO_T) rdHist.shift();
}

// ── Single-agent micro/macro demo math (64-node reservoir) ─────────────────-

function mdStepMicro() {
  reservoirStepRange(mdRes, mdNxt, 0, MD_N, mdP);
  for (let k = 0; k < mdNSens; k++) {
    if (mdInput[k]) mdNxt[k] = 1;
  }
  mdRes.set(mdNxt);
  mdInput.fill(0);
  mdMicroStep++;
}

function mdStepMacro() {
  const MICRO_LOCAL = 5;
  for (let i = 0; i < MICRO_LOCAL; i++) mdStepMicro();
  mdLight = mdRes[MD_N - 1];
  mdMacroStep++;
}

function mdComputeLayout(W, H) {
  mdSensorPos = [];
  mdGridPos = [];
  const cx = W * 0.5, cy = H * 0.50;
  const R = Math.min(W, H) * 0.30;
  const sensR = R + 22;
  for (let k = 0; k < mdNSens; k++) {
    const a = (k / mdNSens) * Math.PI * 2 - Math.PI / 2;
    mdSensorPos.push([cx + Math.cos(a) * sensR, cy + Math.sin(a) * sensR]);
  }
  const golden = Math.PI * (3 - Math.sqrt(5));
  const innerR = R * 0.35, outerR = R * 0.95;
  for (let i = 0; i < MD_N; i++) {
    if (i === MD_N - 1) { mdGridPos.push([cx, cy]); continue; }
    const t = i / (MD_N - 1);
    const radius = innerR + (outerR - innerR) * Math.sqrt(t);
    const angle = i * golden;
    mdGridPos.push([cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius]);
  }
  return { cx, cy, R, sensR };
}


