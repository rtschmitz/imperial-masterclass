import { clientAPI } from './client-api.js';

const q = (id) => document.getElementById(id);
const FRONTEND_VERSION = '4.0.0';
const api = {
  formulaEvent: body => clientAPI.formulaEvent(body),
  formulaBatch: body => clientAPI.formulaBatch(body)
};

const DEFAULT_FORMULA = 'E';

const state = {
  meta: null,
  mode: 'pairs',
  currentEvent: null,
  currentValues: [],
  lastEventIndex: null,
  formulaTouched: false,
  histValues: [],
  processedEvents: 0,
  acceptedEvents: 0,
  requirements: []
};

const presetGroups = [
  {label: 'Momentum', items: [
    {name: 'momentum', formula: 'sqrt(px^2+py^2+pz^2)'},
    {name: 'pT balance', formula: 'pt/pt_scalar'},
    {name: 'pT cancellation', formula: 'pt_scalar-pt'}
  ]},
  {label: 'Mass', items: [
    {name: 'mass', formula: 'sqrt(E^2-px^2-py^2-pz^2)'},
    {name: 'mass from E and p', formula: 'sqrt(E^2-p^2)'}
  ]},
  {label: 'Angles', items: [
    {name: 'angular separation', formula: 'dr'},
    {name: 'opening angle', formula: 'angle'}
  ]},
  {label: 'Other', items: [
    {name: 'charge magnitude', formula: 'abs(charge)'},
    {name: 'energy − momentum', formula: 'E-p'},
    {name: 'energy ÷ momentum', formula: 'E/p'},
    {name: 'energy − scalar pT', formula: 'E-pt_scalar'},
    {name: 'ΔR-weighted pT', formula: 'pt*dr'},
    {name: 'angle-weighted pT', formula: 'pt*angle'}
  ]}
];

const quantityDefinitions = [
  {symbol: 'E', label: 'Energy', short: 'Total energy carried by the selected particles.'},
  {symbol: 'p', label: 'Momentum', short: 'Magnitude of the combined three-dimensional momentum.'},
  {symbol: 'pt', label: 'Transverse momentum', short: 'Combined momentum perpendicular to the beam.'},
  {symbol: 'px', label: 'x momentum', short: 'The x component of the combined momentum.'},
  {symbol: 'py', label: 'y momentum', short: 'The y component of the combined momentum.'},
  {symbol: 'pz', label: 'z momentum', short: 'The component along the beam pipe.'},
  {symbol: 'pt_scalar', displaySymbol: 'ΣpT', label: 'scalar pT', short: 'The scalar sum of individual lepton pT values, without directional cancellation.'},
  {symbol: 'charge', displaySymbol: 'Q', label: 'Charge', short: 'Total electric charge of the selected particles.'},
  {symbol: 'dr', label: 'Separation ΔR', short: 'Angular distance for a pair; the four-lepton view uses the mean of all six pairwise distances.'},
  {symbol: 'angle', displaySymbol: 'θ', label: 'Opening angle', short: 'Three-dimensional angle for a pair; the four-lepton view uses the mean of all six pairwise angles.'}
];

const pairingHelp = {
  closest_distance: 'Selects the pairing with the smallest total angular separation.',
  furthest_distance: 'Selects the pairing with the largest total angular separation.',
  closest_formula: 'Selects the pairing whose two formula values are most alike.',
  furthest_formula: 'Selects the pairing whose two formula values differ the most.',
  same_type_same_charge: 'Prefers pairs with the same lepton type and the same charge.',
  same_type_different_charge: 'Prefers pairs with the same lepton type and different charges.',
  different_type_same_charge: 'Prefers electron–muon pairs whose leptons have the same charge.',
  different_type_different_charge: 'Prefers electron–muon pairs whose leptons have different charges.',
  random: 'Chooses one of the possible pairings at random for each event.'
};

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function numberFmt(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  const value = Number(x);
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1000) return value.toExponential(2);
  return value.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

async function postJSON(method, body) {
  return method(body);
}

let toastTimer = null;
function showToast(message) {
  const toast = q('toast');
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
}

function showError(error) {
  console.error(error);
  q('statusBox').textContent = `v${FRONTEND_VERSION} · error`;
  showToast(error.message || String(error));
}

function currentFormula() {
  return q('formulaInput').value.trim();
}

function normalizedFormula(formula) {
  return String(formula || '').replace(/\s+/g, '');
}

function requestBody(extra = {}) {
  return {
    mode: state.mode,
    formula: currentFormula() || DEFAULT_FORMULA,
    pairing_rule: q('pairingRule').value,
    channel: q('channelSelect').value,
    requirements: state.requirements.map(item => ({id: item.id, value: item.value})),
    ...extra
  };
}

function flavor(absPdg) {
  return Number(absPdg) === 11 ? 'e' : Number(absPdg) === 13 ? 'μ' : '?';
}

function chargeString(value) {
  return Number(value) > 0 ? '+' : Number(value) < 0 ? '−' : '0';
}

function formulaUses(symbol) {
  const formula = currentFormula();
  return new RegExp(`(^|[^A-Za-z0-9_])${symbol}([^A-Za-z0-9_]|$)`).test(formula);
}

function availableQuantities() {
  return quantityDefinitions;
}

function addVariableToFormula(symbol) {
  const input = q('formulaInput');
  const formula = input.value.trim();
  const untouchedDefault = !state.formulaTouched && formula === DEFAULT_FORMULA;
  if (!formula || untouchedDefault) input.value = symbol;
  else if (formulaUses(symbol)) input.value = formula;
  else if (/[+\-*/^(]$/.test(formula)) input.value = `${formula}${symbol}`;
  else input.value = `${formula}+${symbol}`;
  state.formulaTouched = true;
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

function renderQuantityPalette() {
  const box = q('quantityPalette');
  box.innerHTML = availableQuantities().map(item => `
    <button class="quantity-token ${formulaUses(item.symbol) ? 'active' : ''}" data-symbol="${item.symbol}" title="${escapeHTML(item.short)}">
      <span class="quantity-symbol">${item.displaySymbol || item.symbol}</span>
      <span class="quantity-name">${escapeHTML(item.label)}</span>
    </button>
  `).join('');
  box.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
      addVariableToFormula(button.dataset.symbol);
      renderQuantityPalette();
      renderQuantityExplanation();
      renderPresets();
      commitConfiguration();
    });
  });
}

function renderQuantityExplanation() {
  const active = availableQuantities().filter(item => formulaUses(item.symbol));
  const shown = active.length ? active : [availableQuantities()[0]];
  q('quantityExplanation').innerHTML = shown.map(item => `
    <div class="variable-explanation">
      <b>${escapeHTML(item.displaySymbol || item.symbol)} · ${escapeHTML(item.label)}</b>
      <span>${escapeHTML(item.short)}</span>
    </div>
  `).join('');
}

function renderPresets() {
  const selected = normalizedFormula(currentFormula());
  q('presetGrid').innerHTML = presetGroups.map(group => `
    <section class="preset-group" aria-label="${escapeHTML(group.label)} combinations">
      <div class="preset-group-label">${escapeHTML(group.label)}</div>
      <div class="preset-options">
        ${group.items.map(item => {
          const active = selected === normalizedFormula(item.formula);
          return `<button class="preset${active ? ' active' : ''}" data-formula="${escapeHTML(item.formula)}" aria-pressed="${active}">${escapeHTML(item.name)}</button>`;
        }).join('')}
      </div>
    </section>
  `).join('');
  q('presetGrid').querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
      q('formulaInput').value = button.dataset.formula;
      state.formulaTouched = true;
      renderQuantityPalette();
      renderQuantityExplanation();
      renderPresets();
      commitConfiguration();
    });
  });
}

function resetHistogram(resetRange = false) {
  state.histValues = [];
  state.processedEvents = 0;
  state.acceptedEvents = 0;
  state.lastEventIndex = null;
  if (resetRange) {
    q('histBins').value = state.mode === 'pairs' ? 60 : 65;
    q('histMin').value = state.mode === 'pairs' ? 0 : 70;
    q('histMax').value = state.mode === 'pairs' ? 160 : 200;
  }
  renderStudentHistogram();
}

function commitConfiguration() {
  resetHistogram();
  refreshFormulaEvent().catch(showError);
}

function setMode(mode, initial = false) {
  state.mode = mode;
  q('modePairs').classList.toggle('active', mode === 'pairs');
  q('modeFour').classList.toggle('active', mode === 'four');
  q('pairControls').hidden = mode !== 'pairs';
  q('channelControls').hidden = mode !== 'four';
  q('modeTitle').textContent = mode === 'pairs'
    ? 'Find a resonance using two-lepton pairs'
    : 'Find a resonance using all four leptons';
  q('modeHelp').textContent = mode === 'pairs'
    ? 'Choose how the four leptons are paired, then add both selected pair values to the histogram.'
    : 'No pairing choice is needed. Choose which decay channels contribute to the histogram.';
  state.requirements = state.requirements.filter(item => requirementById(item.id)?.modes.includes(mode));
  renderQuantityPalette();
  renderQuantityExplanation();
  renderPresets();
  renderRequirementControls();
  resetHistogram(true);
  if (!initial) loadRandomEvent().catch(showError);
}

async function loadRandomEvent() {
  const data = await postJSON(api.formulaEvent, requestBody({event_index: null}));
  state.currentEvent = data;
  state.currentValues = data.selected_values || [];
  renderCurrentEvent();
}

async function refreshFormulaEvent() {
  if (!currentFormula()) {
    state.currentValues = [];
    renderEmptyFormulaPrompt();
    return;
  }
  if (!state.currentEvent) return loadRandomEvent();
  const eventIndex = state.currentEvent.event.event_index;
  const data = await postJSON(api.formulaEvent, requestBody({event_index: eventIndex}));
  state.currentEvent = data;
  state.currentValues = data.selected_values || [];
  renderCurrentEvent();
}

function selectedPairing(data) {
  return (data.formula_pairings || []).find(row => Number(row.selected_by_rule) === 1) || null;
}

function renderCurrentEvent() {
  const data = state.currentEvent;
  if (!data) return;
  const event = data.event;
  const link = data.event_display_url
    ? ` · <a href="${escapeHTML(data.event_display_url)}" target="_blank" rel="noopener">open CMS display ↗</a>`
    : '';
  q('eventSummary').innerHTML = `Event ${escapeHTML(event.event_index)} · ${escapeHTML(event.n_lepton ?? (data.leptons || []).length)} leptons${link}`;

  const pair = state.mode === 'pairs' ? selectedPairing(data) : null;
  renderEventDisplay(data.leptons || [], pair);
  renderLeptons(data.leptons || [], pair);
  if (state.mode === 'pairs') renderPairResults(data.formula_pairings || []);
  else renderFourResults(data.formula_quads || []);
  renderEventRequirementStatus(data);
  q('addCurrent').disabled = !(data.passes_requirements && state.currentValues.length);
}

function renderEventRequirementStatus(data) {
  const box = q('eventRequirementStatus');
  if (!state.requirements.length) {
    box.className = 'event-pass neutral';
    box.textContent = 'No event requirements selected.';
    return;
  }
  const failed = (data.requirement_results || []).filter(item => !item.passed);
  if (!failed.length) {
    box.className = 'event-pass pass';
    box.textContent = 'This event passes all current requirements.';
  } else {
    box.className = 'event-pass fail';
    box.textContent = `Does not pass: ${failed.map(item => item.label).join(', ')}.`;
  }
}

function absMax(values, fallback = 1) {
  const finite = values.map(Number).filter(Number.isFinite).map(Math.abs);
  return finite.length ? Math.max(fallback, ...finite) : fallback;
}

function bar(label, value, max) {
  const number = Number(value);
  const width = Number.isFinite(number) ? Math.min(100, Math.max(0, Math.abs(number) / max * 100)) : 0;
  return `<div class="barline"><span>${label}</span><div class="bartrack"><i style="width:${width}%"></i></div><b>${numberFmt(number)} GeV</b></div>`;
}

function componentBars(object) {
  const max = absMax([object.energy, object.pt], 1);
  return [bar('E', object.energy, max), bar('pT', object.pt, max)].join('');
}

function pairObject(row, prefix) {
  return {
    energy: row[`${prefix}energy`],
    pt: row[`${prefix}pt`],
    charge: row[`${prefix}charge_sum`]
  };
}

function pairMembership(pair) {
  if (!pair) return new Map();
  return new Map([
    [Number(pair.pair_a_lep1_index), 'A'],
    [Number(pair.pair_a_lep2_index), 'A'],
    [Number(pair.pair_b_lep1_index), 'B'],
    [Number(pair.pair_b_lep2_index), 'B']
  ]);
}

function renderLeptons(leptons, pair) {
  const membership = pairMembership(pair);
  q('leptonList').innerHTML = leptons.map(lepton => {
    const member = membership.get(Number(lepton.lepton_index));
    return `
      <div class="lepton-chip ${member ? `pair-${member.toLowerCase()}` : ''}">
        <div class="lepton-title"><b>lepton ${escapeHTML(lepton.lepton_index)}: ${flavor(lepton.abs_pdgid)}${chargeString(lepton.charge)}</b>${member ? `<span>pair ${member}</span>` : ''}</div>
        <div class="mini-bars">${componentBars(lepton)}</div>
      </div>
    `;
  }).join('');
}

function renderEventDisplay(leptons, pair) {
  const width = 440;
  const height = 250;
  const phiMin = -Math.PI;
  const phiMax = Math.PI;
  const etaMin = -2.6;
  const etaMax = 2.6;
  const x = phi => 42 + (Number(phi) - phiMin) / (phiMax - phiMin) * (width - 84);
  const y = eta => height - 34 - (Number(eta) - etaMin) / (etaMax - etaMin) * (height - 68);
  const byIndex = new Map(leptons.map(lepton => [Number(lepton.lepton_index), lepton]));
  const membership = pairMembership(pair);
  const maxPt = absMax(leptons.map(lepton => lepton.pt), 1);

  const originGuides = leptons.map(lepton => `
    <line class="origin-guide" x1="${width / 2}" y1="${height / 2}" x2="${x(lepton.phi)}" y2="${y(lepton.eta)}"></line>
  `).join('');

  const pairLine = (firstIndex, secondIndex, className) => {
    const first = byIndex.get(Number(firstIndex));
    const second = byIndex.get(Number(secondIndex));
    if (!first || !second) return '';
    return `<line class="pair-link ${className}" x1="${x(first.phi)}" y1="${y(first.eta)}" x2="${x(second.phi)}" y2="${y(second.eta)}"></line>`;
  };
  const links = pair ? [
    pairLine(pair.pair_a_lep1_index, pair.pair_a_lep2_index, 'pair-a'),
    pairLine(pair.pair_b_lep1_index, pair.pair_b_lep2_index, 'pair-b')
  ].join('') : '';

  const points = leptons.map(lepton => {
    const cx = x(lepton.phi);
    const cy = y(lepton.eta);
    const radius = 7 + 11 * Math.sqrt(Math.max(0, Number(lepton.pt) || 0) / maxPt);
    const particleClass = Number(lepton.abs_pdgid) === 11 ? 'electron' : 'muon';
    const member = membership.get(Number(lepton.lepton_index));
    const chargeClass = Number(lepton.charge) > 0 ? 'positive' : 'negative';
    const badgeX = cx + radius * 0.72;
    const badgeY = cy - radius * 0.72;
    return `
      <g class="lepton-point ${particleClass} ${member ? `selected-${member.toLowerCase()}` : ''}">
        <circle cx="${cx}" cy="${cy}" r="${radius}"></circle>
        <text x="${cx}" y="${cy + 4}" text-anchor="middle">${escapeHTML(lepton.lepton_index)}</text>
        <title>lepton ${escapeHTML(lepton.lepton_index)}: ${flavor(lepton.abs_pdgid)}${chargeString(lepton.charge)}, pT ${numberFmt(lepton.pt)} GeV</title>
      </g>
      <g class="charge-badge ${chargeClass}">
        <circle cx="${badgeX}" cy="${badgeY}" r="6.5"></circle>
        <text x="${badgeX}" y="${badgeY + 3.2}" text-anchor="middle">${chargeString(lepton.charge)}</text>
      </g>`;
  }).join('');

  q('eventDisplay').innerHTML = `
    <div class="event-vis-header"><b>Event view</b><span>η–φ map; circle size follows pT</span></div>
    <svg class="eta-phi" viewBox="0 0 ${width} ${height}" role="img" aria-label="Eta phi event display with selected pair highlights">
      <rect x="1" y="1" width="${width - 2}" height="${height - 2}" rx="16"></rect>
      <line class="axis" x1="42" y1="${height / 2}" x2="${width - 42}" y2="${height / 2}"></line>
      <line class="axis" x1="${width / 2}" y1="28" x2="${width / 2}" y2="${height - 34}"></line>
      <text class="axis-label" x="${width / 2}" y="${height - 9}" text-anchor="middle">φ around detector</text>
      <text class="axis-label" transform="translate(14 ${height / 2}) rotate(-90)" text-anchor="middle">η along beam</text>
      ${originGuides}${links}${points}
    </svg>
    <div class="display-legend">
      <span class="legend-item"><span class="dot electron"></span>electron (e)</span>
      <span class="legend-item"><span class="dot muon"></span>muon (μ)</span>
      <span class="charge-key positive">+</span>positive <span class="charge-key negative">−</span>negative
      ${pair ? '<span class="line-key pair-a"></span>pair A <span class="line-key pair-b"></span>pair B' : ''}
    </div>
  `;
}

function pairLabel(row, which) {
  const prefix = which === 'A' ? 'pair_a_' : 'pair_b_';
  return `${row[`${prefix}lep1_index`]} + ${row[`${prefix}lep2_index`]}`;
}

function pairDetail(row, which) {
  const prefix = which === 'A' ? 'pair_a_' : 'pair_b_';
  const object = pairObject(row, prefix);
  const formulaValue = which === 'A' ? row.formula_pair_a : row.formula_pair_b;
  return `
    <div class="pair-object pair-${which.toLowerCase()}">
      <div class="pair-object-title"><b>Pair ${which}</b><span>leptons ${escapeHTML(pairLabel(row, which))}</span></div>
      <div class="formula-value"><code>${escapeHTML(currentFormula())}</code><b>${numberFmt(formulaValue, 4)}</b></div>
      <div class="object-visual">${componentBars(object)}</div>
      <div class="charge-row"><span>charge</span><b>${chargeString(object.charge)}</b></div>
    </div>
  `;
}

function renderEmptyFormulaPrompt() {
  q('formulaResults').innerHTML = '<p class="empty-state">Choose a quantity to calculate values for this event.</p>';
}

function renderPairResults(rows) {
  if (!currentFormula()) return renderEmptyFormulaPrompt();
  q('formulaResults').innerHTML = `
    <div class="result-heading"><h3>Possible pairings</h3><span>formula: <code>${escapeHTML(currentFormula())}</code></span></div>
    <div class="pairing-grid">
      ${rows.map(row => `
        <article class="pair-card ${row.selected_by_rule ? 'selected' : ''}">
          <div class="pair-card-heading"><h4>Pairing ${String.fromCharCode(65 + Number(row.pairing_id || 0))}</h4>${row.selected_by_rule ? '<span>selected</span>' : ''}</div>
          ${pairDetail(row, 'A')}
          ${pairDetail(row, 'B')}
        </article>
      `).join('')}
    </div>
  `;
}

function channelLabel(channel) {
  if (Number(channel) === 1) return '4e';
  if (Number(channel) === 2) return '4μ';
  if (Number(channel) === 3) return '2e2μ';
  return 'other';
}

function renderFourResults(rows) {
  if (!currentFormula()) return renderEmptyFormulaPrompt();
  if (!rows.length) {
    q('formulaResults').innerHTML = '<p class="empty-state">This event has no candidate in the selected channel.</p>';
    return;
  }
  q('formulaResults').innerHTML = `
    <div class="result-heading"><h3>Four-lepton value</h3><span>formula: <code>${escapeHTML(currentFormula())}</code></span></div>
    <div class="four-grid">
      ${rows.map(row => `
        <article class="pair-card selected">
          <div class="pair-card-heading"><h4>Leptons ${escapeHTML(row.lep1_index)} + ${escapeHTML(row.lep2_index)} + ${escapeHTML(row.lep3_index)} + ${escapeHTML(row.lep4_index)}</h4></div>
          <div class="formula-value"><code>${escapeHTML(currentFormula())}</code><b>${numberFmt(row.formula_value, 4)}</b></div>
          <div class="object-visual">${componentBars(row)}</div>
          <div class="quad-details"><span>charge <b>${chargeString(row.charge_sum)}</b></span><span>channel <b>${channelLabel(row.channel)}</b></span></div>
        </article>
      `).join('')}
    </div>
  `;
}

async function addCurrentEvent() {
  if (!currentFormula()) return showToast('Choose a formula before adding an event.');
  if (!state.currentEvent) await loadRandomEvent();
  state.processedEvents += 1;
  const values = (state.currentValues || []).map(Number).filter(Number.isFinite);
  if (values.length) {
    state.acceptedEvents += 1;
    state.histValues.push(...values);
  }
  renderStudentHistogram(values.length ? 'Current event added.' : 'Current event did not pass the requirements.');
  await loadRandomEvent();
}

async function addBatch() {
  if (!currentFormula()) return showToast('Choose a formula before adding events.');
  const button = q('addBatch');
  button.disabled = true;
  button.textContent = 'adding…';
  try {
    const data = await postJSON(api.formulaBatch, requestBody({
      start_after: state.lastEventIndex,
      limit: 1000
    }));
    state.histValues.push(...(data.values || []).map(Number).filter(Number.isFinite));
    state.processedEvents += Number(data.requested_events || 0);
    state.acceptedEvents += Number(data.accepted_events || 0);
    state.lastEventIndex = data.last_event_index;
    renderStudentHistogram(`Added ${Number(data.values.length).toLocaleString()} values from ${Number(data.accepted_events).toLocaleString()} accepted events.`);
    await loadRandomEvent();
  } finally {
    button.disabled = false;
    button.textContent = 'add next 1000 events';
  }
}

function histogramRange() {
  const low = Number(q('histMin').value);
  const high = Number(q('histMax').value);
  if (Number.isFinite(low) && Number.isFinite(high) && high > low) return [low, high];
  return state.mode === 'pairs' ? [0, 160] : [70, 200];
}

function makeHistogram(values, bins, range) {
  const counts = Array(bins).fill(0);
  const [low, high] = range;
  const width = (high - low) / bins;
  values.forEach(value => {
    if (!Number.isFinite(value) || value < low || value >= high) return;
    const index = Math.floor((value - low) / width);
    if (index >= 0 && index < bins) counts[index] += 1;
  });
  return {
    centers: counts.map((_, index) => low + (index + 0.5) * width),
    counts
  };
}

function shapeSignificance(counts) {
  const total = counts.reduce((sum, value) => sum + value, 0);
  if (total < 10 || !counts.some(Boolean)) return null;
  const peak = counts.indexOf(Math.max(...counts));
  const signalIndices = [peak - 1, peak, peak + 1].filter(index => index >= 0 && index < counts.length);
  const sideIndices = [];
  for (let offset = 3; offset <= 6; offset += 1) {
    if (peak - offset >= 0) sideIndices.push(peak - offset);
    if (peak + offset < counts.length) sideIndices.push(peak + offset);
  }
  if (sideIndices.length < 3) return null;
  const observed = signalIndices.reduce((sum, index) => sum + counts[index], 0);
  const sideTotal = sideIndices.reduce((sum, index) => sum + counts[index], 0);
  const background = sideTotal / sideIndices.length * signalIndices.length;
  if (!(background > 0)) return null;
  const signal = Math.max(0, observed - background);
  return signal / Math.sqrt(background * total);
}

function renderStudentHistogram(message = null) {
  const bins = Math.max(10, Math.min(200, Number.parseInt(q('histBins').value || '70', 10)));
  const range = histogramRange();
  const {centers, counts} = makeHistogram(state.histValues, bins, range);
  const formula = currentFormula() || 'no formula selected';
  q('histStatus').textContent = message || `${state.histValues.length.toLocaleString()} values · ${formula}`;
  q('histEntries').textContent = state.histValues.length.toLocaleString();
  q('histAccepted').textContent = `${state.acceptedEvents.toLocaleString()} / ${state.processedEvents.toLocaleString()}`;
  const significance = shapeSignificance(counts);
  q('significanceValue').textContent = significance === null ? '—' : numberFmt(significance, 2);
  if (!window.Plotly) return;
  Plotly.react('studentHist', [{
    type: 'bar',
    x: centers,
    y: counts,
    marker: {color: '#3152d9'},
    hovertemplate: 'value %{x:.3g}<br>entries %{y}<extra></extra>'
  }], {
    margin: {l: 54, r: 16, t: 12, b: 58},
    xaxis: {title: formula, zeroline: false, fixedrange: false},
    yaxis: {title: 'entries', rangemode: 'tozero', fixedrange: false},
    bargap: 0.025,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#263244'}
  }, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});
}

function resizeHistogram() {
  if (!state.histValues.length) {
    showToast('Add events before resizing the histogram.');
    return;
  }
  const finiteValues = state.histValues.map(Number).filter(Number.isFinite);
  if (!finiteValues.length) {
    showToast('No finite histogram values are available.');
    return;
  }
  let minimum = finiteValues[0];
  let maximum = finiteValues[0];
  for (const value of finiteValues) {
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  let low = minimum < 0 ? 1.1 * minimum : Math.min(0, 0.9 * minimum);
  let high = maximum < 0 ? 0.9 * maximum : 1.1 * maximum;
  if (!(high > low)) {
    const padding = Math.max(1, Math.abs(minimum) * 0.1);
    low = minimum - padding;
    high = maximum + padding;
  }
  q('histMin').value = Number(low.toPrecision(6));
  q('histMax').value = Number(high.toPrecision(6));
  renderStudentHistogram('Histogram bounds resized to all accumulated values.');
}

function requirementById(id) {
  return state.meta?.requirement_catalog?.find(item => item.id === id) || null;
}

function renderRequirementControls() {
  if (!state.meta) return;
  const available = state.meta.requirement_catalog.filter(item =>
    item.available && item.modes.includes(state.mode) && !state.requirements.some(active => active.id === item.id)
  );
  q('requirementSelect').innerHTML = available.length
    ? available.map(item => `<option value="${item.id}">${escapeHTML(item.label)}${item.comparison === 'required' ? '' : ` ${item.comparison} ${numberFmt(item.default)} ${escapeHTML(item.unit)}`}</option>`).join('')
    : '<option value="">All available requirements added</option>';
  q('addRequirement').disabled = !available.length;
  syncRequirementAdder(true);
  renderActiveRequirements();
}

function syncRequirementAdder(resetValue = false) {
  const definition = requirementById(q('requirementSelect').value);
  const numerical = definition && definition.comparison !== 'required';
  const editor = q('requirementThreshold');
  editor.hidden = !numerical;
  if (!numerical) return;
  q('requirementComparison').textContent = definition.comparison;
  q('requirementUnit').textContent = definition.unit || '';
  if (resetValue || !q('requirementValue').value) q('requirementValue').value = definition.default;
  q('requirementValue').setAttribute('aria-label', `${definition.label} threshold`);
}

function renderActiveRequirements() {
  const box = q('activeRequirements');
  if (!state.requirements.length) {
    box.innerHTML = '<p class="empty-state">No requirements selected.</p>';
    return;
  }
  box.innerHTML = state.requirements.map(active => {
    const definition = requirementById(active.id);
    const fixed = definition.comparison === 'required';
    return `
      <div class="requirement-row" data-id="${active.id}">
        <div class="requirement-copy">
          <div><b>${escapeHTML(definition.label)}</b></div>
          <small>${escapeHTML(definition.description)}</small>
        </div>
        <div class="requirement-value">
          <span>${definition.comparison}</span>
          ${fixed ? '<b>yes</b>' : `<input type="number" step="any" value="${active.value}" aria-label="${escapeHTML(definition.label)} threshold" /><em>${escapeHTML(definition.unit)}</em>`}
        </div>
        <button class="remove-requirement" aria-label="Remove ${escapeHTML(definition.label)}" title="Remove">×</button>
      </div>
    `;
  }).join('');
  box.querySelectorAll('.requirement-row').forEach(row => {
    const id = row.dataset.id;
    row.querySelector('.remove-requirement').addEventListener('click', () => {
      state.requirements = state.requirements.filter(item => item.id !== id);
      renderRequirementControls();
      commitConfiguration();
    });
    const input = row.querySelector('input');
    if (input) input.addEventListener('change', () => {
      const active = state.requirements.find(item => item.id === id);
      const value = Number(input.value);
      if (active && Number.isFinite(value)) active.value = value;
      commitConfiguration();
    });
  });
}

function addRequirement() {
  const id = q('requirementSelect').value;
  const definition = requirementById(id);
  if (!definition || state.requirements.some(item => item.id === id)) return;
  const numerical = definition.comparison !== 'required';
  const entered = numerical ? Number(q('requirementValue').value) : Number(definition.default);
  if (!Number.isFinite(entered)) return showToast('Enter a valid numerical threshold for this requirement.');
  state.requirements.push({id, value: entered});
  renderRequirementControls();
  commitConfiguration();
}

async function init() {
  state.meta = await clientAPI.metadata();
  if (state.meta?.app_version !== FRONTEND_VERSION) {
    throw new Error(`App version mismatch: interface ${FRONTEND_VERSION}, data ${state.meta?.app_version || 'older/unknown'}. Rebuild and upload the complete static site.`);
  }
  if (!Array.isArray(state.meta.requirement_catalog)) {
    throw new Error(`Backend ${FRONTEND_VERSION} did not return the event-requirement catalogue.`);
  }
  q('statusBox').textContent = `v${FRONTEND_VERSION}`;
  renderPresets();
  setMode('pairs', true);
  await loadRandomEvent();
}

window.addEventListener('DOMContentLoaded', () => {
  clientAPI.onProgress(message => {
    if (!state.meta) q('statusBox').textContent = message;
  });
  q('modePairs').addEventListener('click', () => setMode('pairs'));
  q('modeFour').addEventListener('click', () => setMode('four'));
  q('pairingRule').addEventListener('change', () => {
    q('pairingHelp').textContent = pairingHelp[q('pairingRule').value];
    commitConfiguration();
  });
  q('channelSelect').addEventListener('change', () => {
    resetHistogram();
    loadRandomEvent().catch(showError);
  });
  q('formulaInput').addEventListener('input', () => {
    state.formulaTouched = true;
    renderQuantityPalette();
    renderQuantityExplanation();
    renderPresets();
    if (!currentFormula()) renderEmptyFormulaPrompt();
  });
  q('formulaInput').addEventListener('change', commitConfiguration);
  q('formulaInput').addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      commitConfiguration();
    }
  });
  q('clearFormula').addEventListener('click', () => {
    q('formulaInput').value = '';
    state.formulaTouched = true;
    state.currentValues = [];
    renderQuantityPalette();
    renderQuantityExplanation();
    renderPresets();
    resetHistogram();
    renderEmptyFormulaPrompt();
    q('addCurrent').disabled = true;
  });
  q('loadEvent').addEventListener('click', () => loadRandomEvent().catch(showError));
  q('addCurrent').addEventListener('click', () => addCurrentEvent().catch(showError));
  q('addBatch').addEventListener('click', () => addBatch().catch(showError));
  q('clearHist').addEventListener('click', () => resetHistogram());
  q('resizeHist').addEventListener('click', resizeHistogram);
  ['histBins', 'histMin', 'histMax'].forEach(id => q(id).addEventListener('change', () => renderStudentHistogram()));
  q('requirementSelect').addEventListener('change', () => syncRequirementAdder(true));
  q('addRequirement').addEventListener('click', addRequirement);
  init().catch(showError);
});
