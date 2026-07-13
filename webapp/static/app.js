const q = id => document.getElementById(id);

const api = {
  metadata: '/api/metadata',
  event: '/api/formula_event',
  batch: '/api/formula_batch'
};

const DEFAULT_FORMULA = 'sqrt(E^2-px^2-py^2-pz^2)';

const state = {
  meta: null,
  workflow: 'z',
  optimizeTarget: 'pairs',
  channel: 'all',
  currentEvent: null,
  histValues: [],
  acceptedEvents: 0,
  lastEventIndex: null,
  activeRequirements: [],
  formulaTouched: false,
  toastTimer: null
};

const quantityDefinitions = [
  {symbol: 'E', display: 'E', label: 'energy', modes: ['pairs', 'four'], description: 'Combined energy of the selected particles.'},
  {symbol: 'p', display: 'p', label: 'momentum', modes: ['pairs', 'four'], description: 'Magnitude of the combined three-dimensional momentum.'},
  {symbol: 'pt', display: 'pT', label: 'transverse momentum', modes: ['pairs', 'four'], description: 'Combined momentum perpendicular to the beam.'},
  {symbol: 'px', label: 'x momentum', modes: ['pairs', 'four'], description: 'The x component of the combined momentum; signed components can cancel.'},
  {symbol: 'py', label: 'y momentum', modes: ['pairs', 'four'], description: 'The y component of the combined momentum.'},
  {symbol: 'pz', label: 'z momentum', modes: ['pairs', 'four'], description: 'The component of combined momentum along the beam.'},
  {symbol: 'scalar_pt_sum', display: 'ΣpT', label: 'pT sum', modes: ['pairs', 'four'], description: 'Sum of the individual lepton pT values, without vector cancellation.'},
  {symbol: 'charge', display: 'q', label: 'charge', modes: ['pairs', 'four'], description: 'Total electric charge of the selected object.'},
  {symbol: 'dr', display: 'ΔR', label: 'delta-R', modes: ['pairs'], description: 'Angular distance between the two leptons in a pair.'},
  {symbol: 'angle', display: 'θ', label: 'opening angle', modes: ['pairs'], description: 'Three-dimensional angle between the two leptons in a pair.'}
];

const presets = [
  {name: 'energy', formula: 'E', modes: ['pairs', 'four'], ranges: {pairs: [0, 350], four: [0, 800]}},
  {name: 'transverse momentum', formula: 'pt', modes: ['pairs', 'four'], ranges: {pairs: [0, 250], four: [0, 450]}},
  {name: 'momentum magnitude', formula: 'p', modes: ['pairs', 'four'], ranges: {pairs: [0, 350], four: [0, 700]}},
  {name: 'sum of lepton pT', formula: 'scalar_pt_sum', modes: ['pairs', 'four'], ranges: {pairs: [0, 300], four: [0, 550]}},
  {name: 'energy minus momentum', formula: 'E-p', modes: ['pairs', 'four'], ranges: {pairs: [0, 140], four: [0, 300]}},
  {name: 'invariant mass', formula: DEFAULT_FORMULA, modes: ['pairs', 'four'], ranges: {pairs: [40, 140], four: [70, 180]}},
  {name: 'mass squared', formula: 'E^2-px^2-py^2-pz^2', modes: ['pairs', 'four'], ranges: {pairs: [0, 22000], four: [4000, 40000]}},
  {name: 'angular distance', formula: 'dr', modes: ['pairs'], ranges: {pairs: [0, 6]}}
];

function analysisMode() {
  if (state.workflow === 'z') return 'pairs';
  if (state.workflow === 'h') return 'four';
  return state.optimizeTarget;
}

function currentFormula() {
  return q('formulaInput').value.trim();
}

function numberFmt(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  if (Math.abs(number) >= 10000) return number.toExponential(2);
  return number.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

function flavor(absPdgId) {
  return Number(absPdgId) === 11 ? 'e' : Number(absPdgId) === 13 ? 'μ' : '?';
}

function chargeString(charge) {
  return Number(charge) > 0 ? '+' : Number(charge) < 0 ? '−' : '0';
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    let message = await response.text();
    try { message = JSON.parse(message).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function showToast(message, isError = false) {
  const toast = q('toast');
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove('show'), 3600);
}

function showError(error) {
  console.error(error);
  showToast(error.message || String(error), true);
}

function requestBody(extra = {}) {
  return {
    mode: analysisMode(),
    formula: currentFormula() || 'E',
    pairing_rule: q('pairingRule').value || 'closest_distance',
    save_mode: q('saveMode').value || 'both',
    channel: analysisMode() === 'four' ? state.channel : 'all',
    requirements: state.activeRequirements.map(item => ({id: item.id, value: item.value ?? null})),
    ...extra
  };
}

function formulaUses(symbol) {
  const formula = currentFormula();
  return new RegExp(`(^|[^A-Za-z0-9_])${symbol}([^A-Za-z0-9_]|$)`).test(formula);
}

function insertSymbol(symbol) {
  const input = q('formulaInput');
  const formula = input.value.trim();
  if (!formula || (!state.formulaTouched && formula === DEFAULT_FORMULA)) {
    input.value = symbol;
  } else if (/[+\-*/^(]$/.test(formula)) {
    input.value = `${formula}${symbol}`;
  } else {
    input.value = `${formula}+${symbol}`;
  }
  state.formulaTouched = true;
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
  renderFormulaControls();
  clearHistogram('Formula changed. Rebuild the histogram with the new quantity.');
  refreshCurrentEvent().catch(showError);
}

function renderFormulaControls() {
  const mode = analysisMode();
  const available = quantityDefinitions.filter(item => item.modes.includes(mode));
  q('quantityPalette').innerHTML = available.map(item => `
    <button class="quantity-token ${formulaUses(item.symbol) ? 'active' : ''}" data-symbol="${item.symbol}" title="${item.label}">${item.display || item.symbol}</button>
  `).join('');
  q('quantityPalette').querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => insertSymbol(button.dataset.symbol));
  });

  const used = available.filter(item => formulaUses(item.symbol));
  q('quantityExplanation').innerHTML = used.length
    ? used.map(item => `<b>${item.symbol}</b>: ${item.description}`).join('<br>')
    : 'Choose a quantity or type an expression. Its calculated value will appear on every pair or four-lepton candidate.';
  q('formulaHelp').innerHTML = mode === 'pairs'
    ? 'Allowed: <code>E</code>, <code>p</code>, <code>pt</code>, momentum components, <code>scalar_pt_sum</code>, <code>dr</code>, <code>angle</code>, arithmetic, and <code>sqrt(...)</code>.'
    : 'Allowed: <code>E</code>, <code>p</code>, <code>pt</code>, momentum components, <code>scalar_pt_sum</code>, <code>charge</code>, arithmetic, and <code>sqrt(...)</code>.';

  q('presetGrid').innerHTML = presets.filter(item => item.modes.includes(mode)).map(item => `
    <button class="preset" data-formula="${item.formula}" data-min="${item.ranges[mode][0]}" data-max="${item.ranges[mode][1]}">${item.name}</button>
  `).join('');
  q('presetGrid').querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
      q('formulaInput').value = button.dataset.formula;
      q('histMin').value = button.dataset.min;
      q('histMax').value = button.dataset.max;
      state.formulaTouched = true;
      renderFormulaControls();
      clearHistogram('Formula changed. Rebuild the histogram with the new quantity.');
      refreshCurrentEvent().catch(showError);
    });
  });
}

function setDefaultAxis() {
  if (analysisMode() === 'pairs') {
    q('histMin').value = 40;
    q('histMax').value = 140;
  } else {
    q('histMin').value = 70;
    q('histMax').value = 180;
  }
}

function pruneRequirementsForMode() {
  const mode = analysisMode();
  const valid = new Set(state.meta.requirements.filter(item => item.modes.includes(mode)).map(item => item.id));
  const before = state.activeRequirements.length;
  state.activeRequirements = state.activeRequirements.filter(item => valid.has(item.id));
  if (state.activeRequirements.length < before) showToast('Requirements that do not apply to this object were removed.');
}

function updateTaskUI() {
  const mode = analysisMode();
  document.querySelectorAll('.workflow').forEach(button => button.classList.toggle('active', button.dataset.workflow === state.workflow));
  q('optimizeTargetWrap').classList.toggle('hidden', state.workflow !== 'optimize');
  q('requirementsCard').classList.toggle('emphasis', state.workflow === 'optimize');
  document.querySelectorAll('.seg').forEach(button => button.classList.toggle('active', button.dataset.target === state.optimizeTarget));
  q('pairControls').classList.toggle('hidden', mode !== 'pairs');
  q('channelControls').classList.toggle('hidden', mode !== 'four');

  if (state.workflow === 'z') {
    q('taskLabel').textContent = 'Task 1 · Z candidates';
    q('taskTitle').textContent = 'What should we calculate for each pair?';
    q('taskHelp').textContent = 'The four leptons can be split into two pairs in three different ways.';
  } else if (state.workflow === 'h') {
    q('taskLabel').textContent = 'Task 2 · Higgs candidates';
    q('taskTitle').textContent = 'What should we calculate from all four leptons?';
    q('taskHelp').textContent = 'All four leptons are combined once; no pairing rule is needed.';
  } else {
    q('taskLabel').textContent = `Task 3 · Optimize ${mode === 'pairs' ? 'Z' : 'Higgs'} candidates`;
    q('taskTitle').textContent = 'Which requirements improve the window score?';
    q('taskHelp').textContent = 'Change one requirement at a time, rebuild the histogram, and compare S/√B.';
  }
  q('eventTitle').textContent = mode === 'pairs' ? 'Three possible pairings' : 'Four-lepton candidate';
  renderFormulaControls();
  pruneRequirementsForMode();
  renderRequirementSelector();
  renderActiveRequirements();
  setDefaultAxis();
  clearHistogram('Analysis target changed. Build a new histogram.');
  loadRandomEvent().catch(showError);
}

function setWorkflow(workflow) {
  const previousMode = analysisMode();
  state.workflow = workflow;
  if (workflow === 'optimize') state.optimizeTarget = previousMode;
  if (!state.meta) return;
  updateTaskUI();
}

function setOptimizeTarget(target) {
  state.optimizeTarget = target;
  if (!state.meta) return;
  updateTaskUI();
}

function renderPairingOptions() {
  q('pairingRule').innerHTML = state.meta.pairing_strategies.map(strategy => `<option value="${strategy.id}">${strategy.label}</option>`).join('');
  q('pairingRule').value = 'closest_distance';
  updatePairingDescription();
}

function updatePairingDescription() {
  const strategy = state.meta.pairing_strategies.find(item => item.id === q('pairingRule').value);
  q('pairingDescription').textContent = strategy?.description || '';
}

function availableRequirements() {
  const mode = analysisMode();
  const active = new Set(state.activeRequirements.map(item => item.id));
  return state.meta.requirements.filter(item => item.modes.includes(mode) && !active.has(item.id));
}

function renderRequirementSelector() {
  const available = availableRequirements();
  const groups = [
    ['analysis', 'H→ZZ analysis selections'],
    ['explore', 'Exploratory quantities']
  ];
  q('requirementSelect').innerHTML = groups.map(([kind, label]) => {
    const options = available.filter(item => item.kind === kind).map(item => `<option value="${item.id}">${item.label}</option>`).join('');
    return options ? `<optgroup label="${label}">${options}</optgroup>` : '';
  }).join('') || '<option value="">All available requirements are active</option>';
  q('addRequirement').disabled = available.length === 0;
  updateRequirementDescription();
}

function updateRequirementDescription() {
  const item = state.meta.requirements.find(entry => entry.id === q('requirementSelect').value);
  q('requirementDescription').textContent = item?.description || '';
}

function defaultRequirementValue(item) {
  if (item.input === 'fixed') return null;
  if (item.default_by_mode) return Number(item.default_by_mode[analysisMode()]);
  return Number(item.default);
}

function addRequirement() {
  const item = state.meta.requirements.find(entry => entry.id === q('requirementSelect').value);
  if (!item) return;
  state.activeRequirements.push({id: item.id, value: defaultRequirementValue(item)});
  renderRequirementSelector();
  renderActiveRequirements();
  clearHistogram('Requirements changed. Rebuild the histogram to apply them consistently.');
  refreshCurrentEvent().catch(showError);
}

function removeRequirement(id) {
  state.activeRequirements = state.activeRequirements.filter(item => item.id !== id);
  renderRequirementSelector();
  renderActiveRequirements();
  clearHistogram('Requirements changed. Rebuild the histogram to apply them consistently.');
  refreshCurrentEvent().catch(showError);
}

function renderActiveRequirements() {
  q('selectionCount').textContent = `${state.activeRequirements.length} active`;
  if (!state.activeRequirements.length) {
    q('activeRequirements').innerHTML = '<p class="empty-state">No requirements: every candidate event can enter the histogram.</p>';
    return;
  }
  q('activeRequirements').innerHTML = state.activeRequirements.map(active => {
    const definition = state.meta.requirements.find(item => item.id === active.id);
    const valueControl = definition.input === 'fixed'
      ? '<span class="requirement-value">required</span>'
      : `<label class="requirement-value"><input type="number" step="any" value="${active.value}" data-requirement-value="${active.id}" aria-label="${definition.label}" /> ${definition.unit || ''}</label>`;
    return `
      <div class="requirement-row">
        <div class="requirement-copy"><b>${definition.label}</b><span class="${definition.kind}">${definition.kind === 'analysis' ? 'analysis' : 'explore'}</span></div>
        ${valueControl}
        <button class="remove-requirement" data-remove-requirement="${active.id}" aria-label="Remove ${definition.label}">×</button>
      </div>
    `;
  }).join('');
  q('activeRequirements').querySelectorAll('[data-remove-requirement]').forEach(button => {
    button.addEventListener('click', () => removeRequirement(button.dataset.removeRequirement));
  });
  q('activeRequirements').querySelectorAll('[data-requirement-value]').forEach(input => {
    input.addEventListener('change', () => {
      const active = state.activeRequirements.find(item => item.id === input.dataset.requirementValue);
      if (active) active.value = Number(input.value);
      clearHistogram('A requirement value changed. Rebuild the histogram.');
      refreshCurrentEvent().catch(showError);
    });
  });
}

async function loadRandomEvent() {
  const data = await postJSON(api.event, requestBody({event_index: null}));
  state.currentEvent = data;
  renderCurrentEvent();
}

async function refreshCurrentEvent() {
  if (!state.currentEvent) return loadRandomEvent();
  try {
    const data = await postJSON(api.event, requestBody({event_index: state.currentEvent.event.event_index}));
    state.currentEvent = data;
    renderCurrentEvent();
  } catch (error) {
    if (String(error.message).includes('selected channel')) return loadRandomEvent();
    throw error;
  }
}

function selectedPairing() {
  return state.currentEvent?.pairings?.find(pairing => pairing.selected) || null;
}

function renderCurrentEvent() {
  const data = state.currentEvent;
  if (!data) return;
  const event = data.event;
  q('eventSummary').innerHTML = `
    <b>Event ${event.event_index}</b> · ${data.four_object.channel} · charge sum ${chargeString(data.four_object.charge_sum)}
    ${data.event_display_url ? ` · <a href="${data.event_display_url}" target="_blank" rel="noopener">open matching detector display ↗</a>` : ''}
  `;
  renderEventDisplay(data.leptons, selectedPairing());
  q('leptonList').innerHTML = data.leptons.map(lepton => `
    <div class="lepton-chip">
      <b>lepton ${lepton.lepton_index}: ${flavor(lepton.abs_pdgid)}${chargeString(lepton.charge)}</b>
      <span>E ${numberFmt(lepton.energy)} GeV · pT ${numberFmt(lepton.pt)} GeV · charge ${chargeString(lepton.charge)}</span>
    </div>
  `).join('');
  if (analysisMode() === 'pairs') renderPairResults(data.pairings);
  else renderFourResult(data.four_object);
  renderEventRequirementStatus(data);
}

function renderEventDisplay(leptons, pairing) {
  const width = 430, height = 255;
  const x = phi => 42 + (Number(phi) + Math.PI) / (2 * Math.PI) * (width - 84);
  const y = eta => height - 35 - (Number(eta) + 2.6) / 5.2 * (height - 72);
  const maxPt = Math.max(1, ...leptons.map(lepton => Number(lepton.pt) || 0));
  const group = new Map();
  if (pairing) {
    pairing.pair_a.lepton_indices.forEach(index => group.set(Number(index), 'pair-a'));
    pairing.pair_b.lepton_indices.forEach(index => group.set(Number(index), 'pair-b'));
  }
  const lookup = new Map(leptons.map(lepton => [Number(lepton.lepton_index), lepton]));
  const link = (indices, className) => {
    if (!indices || indices.length !== 2) return '';
    const first = lookup.get(Number(indices[0])), second = lookup.get(Number(indices[1]));
    if (!first || !second) return '';
    return `<line class="pair-link ${className}" x1="${x(first.phi)}" y1="${y(first.eta)}" x2="${x(second.phi)}" y2="${y(second.eta)}"></line>`;
  };
  const links = pairing
    ? link(pairing.pair_a.lepton_indices, 'pair-a') + link(pairing.pair_b.lepton_indices, 'pair-b')
    : '';
  const points = leptons.map(lepton => {
    const radius = 7 + 11 * Math.sqrt(Math.max(0, Number(lepton.pt) || 0) / maxPt);
    const className = pairing ? (group.get(Number(lepton.lepton_index)) || '') : 'four';
    return `<g class="lepton-point ${className}"><circle cx="${x(lepton.phi)}" cy="${y(lepton.eta)}" r="${radius}"></circle><text x="${x(lepton.phi)}" y="${y(lepton.eta) + 4}" text-anchor="middle">${lepton.lepton_index}</text></g>`;
  }).join('');
  q('eventDisplay').innerHTML = `
    <div class="event-vis-header"><b>η–φ event view</b><span>circle size follows pT</span></div>
    <svg class="eta-phi" viewBox="0 0 ${width} ${height}" role="img" aria-label="Four-lepton eta phi view">
      <rect x="1" y="1" width="${width - 2}" height="${height - 2}" rx="13"></rect>
      <line class="axis" x1="42" y1="${height / 2}" x2="${width - 42}" y2="${height / 2}"></line>
      <line class="axis" x1="${width / 2}" y1="28" x2="${width / 2}" y2="${height - 35}"></line>
      <text class="axis-label" x="${width / 2}" y="${height - 10}" text-anchor="middle">φ around detector</text>
      <text class="axis-label" transform="translate(14 ${height / 2}) rotate(-90)" text-anchor="middle">η along beam</text>
      ${links}${points}
    </svg>
    <div class="display-legend">${pairing ? '<span><i class="legend-line a"></i>pair A</span><span><i class="legend-line b"></i>pair B</span>' : '<span><i class="legend-line four"></i>four-lepton candidate</span>'}</div>
  `;
}

function pairSummary(pair) {
  return `
    <div class="pair-value">
      <span>leptons ${pair.lepton_indices.join(' + ')}</span>
      <b>formula → ${numberFmt(pair.formula_value, 3)}</b>
      <small>E ${numberFmt(pair.energy)} GeV · pT ${numberFmt(pair.pt)} GeV · q ${chargeString(pair.charge_sum)}</small>
    </div>
  `;
}

function renderPairResults(pairings) {
  q('formulaResults').innerHTML = `
    <h3>Pairing view · <code>${currentFormula() || 'E'}</code></h3>
    <div class="pairing-grid">
      ${pairings.map(pairing => `
        <div class="pairing-card ${pairing.selected ? 'selected' : ''}">
          <div class="pairing-card-header"><h4>Pairing ${Number(pairing.pairing_id) + 1}</h4>${pairing.selected ? '<span class="selected-badge">chosen by strategy</span>' : ''}</div>
          <div class="pair-values">${pairSummary(pairing.pair_a)}${pairSummary(pairing.pair_b)}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderFourResult(four) {
  q('formulaResults').innerHTML = `
    <h3>Four-lepton view · <code>${currentFormula() || 'E'}</code></h3>
    <div class="four-result">
      <span>formula value</span><strong>${numberFmt(four.formula_value, 3)}</strong>
      <div class="four-components">
        <span class="component-pill">E ${numberFmt(four.energy)} GeV</span>
        <span class="component-pill">pT ${numberFmt(four.pt)} GeV</span>
        <span class="component-pill">charge ${chargeString(four.charge_sum)}</span>
        <span class="component-pill">channel ${four.channel}</span>
      </div>
    </div>
  `;
}

function renderEventRequirementStatus(data) {
  const box = q('eventRequirementStatus');
  if (!state.activeRequirements.length) {
    box.className = 'requirement-status neutral';
    box.textContent = 'The current event has no requirements to test.';
  } else if (data.passes_requirements) {
    box.className = 'requirement-status pass';
    box.textContent = 'Current event passes every active requirement and can enter the histogram.';
  } else {
    const labels = data.failed_requirements.map(id => state.meta.requirements.find(item => item.id === id)?.label || id);
    box.className = 'requirement-status fail';
    box.textContent = `Current event would be rejected by: ${labels.join('; ')}.`;
  }
}

function histogramRange() {
  const minimum = Number(q('histMin').value), maximum = Number(q('histMax').value);
  return Number.isFinite(minimum) && Number.isFinite(maximum) && maximum > minimum ? [minimum, maximum] : [0, 200];
}

function makeHistogram(values, bins, range) {
  const counts = Array(bins).fill(0);
  const width = (range[1] - range[0]) / bins;
  values.forEach(value => {
    if (!Number.isFinite(value) || value < range[0] || value >= range[1]) return;
    const index = Math.floor((value - range[0]) / width);
    if (index >= 0 && index < bins) counts[index] += 1;
  });
  return {counts, centers: counts.map((_, index) => range[0] + (index + .5) * width)};
}

function scoreDefinition() {
  return analysisMode() === 'pairs'
    ? {name: 'Z', signal: [80, 100], sidebands: [[60, 80], [100, 120]]}
    : {name: 'Higgs', signal: [120, 130], sidebands: [[105, 120], [130, 145]]};
}

function windowCount(values, range) {
  return values.filter(value => value >= range[0] && value < range[1]).length;
}

function updateScore() {
  const definition = scoreDefinition();
  const signalCount = windowCount(state.histValues, definition.signal);
  const sidebandCount = definition.sidebands.reduce((total, range) => total + windowCount(state.histValues, range), 0);
  const signalWidth = definition.signal[1] - definition.signal[0];
  const sidebandWidth = definition.sidebands.reduce((total, range) => total + range[1] - range[0], 0);
  const background = sidebandCount * signalWidth / sidebandWidth;
  const signal = Math.max(0, signalCount - background);
  const score = background > 0 ? signal / Math.sqrt(background) : null;
  q('significanceValue').textContent = score === null ? '—' : numberFmt(score, 2);
  q('significanceDetail').textContent = `${definition.name} window ${definition.signal[0]}–${definition.signal[1]}: S ≈ ${numberFmt(signal, 1)}, B ≈ ${numberFmt(background, 1)} from adjacent sidebands.`;
}

function renderHistogram(message = null) {
  const bins = Math.max(10, Math.min(200, Number(q('histBins').value) || 70));
  const range = histogramRange();
  const histogram = makeHistogram(state.histValues, bins, range);
  const definition = scoreDefinition();
  const shapes = [
    {type: 'rect', xref: 'x', yref: 'paper', x0: definition.signal[0], x1: definition.signal[1], y0: 0, y1: 1, fillcolor: 'rgba(13,138,98,.13)', line: {width: 0}, layer: 'below'},
    ...definition.sidebands.map(sideband => ({type: 'rect', xref: 'x', yref: 'paper', x0: sideband[0], x1: sideband[1], y0: 0, y1: 1, fillcolor: 'rgba(100,116,139,.08)', line: {width: 0}, layer: 'below'}))
  ];
  Plotly.react('studentHist', [{type: 'bar', x: histogram.centers, y: histogram.counts, marker: {color: '#1769aa'}, hovertemplate: 'value %{x:.3g}<br>count %{y}<extra></extra>'}], {
    margin: {l: 55, r: 15, t: 12, b: 58},
    xaxis: {title: currentFormula() || 'calculated quantity', range, zeroline: false},
    yaxis: {title: 'entries', rangemode: 'tozero'},
    bargap: .025,
    shapes,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)'
  }, {responsive: true, displaylogo: false});
  q('histEntries').textContent = state.histValues.length.toLocaleString();
  q('acceptedEvents').textContent = state.acceptedEvents.toLocaleString();
  q('histStatus').textContent = message || (state.histValues.length ? `${state.histValues.length.toLocaleString()} calculated values accumulated.` : 'No events added yet.');
  updateScore();
}

function clearHistogram(message = null) {
  state.histValues = [];
  state.acceptedEvents = 0;
  state.lastEventIndex = null;
  renderHistogram(message);
}

async function addCurrentEvent() {
  if (!currentFormula()) return showToast('Choose or type a formula first.', true);
  if (!state.currentEvent) await loadRandomEvent();
  else await refreshCurrentEvent();
  if (!state.currentEvent.passes_requirements) {
    showToast('Current event fails the active requirements and was not added.');
    return loadRandomEvent();
  }
  const values = state.currentEvent.selected_values.map(Number).filter(Number.isFinite);
  state.histValues.push(...values);
  state.acceptedEvents += 1;
  renderHistogram(`Added ${values.length} value${values.length === 1 ? '' : 's'} from the current event.`);
  await loadRandomEvent();
}

async function addBatch() {
  if (!currentFormula()) return showToast('Choose or type a formula first.', true);
  const button = q('addBatch');
  button.disabled = true;
  button.textContent = 'adding…';
  try {
    await refreshCurrentEvent();
    const data = await postJSON(api.batch, requestBody({start_after: state.lastEventIndex, limit: 1000}));
    state.histValues.push(...data.values.map(Number).filter(Number.isFinite));
    state.acceptedEvents += data.accepted_events;
    state.lastEventIndex = data.last_event_index;
    renderHistogram(`Processed ${data.requested_events.toLocaleString()} events: ${data.accepted_events.toLocaleString()} accepted, ${data.failed_requirements.toLocaleString()} rejected by requirements${data.missing_channel ? `, ${data.missing_channel.toLocaleString()} outside the selected channel` : ''}.`);
    await loadRandomEvent();
  } finally {
    button.disabled = false;
    button.textContent = 'add 1,000 events';
  }
}

async function initialize() {
  const response = await fetch(api.metadata);
  if (!response.ok) throw new Error(await response.text());
  state.meta = await response.json();
  const warning = state.meta.load_warnings.length ? ` · ${state.meta.load_warnings.length} data warning` : '';
  q('statusBox').textContent = `${state.meta.sample_name} · ${state.meta.events.toLocaleString()} events · app v${state.meta.app_version || 'unknown'}${warning}`;
  q('eventTitle').textContent = 'Three possible pairings';
  renderPairingOptions();
  renderFormulaControls();
  renderRequirementSelector();
  renderActiveRequirements();
  renderHistogram();
  await loadRandomEvent();
}

window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.workflow').forEach(button => button.addEventListener('click', () => setWorkflow(button.dataset.workflow)));
  document.querySelectorAll('.seg').forEach(button => button.addEventListener('click', () => setOptimizeTarget(button.dataset.target)));
  document.querySelectorAll('.channel').forEach(button => button.addEventListener('click', () => {
    state.channel = button.dataset.channel;
    document.querySelectorAll('.channel').forEach(item => item.classList.toggle('active', item.dataset.channel === state.channel));
    clearHistogram('Channel changed. Build a new histogram.');
    loadRandomEvent().catch(showError);
  }));
  q('pairingRule').addEventListener('change', () => {
    updatePairingDescription();
    clearHistogram('Pairing strategy changed. Build a new histogram.');
    refreshCurrentEvent().catch(showError);
  });
  q('saveMode').addEventListener('change', () => {
    clearHistogram('Histogram entry rule changed. Build a new histogram.');
    refreshCurrentEvent().catch(showError);
  });
  q('formulaInput').addEventListener('input', () => {
    state.formulaTouched = true;
    renderFormulaControls();
    clearHistogram('Formula edited. Calculate it or add events to rebuild the histogram.');
  });
  q('formulaInput').addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      clearHistogram('Formula changed. Build a new histogram.');
      refreshCurrentEvent().catch(showError);
    }
  });
  q('calculateFormula').addEventListener('click', () => {
    clearHistogram('Formula changed. Build a new histogram.');
    refreshCurrentEvent().catch(showError);
  });
  q('clearFormula').addEventListener('click', () => {
    q('formulaInput').value = '';
    state.formulaTouched = true;
    renderFormulaControls();
    clearHistogram('Formula cleared.');
    q('formulaResults').innerHTML = '<p class="empty-state">Choose a quantity or type a formula.</p>';
  });
  q('loadEvent').addEventListener('click', () => loadRandomEvent().catch(showError));
  q('addCurrent').addEventListener('click', () => addCurrentEvent().catch(showError));
  q('addBatch').addEventListener('click', () => addBatch().catch(showError));
  q('clearHist').addEventListener('click', () => clearHistogram());
  ['histBins', 'histMin', 'histMax'].forEach(id => q(id).addEventListener('change', () => renderHistogram()));
  q('requirementSelect').addEventListener('change', updateRequirementDescription);
  q('addRequirement').addEventListener('click', addRequirement);
  initialize().catch(showError);
});
