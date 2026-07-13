const TYPED_ARRAYS = {
  Float32Array,
  Float64Array,
  Int32Array,
  Uint32Array,
  Uint8Array
};

const ALLOWED_FUNCTIONS = {
  sqrt: Math.sqrt,
  abs: Math.abs,
  log: Math.log,
  exp: Math.exp,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  minimum: Math.min,
  maximum: Math.max,
  min: Math.min,
  max: Math.max
};

const LEGACY_PAIRING_RULES = {
  closest_target: 'closest_distance',
  smallest_difference: 'closest_formula',
  largest_difference: 'furthest_formula',
  first: 'closest_distance',
  largest_sum: 'closest_distance',
  smallest_sum: 'closest_distance',
  largest_value: 'closest_distance',
  smallest_value: 'closest_distance'
};

const LEGACY_REQUIREMENT_IDS = {
  pair_min_scalar_pt_sum: 'pair_min_pt_total',
  quad_min_scalar_pt_sum: 'quad_min_pt_total'
};

const CHANNEL_CODES = { '4e': 1, '4mu': 2, '2e2mu': 3 };

function finite(value) {
  return Number.isFinite(Number(value));
}

function jsonValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function lowerBound(array, value) {
  let low = 0;
  let high = array.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (Number(array[middle]) < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBound(array, value) {
  let low = 0;
  let high = array.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (Number(array[middle]) <= value) low = middle + 1;
    else high = middle;
  }
  return low;
}

function crc32(text) {
  let crc = 0xffffffff;
  for (let index = 0; index < text.length; index += 1) {
    crc ^= text.charCodeAt(index);
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

class FormulaParser {
  constructor(expression) {
    this.expression = String(expression || '').replaceAll('^', '**');
    this.tokens = this.tokenize(this.expression);
    this.position = 0;
  }

  tokenize(source) {
    const tokens = [];
    let position = 0;
    while (position < source.length) {
      const rest = source.slice(position);
      const whitespace = rest.match(/^\s+/);
      if (whitespace) {
        position += whitespace[0].length;
        continue;
      }
      const number = rest.match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
      if (number) {
        tokens.push({ type: 'number', value: Number(number[0]) });
        position += number[0].length;
        continue;
      }
      const identifier = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (identifier) {
        tokens.push({ type: 'identifier', value: identifier[0] });
        position += identifier[0].length;
        continue;
      }
      const operator = rest.startsWith('**') ? '**' : rest[0];
      if ('+-*/(),'.includes(operator) || operator === '**') {
        tokens.push({ type: operator, value: operator });
        position += operator.length;
        continue;
      }
      throw new Error(`Unsupported character ${JSON.stringify(rest[0])} in formula`);
    }
    tokens.push({ type: 'end', value: '' });
    return tokens;
  }

  current() {
    return this.tokens[this.position];
  }

  accept(type) {
    if (this.current().type !== type) return null;
    return this.tokens[this.position++];
  }

  expect(type) {
    const token = this.accept(type);
    if (!token) throw new Error(`Expected ${type} in formula`);
    return token;
  }

  parse() {
    const node = this.parseAdditive();
    this.expect('end');
    return node;
  }

  parseAdditive() {
    let node = this.parseMultiplicative();
    while (this.current().type === '+' || this.current().type === '-') {
      const operator = this.current().type;
      this.position += 1;
      node = { type: 'binary', operator, left: node, right: this.parseMultiplicative() };
    }
    return node;
  }

  parseMultiplicative() {
    let node = this.parseUnary();
    while (this.current().type === '*' || this.current().type === '/') {
      const operator = this.current().type;
      this.position += 1;
      node = { type: 'binary', operator, left: node, right: this.parseUnary() };
    }
    return node;
  }

  parseUnary() {
    if (this.current().type === '+' || this.current().type === '-') {
      const operator = this.current().type;
      this.position += 1;
      return { type: 'unary', operator, value: this.parseUnary() };
    }
    return this.parsePower();
  }

  parsePower() {
    const node = this.parsePrimary();
    if (!this.accept('**')) return node;
    return { type: 'binary', operator: '**', left: node, right: this.parseUnary() };
  }

  parsePrimary() {
    const number = this.accept('number');
    if (number) return { type: 'number', value: number.value };

    const identifier = this.accept('identifier');
    if (identifier) {
      if (!this.accept('(')) return { type: 'variable', name: identifier.value };
      if (!(identifier.value in ALLOWED_FUNCTIONS)) {
        throw new Error(`Unsupported function: ${identifier.value}`);
      }
      const args = [this.parseAdditive()];
      if (this.accept(',')) args.push(this.parseAdditive());
      this.expect(')');
      if (args.length < 1 || args.length > 2) throw new Error('Functions take one or two arguments');
      return { type: 'call', name: identifier.value, args };
    }

    if (this.accept('(')) {
      const node = this.parseAdditive();
      this.expect(')');
      return node;
    }
    throw new Error('Expected a number, variable, function, or parenthesized expression');
  }
}

function evaluateAst(node, variables) {
  if (node.type === 'number') return node.value;
  if (node.type === 'variable') {
    if (!(node.name in variables)) throw new Error(`Unknown formula variable: ${node.name}`);
    return variables[node.name];
  }
  if (node.type === 'unary') {
    const value = evaluateAst(node.value, variables);
    return node.operator === '-' ? -value : value;
  }
  if (node.type === 'binary') {
    const left = evaluateAst(node.left, variables);
    const right = evaluateAst(node.right, variables);
    if (node.operator === '+') return left + right;
    if (node.operator === '-') return left - right;
    if (node.operator === '*') return left * right;
    if (node.operator === '/') return left / right;
    if (node.operator === '**') return Math.pow(left, right);
  }
  if (node.type === 'call') {
    const args = node.args.map(argument => evaluateAst(argument, variables));
    return ALLOWED_FUNCTIONS[node.name](...args);
  }
  throw new Error('Unsupported formula syntax');
}

function parseFormula(expression) {
  return new FormulaParser(expression).parse();
}

export class StaticDataEngine {
  constructor(dataBaseUrl, progress = () => {}) {
    this.dataBaseUrl = new URL(dataBaseUrl);
    this.progress = progress;
    this.manifest = null;
    this.collections = new Map();
    this.collectionPromises = new Map();
    this.candidateCache = new Map();
  }

  async init() {
    if (this.manifest) return this.manifest.public_metadata;
    this.progress('Loading masterclass index…');
    const url = new URL(`manifest.json?v=4.0.0`, this.dataBaseUrl);
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Could not load ${url.pathname}: HTTP ${response.status}`);
    this.manifest = await response.json();
    if (this.manifest.app_version !== '4.0.0') {
      throw new Error(`Static data version ${this.manifest.app_version || 'unknown'} does not match app 4.0.0`);
    }
    this.requirementMap = new Map(this.manifest.requirements.map(item => [item.id, item]));
    this.progress('Masterclass index ready');
    return this.manifest.public_metadata;
  }

  async loadCollection(name) {
    await this.init();
    if (this.collections.has(name)) return this.collections.get(name);
    if (this.collectionPromises.has(name)) return this.collectionPromises.get(name);
    const promise = this.loadCollectionNow(name);
    this.collectionPromises.set(name, promise);
    try {
      return await promise;
    } finally {
      this.collectionPromises.delete(name);
    }
  }

  async loadCollectionNow(name) {
    const spec = this.manifest.collections[name];
    if (!spec) throw new Error(`Unknown static collection: ${name}`);
    this.progress(`Loading ${name} data…`);
    const response = await fetch(new URL(spec.file, this.dataBaseUrl));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    const compressed = await response.arrayBuffer();
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser cannot decompress the masterclass data. Please use a current version of Chrome, Edge, Firefox, or Safari.');
    }
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream(spec.compression));
    const buffer = await new Response(stream).arrayBuffer();
    if (buffer.byteLength !== spec.uncompressed_bytes) {
      throw new Error(`${name} data is incomplete: expected ${spec.uncompressed_bytes} bytes, received ${buffer.byteLength}`);
    }
    const columns = {};
    for (const [column, descriptor] of Object.entries(spec.columns)) {
      const Constructor = TYPED_ARRAYS[descriptor.type];
      if (!Constructor) throw new Error(`Unsupported browser array type ${descriptor.type}`);
      columns[column] = new Constructor(buffer, descriptor.offset, descriptor.length);
    }
    const collection = { name, rows: spec.rows, columns, buffer };
    this.collections.set(name, collection);
    this.progress(`${name} ready`);
    return collection;
  }

  async metadata() {
    return this.init();
  }

  normalizeRequest(input) {
    const request = {
      mode: input?.mode === 'four' ? 'four' : 'pairs',
      formula: String(input?.formula || 'E'),
      pairing_rule: LEGACY_PAIRING_RULES[input?.pairing_rule] || input?.pairing_rule || 'closest_distance',
      channel: ['all', '4e', '4mu', '2e2mu'].includes(input?.channel) ? input.channel : 'all',
      requirements: Array.isArray(input?.requirements)
        ? input.requirements.map(item => ({
            id: LEGACY_REQUIREMENT_IDS[item.id] || String(item.id),
            value: item.value === null || item.value === undefined ? null : Number(item.value)
          }))
        : [],
      event_index: input?.event_index === null || input?.event_index === undefined ? null : Number(input.event_index),
      start_after: input?.start_after === null || input?.start_after === undefined ? null : Number(input.start_after),
      limit: Math.max(1, Math.min(5000, Number(input?.limit || 1)))
    };
    return request;
  }

  eventRange(collection, eventIndex) {
    const values = collection.columns.event_index;
    return [lowerBound(values, eventIndex), upperBound(values, eventIndex)];
  }

  row(collection, index) {
    const result = {};
    for (const [name, values] of Object.entries(collection.columns)) {
      result[name] = jsonValue(values[index]);
    }
    return result;
  }

  rows(collection, indices, limit = 200) {
    return indices.slice(0, limit).map(index => this.row(collection, index));
  }

  eventIndices(collection, eventIndex) {
    const [start, end] = this.eventRange(collection, eventIndex);
    return Array.from({ length: end - start }, (_, offset) => start + offset);
  }

  value(collection, name, index, fallback = NaN) {
    const column = collection.columns[name];
    return column ? Number(column[index]) : fallback;
  }

  formulaEnvironment(collection, index, prefix = '') {
    const get = (name, fallback = NaN) => this.value(collection, `${prefix}${name}`, index, fallback);
    return {
      E: get('energy'),
      energy: get('energy'),
      px: get('px'),
      py: get('py'),
      pz: get('pz'),
      p: get('momentum'),
      momentum: get('momentum'),
      pt: get('pt'),
      m: get('mass'),
      mass: get('mass'),
      mass2: get('mass_squared'),
      scalar_pt_sum: get('pt_scalar'),
      pt_total: get('pt_scalar'),
      pt_scalar: get('pt_scalar'),
      charge: get('charge_sum'),
      charge_sum: get('charge_sum'),
      dr: get('dr'),
      dphi: get('dphi'),
      angle: get('opening_angle'),
      opening_angle: get('opening_angle')
    };
  }

  pairingIndices(collection, eventIndex) {
    return this.eventIndices(collection, eventIndex).filter(index =>
      !collection.columns.quad_is_best || this.value(collection, 'quad_is_best', index) === 1
    );
  }

  quadIndices(collection, eventIndex, channel) {
    const code = CHANNEL_CODES[channel];
    return this.eventIndices(collection, eventIndex).filter(index => {
      if (collection.columns.is_best && this.value(collection, 'is_best', index) !== 1) return false;
      return channel === 'all' || this.value(collection, 'channel', index) === code;
    });
  }

  choosePairing(aValues, bValues, indices, collection, request) {
    const valid = indices.map((_, local) => local).filter(local => finite(aValues[local]) && finite(bValues[local]));
    if (!valid.length) return null;
    const bestBy = (score, largest = false) => valid.reduce((best, local) => {
      if (best === null) return local;
      const candidate = score(local);
      const previous = score(best);
      return largest ? (candidate > previous ? local : best) : (candidate < previous ? local : best);
    }, null);

    if (request.pairing_rule === 'closest_distance' || request.pairing_rule === 'furthest_distance') {
      return bestBy(local => {
        const index = indices[local];
        return this.value(collection, 'pair_a_dr', index) + this.value(collection, 'pair_b_dr', index);
      }, request.pairing_rule === 'furthest_distance');
    }
    if (request.pairing_rule === 'closest_formula' || request.pairing_rule === 'furthest_formula') {
      return bestBy(local => Math.abs(aValues[local] - bValues[local]), request.pairing_rule === 'furthest_formula');
    }
    if ([
      'same_type_same_charge', 'same_type_different_charge',
      'different_type_same_charge', 'different_type_different_charge'
    ].includes(request.pairing_rule)) {
      const sameType = request.pairing_rule.startsWith('same_type') ? 1 : 0;
      const suffix = request.pairing_rule.endsWith('same_charge') ? 'is_ss' : 'is_os';
      return bestBy(local => {
        const index = indices[local];
        const a = this.value(collection, 'pair_a_is_sf', index) === sameType
          && this.value(collection, `pair_a_${suffix}`, index) === 1;
        const b = this.value(collection, 'pair_b_is_sf', index) === sameType
          && this.value(collection, `pair_b_${suffix}`, index) === 1;
        return Number(a) + Number(b);
      }, true);
    }
    if (request.pairing_rule === 'random') {
      const first = indices[0];
      const event = this.value(collection, 'event_index', first, 0);
      const quad = this.value(collection, 'quad_index', first, 0);
      return valid[crc32(`${event}:${quad}`) % valid.length];
    }
    return valid[0];
  }

  evaluateRequirements(request, row) {
    const results = [];
    for (const selected of request.requirements) {
      const definition = this.requirementMap.get(selected.id);
      if (!definition) throw new Error(`Unknown event requirement: ${selected.id}`);
      if (!definition.modes.includes(request.mode)) {
        throw new Error(`Requirement ${selected.id} is not valid in ${request.mode} mode`);
      }
      const threshold = selected.value === null ? Number(definition.default) : Number(selected.value);
      let missing = !row;
      const values = [];
      if (row) {
        for (const field of definition.fields) {
          const value = row[field];
          if (!finite(value)) {
            missing = true;
            break;
          }
          values.push(Number(value));
        }
      }
      let passed = false;
      if (!missing) {
        if (definition.comparison === 'required') {
          const expected = definition.expected || values.map(() => 1);
          passed = expected.length === values.length && values.every((value, index) => value === expected[index]);
        } else if (selected.id === 'quad_charge_zero') {
          passed = Math.abs(values[0]) <= threshold;
        } else if (selected.id === 'max_lepton_iso') {
          passed = values[0] < threshold || values[0] < 0;
        } else if (definition.comparison === '>') {
          passed = values.every(value => value > threshold);
        } else if (definition.comparison === '<') {
          passed = values.every(value => value < threshold);
        }
      }
      results.push({
        id: selected.id,
        label: definition.label,
        value: threshold,
        passed,
        available: !missing
      });
    }
    return { passed: results.every(item => item.passed), results };
  }

  candidateEvents(collection, mode, channel = 'all') {
    const key = `${mode}:${channel}`;
    if (this.candidateCache.has(key)) return this.candidateCache.get(key);
    const candidates = [];
    const events = collection.columns.event_index;
    for (let start = 0; start < collection.rows;) {
      const eventIndex = Number(events[start]);
      let end = start + 1;
      while (end < collection.rows && Number(events[end]) === eventIndex) end += 1;
      let eligible = false;
      for (let index = start; index < end && !eligible; index += 1) {
        eligible = mode === 'pairs'
          ? (!collection.columns.quad_is_best || this.value(collection, 'quad_is_best', index) === 1)
          : ((!collection.columns.is_best || this.value(collection, 'is_best', index) === 1)
            && (channel === 'all' || this.value(collection, 'channel', index) === CHANNEL_CODES[channel]));
      }
      if (eligible) candidates.push(eventIndex);
      start = end;
    }
    this.candidateCache.set(key, candidates);
    return candidates;
  }

  async formulaEvent(input) {
    const request = this.normalizeRequest(input);
    const sourceName = request.mode === 'pairs' ? 'PairingOptions' : 'QuadLeptons';
    const [events, leptons, source] = await Promise.all([
      this.loadCollection('Events'),
      this.loadCollection('Leptons'),
      this.loadCollection(sourceName)
    ]);
    if (request.event_index === null) {
      const candidates = this.candidateEvents(source, request.mode, request.channel);
      if (!candidates.length) throw new Error('No events match the selected mode and channel');
      request.event_index = candidates[Math.floor(Math.random() * candidates.length)];
    }
    const ast = parseFormula(request.formula);
    return this.eventDetail(request.event_index, request, ast, {
      events,
      leptons,
      pairings: request.mode === 'pairs' ? source : null,
      quads: request.mode === 'four' ? source : null
    });
  }

  eventDisplayUrl(event, eventIndex) {
    const template = this.manifest.event_display_url_template;
    if (!template) return '';
    return template
      .replaceAll('{run}', event.run ?? '')
      .replaceAll('{lumi}', event.luminosityBlock ?? '')
      .replaceAll('{event}', event.event ?? '')
      .replaceAll('{event_index}', eventIndex);
  }

  eventDetail(eventIndex, request, ast, loaded) {
    const { events, leptons, pairings, quads } = loaded;
    const eventRows = this.eventIndices(events, eventIndex);
    if (!eventRows.length) throw new Error(`No event_index ${eventIndex}`);
    const event = this.row(events, eventRows[0]);
    const leptonIndices = this.eventIndices(leptons, eventIndex);
    const allPairingIndices = pairings ? this.eventIndices(pairings, eventIndex) : [];
    const allQuadIndices = quads ? this.eventIndices(quads, eventIndex) : [];
    const payload = {
      event,
      leptons: this.rows(leptons, leptonIndices),
      quads: quads ? this.rows(quads, allQuadIndices) : [],
      pairings: pairings ? this.rows(pairings, allPairingIndices) : [],
      event_display_url: this.eventDisplayUrl(event, eventIndex)
    };

    if (request.mode === 'pairs') {
      const indices = this.pairingIndices(pairings, eventIndex);
      const aValues = indices.map(index => evaluateAst(ast, this.formulaEnvironment(pairings, index, 'pair_a_')));
      const bValues = indices.map(index => evaluateAst(ast, this.formulaEnvironment(pairings, index, 'pair_b_')));
      const choice = this.choosePairing(aValues, bValues, indices, pairings, request);
      const rows = this.rows(pairings, indices);
      rows.forEach((row, local) => {
        row.formula_pair_a = jsonValue(aValues[local]);
        row.formula_pair_b = jsonValue(bValues[local]);
        row.selected_by_rule = choice === local ? 1 : 0;
      });
      const chosenRow = choice === null ? null : rows[choice];
      const requirement = this.evaluateRequirements(request, chosenRow);
      payload.formula_pairings = rows;
      payload.passes_requirements = requirement.passed;
      payload.requirement_results = requirement.results;
      payload.selected_values = choice !== null && requirement.passed
        && finite(aValues[choice]) && finite(bValues[choice])
        ? [Number(aValues[choice]), Number(bValues[choice])]
        : [];
    } else {
      const indices = this.quadIndices(quads, eventIndex, request.channel);
      const values = indices.map(index => evaluateAst(ast, this.formulaEnvironment(quads, index)));
      const rows = this.rows(quads, indices);
      rows.forEach((row, local) => {
        row.formula_value = jsonValue(values[local]);
        row.selected_by_rule = local === 0 ? 1 : 0;
      });
      const chosenRow = rows[0] || null;
      const requirement = this.evaluateRequirements(request, chosenRow);
      payload.formula_quads = rows;
      payload.passes_requirements = requirement.passed;
      payload.requirement_results = requirement.results;
      payload.selected_values = values.length && finite(values[0]) && requirement.passed
        ? [Number(values[0])]
        : [];
    }
    return payload;
  }

  selectedValues(eventIndex, request, ast, pairings, quads) {
    if (request.mode === 'pairs') {
      const indices = this.pairingIndices(pairings, eventIndex);
      const aValues = indices.map(index => evaluateAst(ast, this.formulaEnvironment(pairings, index, 'pair_a_')));
      const bValues = indices.map(index => evaluateAst(ast, this.formulaEnvironment(pairings, index, 'pair_b_')));
      const choice = this.choosePairing(aValues, bValues, indices, pairings, request);
      if (choice === null) return [];
      const row = this.row(pairings, indices[choice]);
      const requirement = this.evaluateRequirements(request, row);
      return requirement.passed && finite(aValues[choice]) && finite(bValues[choice])
        ? [Number(aValues[choice]), Number(bValues[choice])]
        : [];
    }
    const indices = this.quadIndices(quads, eventIndex, request.channel);
    if (!indices.length) return [];
    const value = evaluateAst(ast, this.formulaEnvironment(quads, indices[0]));
    const row = this.row(quads, indices[0]);
    const requirement = this.evaluateRequirements(request, row);
    return requirement.passed && finite(value) ? [Number(value)] : [];
  }

  async formulaBatch(input) {
    const request = this.normalizeRequest(input);
    const sourceName = request.mode === 'pairs' ? 'PairingOptions' : 'QuadLeptons';
    const [events, source] = await Promise.all([
      this.loadCollection('Events'),
      this.loadCollection(sourceName)
    ]);
    const pairings = request.mode === 'pairs' ? source : null;
    const quads = request.mode === 'four' ? source : null;
    const eventIndices = events.columns.event_index;
    if (!eventIndices.length) throw new Error('No events loaded');
    let start = request.start_after === null ? 0 : upperBound(eventIndices, request.start_after);
    if (start >= eventIndices.length) start = 0;
    const selected = [];
    for (let offset = 0; offset < request.limit; offset += 1) {
      selected.push(Number(eventIndices[(start + offset) % eventIndices.length]));
    }

    const ast = parseFormula(request.formula);
    const values = [];
    let acceptedEvents = 0;
    for (const eventIndex of selected) {
      const eventValues = this.selectedValues(eventIndex, request, ast, pairings, quads);
      if (eventValues.length) {
        acceptedEvents += 1;
        values.push(...eventValues);
      }
    }
    return {
      mode: request.mode,
      formula: request.formula,
      requested_events: selected.length,
      accepted_events: acceptedEvents,
      values,
      last_event_index: selected.length ? selected[selected.length - 1] : request.start_after
    };
  }
}

export { parseFormula, evaluateAst };
