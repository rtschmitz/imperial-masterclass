# Verification performed for version 4.0.0

The release was checked against the supplied 79 MiB ROOT file and version 3.2.8
FastAPI implementation.

- All JavaScript modules passed syntax checking.
- The static-data exporter completed with all 34 event requirements available.
- SHA-256 and decompressed-size checks passed for all four generated datasets.
- Every DOM identifier referenced by `app.js` exists in `index.html`.
- Five representative event configurations matched the Python backend for
  formula values, selected pairing, lepton count, and requirement results.
- Pair-mode and four-lepton-mode 1,000-event batches matched the Python backend
  for accepted-event counts, last event index, number of values, and values.
- Candidate-event counts matched for pair mode, all four-lepton events, 4e,
  4μ, and 2e2μ.
- Unsafe formula syntax was rejected by the browser parser.
- The application and data loaded successfully beneath a simulated
  `/~rschmitz/` URL rather than only at a domain root.
- `deploy.sh` produced a byte-identical copy of the `site/` directory.

Floating-point comparisons used a relative tolerance of 2e-5 because the
source branches are primarily 32-bit floating-point arrays.
