# Imperial CMS Masterclass — browser-only deployment

This release performs all event selection, pairing, formula evaluation, and
histogram preparation inside each student's browser. The HEP web server only
delivers static files.

There is no Python process, FastAPI application, ASGI/CGI entry point, PHP,
database, upload endpoint, or user-written code running on the server.

## Layout

```text
client-masterclass/
├── site/                    files that go in public_html
│   ├── index.html
│   ├── static/
│   │   ├── app.js           existing interface, connected to the worker
│   │   ├── client-api.js    promise-based worker interface
│   │   ├── data-worker.js   worker message dispatcher
│   │   ├── data-engine.js   selections, pairing and safe formula engine
│   │   ├── styles.css
│   │   └── plotly-2.35.2.min.js
│   └── data/
│       ├── manifest.json
│       └── *.dat            gzip-compressed typed arrays
├── tools/build_static_data.py
├── config/requirements.json
├── source-assets/           metadata used by the offline build
├── requirements-build.txt
├── serve-local.sh
└── deploy.sh
```

The original 79 MiB ROOT file is not deployed. The build exports only the
branches used by the interface. The resulting four compressed data files total
about 29 MiB and are loaded lazily: pair mode does not download the four-lepton
collection until the student opens that mode.

## Test locally

Browser modules and Web Workers must be loaded over HTTP, not by double-clicking
`index.html`.

```bash
./serve-local.sh
```

Open <http://localhost:8000/>. To use another port:

```bash
PORT=8080 ./serve-local.sh
```

This command uses Python's static file server only. It does not execute any
application code on the server.

## Deploy on the HEP homes server

Copy the project to an LX machine, then choose one of these locations.

For a subdirectory:

```bash
./deploy.sh "$HOME/public_html/masterclass"
```

The URL will be:

```text
https://homes.hep.ph.ic.ac.uk/~rschmitz/masterclass/
```

To place the application directly at the top of the personal site:

```bash
./deploy.sh "$HOME/public_html"
```

The URL will then be:

```text
https://homes.hep.ph.ic.ac.uk/~rschmitz/
```

`deploy.sh` does not delete pre-existing files. Review `~/public_html` first if
you already host anything at the top-level URL.

## Rebuild after changing the ROOT file

The deployed site does not need Python, NumPy, or Uproot. They are needed only
on the development machine when producing new static data.

```bash
python3 -m venv .build-venv
source .build-venv/bin/activate
python -m pip install -r requirements-build.txt

python tools/build_static_data.py \
  /path/to/zz4l_masterclass.root \
  source-assets/zz4l_masterclass.metadata.json
```

The command replaces `site/data/manifest.json` and the four `site/data/*.dat`
files. Upload the complete `site/` directory afterward so the interface and
data versions cannot become mismatched.

## Security model

The formula parser is a small recursive-descent parser. It accepts numerical
constants, known physics variables, parentheses, arithmetic operators, and the
explicit functions `sqrt`, `abs`, `log`, `exp`, `sin`, `cos`, `tan`, `min`,
and `max`. It does not call JavaScript `eval`, `Function`, a shell, Python, or a
remote service.

All datasets are read-only browser arrays. Reloading the page discards the
student's accumulated histogram and restores the original static data.

## Browser support

The data files use the standard browser `DecompressionStream` API. Use a current
version of Chrome, Edge, Firefox, or Safari. The interface reports a clear error
instead of silently failing if that API is unavailable.
