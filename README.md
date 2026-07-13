# Particle Physics Analysis Tool

Current version: **3.2.8**

## Update an existing copy

The versioned update archive is intentionally flat. From the existing project
directory, extract it directly over the top-level files:

```bash
cd ~/CMS_Four_Lepton_Lab_webapp_v3.1
unzip -o CMS_Four_Lepton_Lab_webapp_v3.2.8_FLAT_UPDATE.zip -d .
chmod +x startup.sh
./startup.sh
```

This replaces `app.py`, `startup.sh`, `requirements.txt`, and `static/` while
leaving the ROOT and JSON files already inside `data/` intact. The old
`webapp/` directory is not used by version 3.2.8.

## Run

1. Put the reduced ROOT file at `data/zz4l_masterclass.root`, or set
   `MASTERCLASS_ROOT` to its location.
2. Make the launcher executable once, then run it:

   ```bash
   chmod +x startup.sh
   ./startup.sh
   ```

3. Open <http://localhost:8000>.

At startup, the terminal must print `Particle Physics Analysis Tool v3.2.8`.
The webpage status badge must show `v3.2.8`. If either version is
different, an older file is still being launched. After replacing an older
copy, use a hard refresh (`Ctrl+Shift+R`) once.

The launcher creates `.venv` when needed, installs the top-level
`requirements.txt`, validates both data files, and starts the top-level
`app.py`. Set `PORT`, `MASTERCLASS_ROOT`, or `MASTERCLASS_METADATA` to override
their defaults.

The app itself also checks for `../zz4l_masterclass.root` to remain compatible
with the earlier directory layout when it is started without `startup.sh`.

## Student workflow

- **Z search:** define a two-lepton quantity and compare nine simple pairing
  strategies. Both values from the selected pairing enter the histogram.
- **H search:** apply the same formula builder to the complete four-lepton
  candidate and choose all, 2e2μ, 4e, or 4μ events.
- **Event requirements:** choose structural or numerical cuts from one list,
  set numerical thresholds before adding them, and edit those thresholds later.
  Current events and batches are accepted only when every active requirement
  passes.

Changing the formula, mode, pairing strategy, channel, or event requirements
clears the accumulated histogram so incompatible configurations are never
mixed.
