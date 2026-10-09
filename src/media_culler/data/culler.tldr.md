# culler

> Score, group and cull photo collections (technical quality, AI aesthetics, faces, duplicates).
> More information: `culler --help`.

- Analyse a folder (writes `reports/inventory.json` and `.csv`):

`culler run {{path/to/photos}}`

- Fast technical-only pass (no AI models, no faces):

`culler run {{path/to/photos}} --no-ai --no-faces`

- Re-run after tweaking weights/thresholds (analysis is cached, so this is near-instant):

`culler run {{path/to/photos}} --config {{culler.yaml}}`

- Machine-readable summary for scripts:

`culler run {{path/to/photos}} --json`

- Preview, then apply, sorting photos into folders by verdict:

`culler organize reports/inventory.json --dest {{path/to/sorted}}`

`culler organize reports/inventory.json --dest {{path/to/sorted}} --apply`

- Re-display the last report without re-analysing:

`culler report`

- Download the face-landmark model, then check your setup:

`culler models download`

`culler doctor`

- Write a commented starter config:

`culler config init`
