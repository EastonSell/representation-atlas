# Copy-target activation interchange: complete evidence transport

Start with `report/REPORT.md` and `report/transfer-map.png`. The unmodified frozen
scientific source/data snapshot is under `source/`. Its status and launch-plan
files are historical pre-execution records, not claims that the completed run is
still pending. Full scientific output is carried by the two companion blob ZIPs.

## Lossless reconstruction

All 3,294 files from the original baseline and patch phase directories are
represented in RECONSTRUCTION.json. The original 577,232,014 bytes deduplicate to
1,305 unique byte strings. Deduplication does not deserialize or round floats:
identical bytes are stored once under their SHA-256. Every original relative
path and SHA-256 is retained, including full-vocabulary vectors, failed baseline
row 58/pair 29, controls, metadata, logs, summaries and original manifests.
No scientific phase file was omitted or redacted, and no original manifest was
rewritten. Generic runtime hardware/software details and the original warning
about an unwritable default cache path are preserved as execution provenance.

Private coordination, authorization and handoff records were always stored
outside these scientific phase directories and are not included. Neither model
weights nor an installed Python environment is bundled. The exact model revision,
weight hash and runtime versions are declared in the release manifest and source.

Obtain the trusted reconstruction-map SHA-256 from the independently reviewed
release index, not from an untrusted map's self-report. With the files in this
bundle extracted, run in a trusted otherwise-idle local workspace:

```sh
python tools/reconstruct.py --map RECONSTRUCTION.json \
  --map-sha256 8266d46429204e8bd7cde222e57f74e779223dac5b673e4caae7b9800f280de8 \
  --zips evidence-blobs-1.zip evidence-blobs-2.zip \
  --out NEW_RECONSTRUCTION_DIRECTORY --scientific-phase-links
```

The parent directory must already exist; the output directory must not exist.
The reconstructor verifies strict map syntax, exact blob coverage, hashes/sizes,
regular-file members and path safety before writing original files. It reserves
a fresh directory and retains that directory, including any partial output, on
failure. It never overwrites an existing run. It verifies the full reconstructed
file set and original inner/outer acceptance/upstream manifest linkage. This is
byte-level reproducibility, not a new model execution. Its workspace checks are
not a sandbox against concurrent hostile filesystem mutation.

Read-only re-verification:

```sh
python tools/reconstruct.py --map RECONSTRUCTION.json \
  --map-sha256 8266d46429204e8bd7cde222e57f74e779223dac5b673e4caae7b9800f280de8 \
  --out RECONSTRUCTED_DIRECTORY --verify-only --scientific-phase-links
```

`tools/test_reconstruct.py` contains small stdlib-only synthetic tests. These
establish selected transport properties, not scientific reproduction. Source
unit-test evidence is preserved in `source/TEST_RESULTS.txt`. Recomputing
scientific summaries/audits uses the original source and pinned numerical
runtime; new model execution is separate from transporting these retained bytes.
See the source protocol for budgets, conditions and limitations. Plotting and
reporting scripts use retained numerical summaries only; Matplotlib is optional
for re-rendering and is not installed by this package.

Each blob ZIP has less than 100,000,000 uncompressed bytes so the unchanged release
validator can check it independently. Exact external manifests describe archive
contents; no publication or human approval is implied by packaging.
