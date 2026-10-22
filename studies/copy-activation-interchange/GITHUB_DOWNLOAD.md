# Download and reconstruct the complete study

Read [the scientific report](report/REPORT.md) and [the main README](README.md)
for findings and limits. The readable source, report, figures and strict
reconstructor here are byte-identical to `source-report-tools.zip`.

The two evidence ZIPs are stored as ordered binary parts, each at most 512 KiB,
so supported GitHub file transport can preserve the complete evidence. This
changes packaging only: no scientific byte, failed case, vector or manifest is
removed or changed. `GITHUB_TRANSPORT.json` records every part's order, length,
SHA-256 and Git blob SHA-1, plus each assembled ZIP's length and SHA-256.

Clone or download this repository at the desired commit, then enter
`studies/copy-activation-interchange`. Do not concatenate GitHub web-page HTML;
use the actual downloaded binary files. The original evidence ZIPs are not
separate repository files: assemble them from the parts below.

## Verify and assemble on Linux or macOS

In a trusted, otherwise-idle workspace, first verify every part. On Linux:

```sh
sha256sum -c PARTS.sha256
```

On macOS, use `shasum -a 256 -c PARTS.sha256` instead. Stop if any check fails.
Then assemble in zero-padded numeric filename order (start in a fresh working
copy without already assembled evidence ZIPs; redirection replaces that filename):

```sh
export LC_ALL=C
cat evidence-parts/evidence-blobs-1.zip.part-[0-9][0-9][0-9][0-9] > evidence-blobs-1.zip
cat evidence-parts/evidence-blobs-2.zip.part-[0-9][0-9][0-9][0-9] > evidence-blobs-2.zip
sha256sum -c ARCHIVES.sha256
```

For the last command on macOS use `shasum -a 256 -c ARCHIVES.sha256`.
The archive check also verifies the retained source ZIP and reconstruction map.
Do not proceed unless all checks pass. These checks detect missing, extra,
reordered or changed assembled part bytes by the assembled archive hashes;
the part inventory is also available for individual-file diagnosis.

## Reconstruct original scientific files

Use Python 3 and the existing standard-library-only strict reconstructor:

```sh
python3 -B tools/reconstruct.py --map RECONSTRUCTION.json \
  --map-sha256 8266d46429204e8bd7cde222e57f74e779223dac5b673e4caae7b9800f280de8 \
  --zips evidence-blobs-1.zip evidence-blobs-2.zip \
  --out NEW_RECONSTRUCTION_DIRECTORY --scientific-phase-links
```

The output parent must already exist and the output directory must not exist.
On failure retain the fresh partial directory for inspection and use a different
fresh directory for a retry. The expected result is 3,294 files totaling
577,232,014 bytes, with exact per-file hashes and original phase linkage.
This is a lossless evidence roundtrip, not a rerun of model inference.
Model weights, the numerical runtime and optional plotting dependencies remain
external as declared in the unchanged source documentation.
