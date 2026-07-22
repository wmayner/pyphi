Campaign directories store substrates as `substrate-<label>.msgpack.gz`
instead of `.json.gz`. msgpack writes the npy-encoded TPM factors as raw
bytes rather than base64 text, shrinking the file and removing the
base64 encode/decode cost on every job. Existing campaign directories
are unaffected in practice: `collect` already refuses outputs from a
different PyPhi version, so campaigns are re-prepared on upgrade.
