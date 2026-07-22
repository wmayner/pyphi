Serialized substrates store only the on-probability slice of each binary
TPM factor when its off slice is the exact float complement (verified
elementwise at encode time; factors failing the check, and all
non-binary factors, are stored in full). This halves the dominant term
in the file size: a dense binary substrate now serializes at ~1.0× its
raw state-by-node array (gzip takes it below), where it was ~2×. The
serialization format version is 2; older PyPhi versions refuse to load
new files rather than misread them.
