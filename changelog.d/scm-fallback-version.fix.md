Builds from source trees without git metadata (such as GitHub "Download ZIP"
exports) now fall back to a placeholder version instead of failing with a
setuptools-scm error. Proper sdists and normal git builds are unaffected.
