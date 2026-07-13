Fixed the documentation build on case-insensitive filesystems: the
`pyphi.mcp.content` module defines both a `TOPICS` constant and a `topics()`
function, whose autosummary stub filenames collided. They are now mapped to
distinct filenames in `autosummary_filename_map`, alongside the existing
`pyphi.relations` entries.
