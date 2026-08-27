Disk-cache writes now use a per-call temporary filename, so two threads writing the same key no longer collide on a shared temp path (which raised `FileNotFoundError`).
