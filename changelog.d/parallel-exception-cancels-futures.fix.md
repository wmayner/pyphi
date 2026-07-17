A worker exception during parallel evaluation now cancels the remaining
pending chunks instead of leaving them running — orphaned chunks previously
kept burning CPU in the shared process pool (delaying the next parallel
computation) and forced the thread backend to block until every orphaned
task had finished.
