import pstats

"""
Prints the pstats results for the functions in which the most
time was spent.

Use `load_profile.sh` to load a pstats file into `profile.pstats`.
TODO: take the pstats file as command line arg.
"""

FILE = "profile.pstats"

p = pstats.Stats(FILE)
p.strip_dirs()
p.sort_stats('tottime', 'calls', 'name')
p.print_stats(25)
