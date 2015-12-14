#!/bin/sh
# 
# Script to quickly profile a single network, produce a call graph,
# and display the top pstats results.
#
# The network should be passed in as "5-AND-circle", "6-MAJ-complete", etc.

NETWORK=$1

python code_to_profile.py $NETWORK
./loadprofile.sh pstats/$NETWORK.pstats
./makecallgraph
python print_stats.py
open -a Firefox callgraph.svg
