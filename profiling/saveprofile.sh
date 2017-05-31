#!/bin/sh

outputfile="profile.pstats"
path="profile_data/"
prefix="profile"
sep="."
extension=".pstats"
if [ -z $1 ]
then
    filename="`date +%s`"
else
    filename=$1
fi
save=$path$prefix$sep$filename$extension
echo "Saved '$outputfile' to '$save'"
cp $outputfile $save
