#!/bin/bash 
if [ $# -lt 1 ]; then
  echo "USAGE : ./generateDS.py schame.xsd"
  exit
fi
schamaName="$1"
name=$(basename $schamaName)
name=${name/".xsd"/".py"}
echo "Generating $name ..."
generateDS.py -f -o $name $1 
echo ".. Done"
