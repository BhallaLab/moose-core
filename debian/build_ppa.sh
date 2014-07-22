#!/usr/bin/env bash
# This script generates files for ppa.
# Dilawar Singh <dilawars@ncbs.res.in>
# Friday 18 July 2014 12:10:02 PM IST
set -e

# Just clean the svn repo so I can create a fresh tar ball
(
    cd ../../
    svn status --no-ignore | grep '^[I?]' | cut -c 9- | while IFS= read -r f; do rm -rf "$f"; done
)

MAJOR=1
MINOR=1
TARNAME=moose_$MAJOR.$MINOR.orig.tar.gz
rm -rf $TARNAME
tar -avcf $TARNAME \
    --exclude .git \
    --exclude .svn \
    --exclude gsl \
    --exclude debian \
    --exclude "*.so" \
    --exclude "*.o" \
    --exclude "*~" \
    --exclude "moose" \
    ../..
(
    cd .. && debuild --check-dirname-level=0 -S -sa
)
