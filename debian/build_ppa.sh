#!/usr/bin/env bash
# This script generates files for ppa.
# Dilawar Singh <dilawars@ncbs.res.in>
# Friday 18 July 2014 12:10:02 PM IST
set -e
shopt -s extglob
(
    cd ../buildMooseUsingCmake
    rm -rf !(*.sh)
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
    ../
debuild --check-dirname-level=0 -S -sa
