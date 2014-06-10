#!/bin/bash
find . -type f -regex ".*\.\(sh\|py\|Make*\|nw\|xml\|xsd\)" -print0 | xargs -0 -I file git add file 
git commit -m "$1"
