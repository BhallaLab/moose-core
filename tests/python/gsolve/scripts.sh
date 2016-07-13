#!/bin/bash
for i in {1..10}
do
        python camkii_pp1_scheme.py
        diff camkii_pp1_scheme.py.dat camkii_pp1_scheme.py1.dat
done
