#!/bin/sh

moose Traub91.g passive
moose Traub91.g reference
moose Traub91.g A0
moose Traub91.g B0
moose Traub91.g C0
moose Traub91.g D0
moose Traub91.g A1
moose Traub91.g B1
moose Traub91.g C1
moose Traub91.g D1

genesis Traub91.g passive
genesis Traub91.g reference
genesis Traub91.g A0
genesis Traub91.g B0
genesis Traub91.g C0
genesis Traub91.g D0
genesis Traub91.g A1
genesis Traub91.g B1
genesis Traub91.g C1
genesis Traub91.g D1
