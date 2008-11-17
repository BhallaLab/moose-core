@echo off

set MOOSE=..\..\moose_vcpp2005\debug\moose

set NEARDIFF=..\regression\neardiff

:: N stands for Number of Nodes
set N=1
:CC
IF (%N%)==(16) GOTO DD
ECHO element_manipulation(%N%)
mpiexec -n %N% %MOOSE% element_manipulation.g %N%
ECHO showField(%N%)
mpiexec -n %N% %MOOSE% showField.g %N%
SET /A N=%N% * 2
GOTO CC

:DD
ECHO Done with regression tests
