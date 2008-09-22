//genesis
// moose
// This script tests how the sigNeur can generate a 1-d diffusion model
// It is currently a 50-compartment model with total length 100 microns
// and diffusion constant of 1e-12 m^2/sec.
// Currently only tests for conservation following diffusion. Later
// to add analytic solution for comparison.

// Create a library of prototype elements to be used by the cell reader
create neutral /library/soma
create Molecule /library/soma/Ca
setfield /library/soma/Ca D 1
create neutral /library/dend
create Molecule /library/dend/Ca
setfield /library/dend/Ca D 1

create neutral /library/cell
create symcompartment /library/cell/soma
setfield /library/cell/soma Rm 3e8 Ra 2.2e6 diameter 10e-6 length 0.0001

create SigNeur /sig
setfield /sig cellProto /library/cell
setfield /sig somaProto /library/soma
// setfield /sig dendProto /library/dend
//setfield /sig somaMethod ee
//setfield /sig dendMethod ee
//setfield /sig spineMethod ee
setfield /sig lambda 2.0001e-6
setfield /sig Dscale 1e-12

showfield /sig *
le /sig

setfield /sig build foo
// int nmol = {getfield /sig/kinetics/solve/hub nMol}
int nmol = 50
echo nmol = {nmol}
// setfield /sig/kinetics method ee

/*
int i
for ( i = 0; i < 24; i = i + 1 )
	showmsg /sig/kinetics/soma[{i}]/Ca
	showfield /sig/kinetics/soma[{i}]/Ca mode
end
*/

reset

setfield /sig/kinetics/soma[{(nmol - 1 )/2}]/Ca n 1 nInit 1
//showfield /sig/kinetics/soma[11]/Ca n

step 10.0 -t

echo done steps
showfield /sig/kinetics/soma[]/Ca n
// showfield /sig/kinetics/soma[12]/Ca n
float tot = 0;
int i
for ( i = 0; i < nmol; i = i + 1 )
	tot = tot + {getfield /sig/kinetics/soma[{i}]/Ca n}
end
echo tot = {tot}
