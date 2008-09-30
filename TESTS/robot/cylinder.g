//genesis
// moose
// This script tests how the sigNeur can generate a 1-d diffusion model
// It is currently a 50-compartment model with total length 100 microns
// and diffusion constant of 1e-12 m^2/sec.
// Currently only tests for conservation following diffusion. Later
// to add analytic solution for comparison.

float PI = 3.1415926535
float RUNTIME = 10.0
int nmol = 50
float lambda = 2.000001e-6
float D = 1e-12
float length = 0.0001

// Create a library of prototype elements to be used by the cell reader
create KinCompt /library/soma
create Molecule /library/soma/Ca
setfield /library/soma/Ca D 1
create KinCompt /library/dend
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
setfield /sig lambda {lambda}
setfield /sig Dscale {D}

showfield /sig *
le /sig

setfield /sig build foo
// int nmol = {getfield /sig/kinetics/solve/hub nMol}
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

step {RUNTIME} -t

echo done steps
showfield /sig/kinetics/soma[]/Ca n
// showfield /sig/kinetics/soma[12]/Ca n

function diffn( N, D, x, t )
	float N
	float D
	float x
	float t
	
	float ret = lambda * ( N / { sqrt { 4 * PI * D * t } } ) * { exp {-( x * x ) / ( 4 * D * t ) } }
	return {ret}
end

float tot = 0
float tot2 = 0
float ret = 0

openfile test.plot w
writefile test.plot /newplot
writefile test.plot /plotname theory
int i
for ( i = 0; i < nmol; i = i + 1 )
	tot = tot + {getfield /sig/kinetics/soma[{i}]/Ca n}
	ret = { diffn 1 {D} { ( i - ( nmol - 1 )/2 ) * lambda} {RUNTIME} }
	tot2 = tot2 + ret

	// echo { ret } {getfield /sig/kinetics/soma[{i}]/Ca n}
	writefile test.plot { ret }
end

writefile test.plot /newplot
writefile test.plot /plotname simulation
for ( i = 0; i < nmol; i = i + 1 )
	writefile test.plot {getfield /sig/kinetics/soma[{i}]/Ca n}
end
closefile test.plot

echo tot = {tot}, tot2 = {tot2}
