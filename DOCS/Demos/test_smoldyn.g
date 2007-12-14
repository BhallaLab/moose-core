//moose
include kkit_enz_1000mol.g
MAXTIME = 20

create neutral /kinetics/geometry
create KinCompt /kinetics/geometry/cytoplasm
create KinCompt /kinetics/geometry/membrane
create KinCompt /kinetics/geometry/nuclearMembrane
create KinCompt /kinetics/geometry/nucleus

addmsg /kinetics/geometry/cytoplasm/outside /kinetics/geometry/membrane/inside
addmsg /kinetics/geometry/nuclearMembrane/outside /kinetics/geometry/cytoplasm/inside
addmsg /kinetics/geometry/nucleus/outside /kinetics/geometry/nuclearMembrane/inside

create Surface /kinetics/geometry/surface0
ce ^
// 	showfield cap1 x[0]
// 	showfield cap1 x[1]
// 	showfield cap1 x[2]

// 	showfield cyl x[0]
// 	showfield cyl x[1]
// 	showfield cyl x[2]

	create HemispherePanel cap2
	setfield cap2 x[0] -1.5e-6
	setfield cap2 y[0] 0
	setfield cap2 z[0] 0
	setfield cap2 x[1] 0.5e-6
	setfield cap2 y[1] 0
	setfield cap2 z[1] 0
	setfield cap2 x[2] 1 // This is outward vector
	setfield cap2 y[2] 0
	setfield cap2 z[2] 0
	setfield cap2 x[3] 1 // This is front: pointing outward.
	setfield cap2 y[3] 0
	setfield cap2 z[3] 0

	create CylPanel cyl
	setfield cyl x[0] 1.5e-6
	setfield cyl y[0] 0
	setfield cyl z[0] 0
	setfield cyl x[1] -1.5e-6
	setfield cyl y[1] 0
	setfield cyl z[1] 0
	setfield cyl x[2] 0.5e-6
	setfield cyl z[3] 1


	create HemispherePanel cap1
	setfield cap1 x[0] 1.5e-6
	setfield cap1 y[0] 0
	setfield cap1 z[0] 0
	setfield cap1 x[1] 0.5e-6
	setfield cap1 y[1] 0
	setfield cap1 z[1] 0
	setfield cap1 x[2] -1 // This is outward vector
	setfield cap1 y[2] 0
	setfield cap1 z[2] 0
	setfield cap1 x[3] 1 // This is front: pointing outside.
	setfield cap1 y[3] 0
	setfield cap1 z[3] 0


// 	showfield cap2 x[0]
// 	showfield cap2 x[1]
// 	showfield cap2 x[2]

	addmsg cap1/neighborSrc cyl/neighbor
	addmsg cap2/neighborSrc cyl/neighbor
ce /

create Surface /kinetics/geometry/surface1
ce ^
create SpherePanel nucleus
	setfield nucleus x[0] 0
	setfield nucleus y[0] 0
	setfield nucleus z[0] 0
	setfield nucleus x[1] 0.45e-6
	setfield nucleus x[2] 1 // Front: pointing outside
	setfield nucleus y[2] 0
	setfield nucleus z[2] 0

	showfield nucleus x[0]
	showfield nucleus x[1]
ce /

addmsg /kinetics/geometry/surface0/surface /kinetics/geometry/membrane/exterior
addmsg /kinetics/geometry/surface0/surface /kinetics/geometry/cytoplasm/exterior
addmsg /kinetics/geometry/surface1/surface /kinetics/geometry/cytoplasm/interior
addmsg /kinetics/geometry/surface1/surface /kinetics/geometry/nuclearMembrane/exterior
addmsg /kinetics/geometry/surface1/surface /kinetics/geometry/nucleus/exterior

setfield /kinetics method Smoldyn
setfield /kinetics/solve/hub seed 1197006306
int i
for ( i = 0; i < 1000; i = i + 1 )
	setfield /kinetics/S x[{i}] -1e-6 y[{i}] 0 z[{i}] 0
	setfield /kinetics/E x[{i}] 1e-6 y[{i}] 0 z[{i}] 0
end

reset
echo starting
step {MAXTIME} -t
do_save_all_plots smol.plot
/*
*/
