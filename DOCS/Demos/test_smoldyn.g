//moose
include kkit_enz_1000mol.g

create neutral /kinetics/geometry
create KinCompt /kinetics/geometry/cytoplasm
create KinCompt /kinetics/geometry/membrane
create KinCompt /kinetics/geometry/nuclearMembrane
create KinCompt /kinetics/geometry/nucleus

create Surface /kinetics/geometry/surface0
ce ^
	create HemispherePanel cap1
	create CylPanel cyl
	create HemispherePanel cap2
	addmsg cap1/neighborSrc cyl/neighbor
	addmsg cap2/neighborSrc cyl/neighbor
ce /

create Surface /kinetics/geometry/surface1
ce ^
create SpherePanel nucleus
ce /

setfield /kinetics method Smoldyn
reset
step {MAXTIME} -t
do_save_all_plots smol.plot
