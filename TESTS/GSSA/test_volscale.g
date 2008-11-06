//moose

/*
include reac2.g
showfield /kinetics volume
showfield /kinetics/A volumeScale nInit concInit
// showfield /kinetics/B volumeScale nInit concInit
showfield /kinetics/kreac kf kb Kf Kb 

setfield /kinetics volume 1.66667e-18

showfield /kinetics volume
showfield /kinetics/A volumeScale nInit concInit
// showfield /kinetics/B volumeScale nInit concInit
showfield /kinetics/kreac kf kb Kf Kb 

*/

include acc4_1.6e-21.g 
showfield /kinetics volume
showfield /kinetics/MAPK volume

showfield /kinetics/##[TYPE=Molecule] volumeScale
showfield /kinetics/##[TYPE=Reaction] kf kb Kf Kb

setfield /kinetics volume 1.66667e-18
showfield /kinetics volume
showfield /kinetics/MAPK volume
showfield /kinetics/##[TYPE=Reaction] kf kb Kf Kb
