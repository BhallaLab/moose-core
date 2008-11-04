//moose
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

