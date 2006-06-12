//moose

include kholodenko.g
// include kkit_reac.g
// include kkit_enz.g
// include kkit_MMenz.g

create Stoich /s
setfield /s path "/kinetics/##"

create ForwardEuler /fe
addmsg /fe/integrate /s/integrate

useclock /fe 4

reset

step {MAXTIME} -t

