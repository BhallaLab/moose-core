echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(Bug Id: 2116030)
Showmsg crashes if a target finfo is a DynamicFinfo.

(Very similar to dynFinfo.g)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

create Compartment cc
create Table tab
addmsg tab/outputSrc cc/Vm
showmsg tab
