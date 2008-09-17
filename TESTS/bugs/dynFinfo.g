echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(Bug Id: 2116030)
Showmsg crashes if a target finfo is a DynamicFinfo.

(Also look at dynFinfo1.g)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

create Compartment cc
create Table tab
addmsg tab/inputRequest cc/Vm
showmsg tab
