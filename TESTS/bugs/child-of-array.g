echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(Bug Id: 2116557)

When SimpleElement::isDescendant is called to find out if A is a child of B, then
MOOSE crashes if A is a SimpleElement and B is an ArrayElement.

This is encountered in the Biophysics solver, because the above call is made
implicitly during autoscheduling.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

ce /library
create Cell cell
create Compartment cell/compt
ce /

createmap /library/cell / 1 5
reset
