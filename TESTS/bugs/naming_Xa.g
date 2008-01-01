/*
Field access in objects named Xa, where X can be any char, do not work except in root. This causes problems in /library/Na and /library/Ca.
*/

function makeC (path)
	str path
	create Compartment {path}
	setfield {path} x 10
end

makeC Na
echo "Compartment /Na created with field x set to 10 - works"
showfield Na x

create Neutral n
ce n 

makeC Xa
echo "Compartment /n/Na created with field x set to 10 - does not work"
showfield Xa x

makeC Na1
echo "Compartment /n/Na1 created with field x set to 10 - work"
showfield Na1 x
