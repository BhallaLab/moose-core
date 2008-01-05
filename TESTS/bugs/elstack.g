// moose
// Element stack not functional

create Compartment cc
pushe /cc

echo
echo "** After 'pushe /cc': **"

echo
echo "** pwe: **"
pwe

echo
echo "** Create element 'spike', followed by le: **"
create SpikeGen spike
le

echo
echo "** Add message: **"
addmsg ./VmSrc spike/Vm

echo
echo "** Show fields: **"
showfield . *
