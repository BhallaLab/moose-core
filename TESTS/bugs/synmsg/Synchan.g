// moose

include defaults.g
include chan.g

ce /library
make_Na_mit_usb
make_K_mit_usb
ce /

readcell axon.p /cell

ce /cell/soma
create SynChan syn
// addmsg ./channel syn/channel

addmsg . syn VOLTAGE Vm
addmsg syn . CHANNEL Gk Ek

ce /

showmsg /cell/soma
