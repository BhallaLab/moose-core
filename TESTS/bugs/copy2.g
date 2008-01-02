/*Not able to duplicate the gate message when successive copy of the channel is made. Such copying is often used. /library has prototype channels. readcell copies the prototypes in library and create the prototype neuron. When this prototype is copied using createmap or copy, channel messages are no where to be found. */

create HHChannel h
setfield h Xpower 1
copy h n1
copy n1 n2
showmsg h
showmsg n1
showmsg n2 

