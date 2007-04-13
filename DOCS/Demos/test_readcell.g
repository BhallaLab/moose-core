//moose

include bulbchan.g

create neutral /library

ce /library
make_LCa3_mit_usb
make_Na_rat_smsnn
make_KA_bsg_yka
make_KM_bsg_yka
make_K_mit_usb
make_K2_mit_usb
make_Na_mit_usb
// make_Kca_mit_usb
make_Ca_mit_conc
ce /

readcell mit.p /mit
