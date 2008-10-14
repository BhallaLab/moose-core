call music addPort out event UTDATA
create TimeTable tt
call /tt load "testtider.txt" 0

addmsg /tt/event /music/UTDATA/channel[0]/event

setclock 2 0.01
useclock /music 2
useclock /music/UTDATA 2

reset
call /music reinitialize

step 1 -t 

quit
