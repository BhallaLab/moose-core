call music addPort out event UTDATA
create TimeTable tt
call /tt load "testtider.txt" 0

addmsg /tt/state /music/UTDATA/channel[0]

reset

step 1 -t 
