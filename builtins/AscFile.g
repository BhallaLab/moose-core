create AscFile a
setfield a fileName "utdata2.txt"

create TimeTable tt
call /tt load "testtider.txt" 0

addmsg /tt/state /a/save

setclock 0 0.1 0

reset
reset

step 10 -t
