// This is a test-script to show the functionality of Music
//

call music addPort in event INDATA
create AscFile /output
setfield /output fileName "utdata.txt"
setfield /music/INDATA accLatency 0.1
setfield /music/INDATA maxBuffered 100

addmsg /music/INDATA/channel[0]/event /output/save

setclock 2 0.01
useclock /music 2
useclock /music/INDATA 2

reset
call /music reinitialize

step 0.8 -t
quit
