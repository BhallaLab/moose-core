// This is a test-script to show the functionality of Music
//

call music addPort in event INDATA
create AscFile output fileName "utdata.txt"

addmsg /music/INDATA/channel[0] /output/save

reset

step 1 -t
