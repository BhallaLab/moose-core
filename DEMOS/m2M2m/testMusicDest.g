// This is a test-script to show the functionality of Music
//

if ( ! {exists /music} )
	echo "Error! This MOOSE does not have MUSIC support."
	echo "Install a MOOSE package which supports the MUSIC library, and try again."
	echo "Will now exit..."
	quit
end

call /music addPort in event INDATA
create AscFile /output
setfield /output fileName "utdata.txt"
setfield /music/INDATA accLatency 0.1
setfield /music/INDATA maxBuffered 100

addmsg /music/INDATA/channel[0]/event /output/save

setclock 0 0.01 0
setclock 1 0.01 1
useclock /music/INDATA 0
useclock /music 1

reset

step 0.8 -t
quit
