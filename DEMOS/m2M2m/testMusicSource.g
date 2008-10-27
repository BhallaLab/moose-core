if ( ! {exists /music} )
	echo "Error! This MOOSE does not have MUSIC support."
	echo "Install a MOOSE package which supports the MUSIC library, and try again."
	echo "Will now exit..."
	quit
end

call /music addPort out event UTDATA
create TimeTable /tt
call /tt load "testtider.txt" 0

addmsg /tt/event /music/UTDATA/channel[0]/synapse

setclock 0 0.01 0
setclock 1 0.01 1
useclock /music/UTDATA 0
useclock /music 1

reset

step 1 -t 

quit
