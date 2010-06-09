# Gnuplot script
#
# Run this script using:
#   gnuplot plot.gnuplot
#
# To generate *.plot files, first run:
#	genesis MusicNetwork.g full
#	moose MusicNetwork.g full
#	mpirun -np 2 music net-1.music
#	mpirun -np 4 music net-2.music
#

set datafile commentschars '/#'
set xlabel 'Step # [dt = 100e-6 s]'    # This is the plot dt
set ylabel 'Vm (V)'

#
# Output number 0
#
set title 'MOOSE/MUSIC/MOOSE feedforward network: Output #0'
plot \
	'output/moose.plot' every :::0::0 with line title 'MOOSE', \
	'output/genesis.plot' every :::0::0 with line title 'GENESIS', \
	'output/music-1.plot' every :::0::0 with line title 'MUSIC (1+1 processes)', \
	'output/music-2.plot' every :::0::0 with line title 'MUSIC (2+2 processes)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/c0.png'
replot
set output
set term pop

print "Plot image written to output/c0.png.\n"

#
# Output number 1
#
set title 'MOOSE/MUSIC/MOOSE feedforward network: Output #1'
plot \
	'output/moose.plot' every :::1::1 with line title 'MOOSE', \
	'output/genesis.plot' every :::1::1 with line title 'GENESIS', \
	'output/music-1.plot' every :::1::1 with line title 'MUSIC (1+1 processes)', \
	'output/music-2.plot' every :::1::1 with line title 'MUSIC (2+2 processes)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/c1.png'
replot
set output
set term pop

print "Plot image written to output/c1.png.\n"

#
# Output number 2
#
set title 'MOOSE/MUSIC/MOOSE feedforward network: Output #2'
plot \
	'output/moose.plot' every :::2::2 with line title 'MOOSE', \
	'output/genesis.plot' every :::2::2 with line title 'GENESIS', \
	'output/music-1.plot' every :::2::2 with line title 'MUSIC (1+1 processes)', \
	'output/music-2.plot' every :::2::2 with line title 'MUSIC (2+2 processes)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/c2.png'
replot
set output
set term pop

print "Plot image written to output/c2.png.\n"

#
# Output number 3
#
set title 'MOOSE/MUSIC/MOOSE feedforward network: Output #3'
plot \
	'output/moose.plot' every :::3::3 with line title 'MOOSE', \
	'output/genesis.plot' every :::3::3 with line title 'GENESIS', \
	'output/music-1.plot' every :::3::3 with line title 'MUSIC (1+1 processes)', \
	'output/music-2.plot' every :::3::3 with line title 'MUSIC (2+2 processes)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/c3.png'
replot
set output
set term pop

print "Plot image written to output/c3.png.\n"

#
# Output number 4
#
set title 'MOOSE/MUSIC/MOOSE feedforward network: Output #4'
plot \
	'output/moose.plot' every :::4::4 with line title 'MOOSE', \
	'output/genesis.plot' every :::4::4 with line title 'GENESIS', \
	'output/music-1.plot' every :::4::4 with line title 'MUSIC (1+1 processes)', \
	'output/music-2.plot' every :::4::4 with line title 'MUSIC (2+2 processes)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/c4.png'
replot
set output
set term pop

print "Plot image written to output/c4.png.\n"
