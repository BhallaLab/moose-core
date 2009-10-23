# Gnuplot script
# First run 'moose TimeTable.g' to generate *.plot files

set datafile commentschars '/#'
set xlabel 'Time (s)'

#
# Pulsgen output - AscFile object 1
#
set title 'Output of PulseGen objects (recorded by first AscFile object).'
set ylabel 'Pulsegen output'
set yrange [ -1.0 : 6.0 ]
p \
  'output/a1.genesis.plot' using 1:2 with line title 'Column 2 : GENESIS', \
  'output/a1.moose.plot' using 1:2 with line title 'Column 2 : MOOSE', \
  'output/a1.genesis.plot' using 1:3 with line title 'Column 3 : GENESIS', \
  'output/a1.moose.plot' using 1:3 with line title 'Column 3 : MOOSE'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/a1.png'
replot
set output
set term wxt

print "Plot image written to output/a1.png.\n"

#
# Pulsgen output - AscFile object 2
#
set title 'Output of PulseGen objects (recorded by second AscFile object).'
set ylabel 'Pulsegen output'
set yrange [ -1.0 : 6.0 ]
p \
  'output/a2.genesis.plot' using 1:2 with line title 'Column 2 : GENESIS', \
  'output/a2.moose.plot' using 1:2 with line title 'Column 2 : MOOSE', \
  'output/a2.genesis.plot' using 1:3 with line title 'Column 3 : GENESIS', \
  'output/a2.moose.plot' using 1:3 with line title 'Column 3 : MOOSE'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/a2.png'
replot
set output
set term wxt

print "Plot image written to output/a2.png.\n"

#
# Difference of columns - AscFile object 1
#
set title 'Difference of columns (recorded by first AscFile object).'
set ylabel 'Difference: Column 2 - Column 3'
set autoscale
p \
  'output/a1.genesis.plot' using 1:($2)-($3) with line title '(Col 2) - (Col 3) : GENESIS', \
  'output/a1.moose.plot' using 1:($2)-($3) with line title '(Col 2) - (Col 3) : MOOSE'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/diff1.png'
replot
set output
set term wxt

print "Plot image written to output/diff1.png.\n"

#
# Difference of columns - AscFile object 2
#
set title 'Difference of columns (recorded by second AscFile object).'
set ylabel 'Difference: Column 2 - Column 3'
set autoscale
p \
  'output/a2.genesis.plot' using 1:($2)-($3) with line title '(Col 2) - (Col 3) : GENESIS', \
  'output/a2.moose.plot' using 1:($2)-($3) with line title '(Col 2) - (Col 3) : MOOSE'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/diff2.png'
replot
set output
set term wxt

print "Plot image written to output/diff2.png.\n"
