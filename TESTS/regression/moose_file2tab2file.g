//genesis
//moose
// This test is for the loading of files into tables, and dumping of
// tables back out to files.

create table /foo
call /foo TABCREATE 16 0 16

file2tab moose_file2tab.plot /foo table -skiplines 2

openfile test.plot "w"
writefile test.plot "/newplot"
writefile test.plot "/plotname test_file2tab2file"
closefile test.plot
tab2file test.plot /foo table
quit
