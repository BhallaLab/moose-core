//moose
// Illustrates problem that addmsg to or from a value field causes
// duplication in the reported fields using showfield *
// MOOSE only program: GENESIS doesn't have this feature at all.

create compartment /foo
create table /bar
addmsg /foo/Vm /bar/inputRequest

showmsg /foo

showfield /foo *
