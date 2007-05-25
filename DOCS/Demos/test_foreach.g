//moose
create neutral /n
create pool /n/a
create pool /n/b
create pool /n/c
create pool /n/d
create pool /n/e

str name
foreach name ( {el /n/#} )
	echo name = {name}, from getfield name = {getfield {name} name}
end
