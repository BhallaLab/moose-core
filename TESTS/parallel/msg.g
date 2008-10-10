//moose

int numNodes = $1
int i
int numStages = 10

for ( i = 0; i < numStages; i = i + 1 )
	int node = 0
	if ( numNodes <= 1 ) 
		create table /tab{i}
	else
		node = i % numNodes
		create table /tab{i}@{node}
	end

	call /tab{i} TABCREATE 1 0 1
	setfield /tab{i} step_mode 0

	if ( i > 0 )
		addmsg /tab{i - 1} /tab{i} SUM output
	end
	if ( i > 1 )
		addmsg /tab{i - 2} /tab{i} SUM output
	end

	setfield /tab{i} output 0
	
end

setfield /tab0 input 1 output 1 table->table[0] 1 table->table[1] 1

reset
step {numStages}
showfield /tab# output
quit
