//moose

create table /foo

call /foo TABCREATE 5 0 1

int i

for ( i = 0; i <= 5; i = i + 1 )
	setfield /foo table->table[{i}] {i * i}
end

if ( {version} > 2.5 ) // moose
	// This should produce just one copy of the data
	setfield /foo print pfile
	setfield /foo print pfile
	
	// This should produce two copies of the data.
	setfield /foo append afile
	setfield /foo append afile
end

// The lines below should work both with MOOSE and GENESIS

	// This should produce just one copy of the data
	tab2file genpfile /foo table -nentries 6 -overwrite
	tab2file genpfile /foo table -nentries 6 -overwrite

	// This should produce two copies of the data.
	tab2file genafile /foo table -nentries 6
	tab2file genafile /foo table -nentries 6
