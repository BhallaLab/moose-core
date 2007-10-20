/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include "moose.h"
#include "Shell.h"

void Shell::innerLoadTab(const string& line)
{
	istringstream istr(line);
	vector< double > values;

	string name;

	istr >> name;
	if ( name == "-continue" ) {
		if ( !lastTab_.good() ) {
			cout << "Error: loadtab -continue: No previous table loaded\n";
			return;
		}
		Element* tab = lastTab_();
		// int xdivs;
		// get< int >( tab, "xdivs", xdivs );
		double y;
		while (istr >> y) {
			values.push_back(y);
		}
		/*
		if (values.size() > xdivs + 1)
			cerr << "Error: loadTab: Overfill\n";
		else
		*/
		set< vector< double > >( tab, "appendTableVector", values );
	} else {
		Id tabId( name );
		if ( tabId.good() ) {
			lastTab_ = tabId;
			Element* tab = tabId();
			// Create an interpol for the data, as a child of the
			// table. Later we'll do this right with a special kind of
			// field that handles interpols and the like.
			string tabname;
			int dummyint;
			unsigned int xdivs;
			double xmin;
			double xmax;
			istr >> tabname >> dummyint >> xdivs >> xmin >> xmax;
			if ( tab->cinfo()->isA( Cinfo::find( "Interpol" ) ) || 
				tab->cinfo()->isA( Cinfo::find( "Table" ) ) ) {
				set< int >( tab, "xdivs", static_cast< int >( xdivs ) );
				set< double >( tab, "xmin", xmin );
				set< double >( tab, "xmax", xmax );
			} else if ( tab->cinfo()->isA( Cinfo::find( "HHChannel" ) ) ) {
				// do stuff here.
			} else {
				cerr << "Error: loadTab: " << name << " is an unknown type\n";
				return;
			}
			values.resize(0);
			double y;
			while (istr >> y) {
				values.push_back(y);
			}
			if (values.size() > xdivs + 1)
				cerr << "Error: loadTab: Overfill\n";
			// underfull is not an issue, we assume a -continue.
			else
				set< vector< double > >( tab, "tableVector", values );
		} else {
			cout << "Error: table '" << name << "' not found\n";
			return;
		}
	}
}

/*

void kp::FillNotes(const char* line) {
		char *continue_line = strrchr(line, '\\');
		char *close_quotes;
		char *open_quotes = strchr(line, '\"');

		if (open_quotes)
				*open_quotes = ' ';
		close_quotes = strrchr(line, '\"');
		if (close_quotes)
				*close_quotes = '\0';

		notes_value += line;

		if (continue_line == 0) {
			for (unsigned long i = notes_value.length()-1; i > 0; i--) {
				if (notes_value[i] == '\"')
					notes_value[i] = ' ';
			}
			state = DEFAULT;
			const cinfo* stype = cinfo::Find("string");
			element* notes = stype->Create(notes_parent, "notes", 0);
			SetField(notes, "strval", notes_value);
		}
}

void kp::InitNotes(const char* call_line) {
	char name[LINELEN];
	char type[LINELEN];


	sscanf(call_line, "%s %s", name, type);
	if (strcmp(type, "LOAD") != 0)
			return;

	// Get rid of the terminal object 'notes' on the object name
	char* endname = strrchr(name, '/');
	*endname = '\0';
	element* e = element::Find(name);
	if (!e) {
			cerr << "Error: failed to find object " << name << "\n";
			return;
	}
	
	notes_parent = e;
	notes_value = "";
	state = NOTES;
	char* pos = strchr(call_line, ' ');
	if (strncmp(pos + 1, "LOAD", 4) == 0)
		FillNotes(pos + 6);
}
*/
