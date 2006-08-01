/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

enum FieldMode { RW, RO, CONST };

extern int next_token(string& ret, const string& s, int i);

extern const string checkForVector( const string& temp, 
	const string& line, unsigned int& j );

// Iterates through the string vector till all braces balance out
// Begins the search at startpos on the line pointed to by i.
// Optionally test for an opening brace and complain if it isn't found
extern bool balanceBraces( 
	vector< string >::iterator& i,
	vector< string >::iterator end,
	unsigned long startpos,
	bool lookForOpeningBrace );
