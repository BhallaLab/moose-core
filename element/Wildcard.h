/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _WILDCARD_H
#define _WILDCARD_H

// Just a couple of extern definitions for general use.

int wildcardRelativeFind( Element* e,
	const string& n, vector< Element* >& ret, int doublehash);
int simpleWildcardFind( const string& path, vector<Element *>& ret);
int wildcardFind(const string& n, vector<Element *>& ret);

#endif // _WILDCARD_H
