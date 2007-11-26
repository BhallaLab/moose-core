/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _PARSHELL_H
#define _PARSHELL_H

class ParShell:public Shell
{
	public:
		ParShell();
		static void planarconnect( const Conn& c, string source, string dest, double probability);

};

#endif // _PARSHELL_H
