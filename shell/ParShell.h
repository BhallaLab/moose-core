/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 *
 * This class is a parallel version of Moose shell.
 * This file is compiled only when the parallelizatin flag is enabled.
 * This class derives from "Shell" class, the shell class for serial Moose. 
 *
 * 
 * This class refers to the base class, "Shell", for all shell functionality. 
 * Parallel moose shell requires overriding of some of the base class functionality. 
 * Such functions are overridden in this class. 
 *
 * 
 * Parallel moose requires some Moose commands to be executed differently than the serial version.
 * This class provides functionality for such commands.
 * 
 */

#ifndef _PARSHELL_H
#define _PARSHELL_H

class ParShell:public Shell
{
	public:
		ParShell();
	/**
	 * This function provides functionality for the planarconnect command on parallel moose.
	 */
	static void planarconnect(const Conn* c, string source, string dest, string spikegenRank, string synchanRank);
	
	/**
	 * This function provides functionality for the planardelay command on parallel moose.
	 */
	static void planardelay(const Conn* c, string source, double delay);
	
	/**
	 * This function provides functionality for the planarweight command on parallel moose.
	 */
	static void planarweight(const Conn* c, string source, double weight);


};

#endif // _PARSHELL_H
