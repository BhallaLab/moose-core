/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class manages the context of simulations and is passed around
 * by the various scheduling objects.
 * Later it may probably also handle other aspects of the context.
 */
class ProcInfo
{
		public:
				double dt;
				double currentTime;
};
