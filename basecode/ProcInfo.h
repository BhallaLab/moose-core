/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef PROCINFO_H
#define PROCINFO_H
/**
 * This class manages the context of simulations and is passed around
 * by the various scheduling objects.
 * Later it may probably also handle other aspects of the context.
 */
class ProcInfoBase
{
	public:
		ProcInfoBase( unsigned int shell = 0,
			double dt = 1.0, double currTime = 0.0, const string& doc="" )
			: dt_( dt ), currTime_( currTime ), shell_( shell )
		{
			;
		}

		double dt_;
		double currTime_;

	private:
		unsigned int shell_;
};

typedef ProcInfoBase* ProcInfo;

#endif // PROCINFO_H
