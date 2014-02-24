/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_POOL_INTERFACE_H
#define _ZOMBIE_POOL_INTERFACE_H

/**
 * This pure virtual base class is for solvers that want to talk to
 * the zombie pool.
 */
class ZombiePoolInterface
{
	public:
		virtual void setNinit( const Eref& e, double val ) = 0;
		virtual double getNinit( const Eref& e ) const = 0;
		virtual void setN( const Eref& e, double val ) = 0;
		virtual double getN( const Eref& e ) const = 0;
		virtual void setDiffConst( double val ) = 0;
		virtual double getDiffConst() const = 0;
};

#endif	// _ZOMBIE_POOL_INTERFACE_H
