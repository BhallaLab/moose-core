/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TABLE_H
#define _TABLE_H

/**
 * Receives and records inputs. Handles plot and spiking data in batch mode.
 */
class Table: public TableBase
{
	public: 
		Table();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setThreshold( double v );
		double getThreshold() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		void input( double v );
		void spike( double v );

		void recvData( double v );

		//////////////////////////////////////////////////////////////////
		// Lookup funcs for table
		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();
	private:
		double threshold_;
		double lastTime_;
		double input_;
};

#endif	// _TABLE_H
