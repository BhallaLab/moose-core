/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _BUF_MOL_H
#define _BUF_MOL_H

class BufMol: public Mol
{
	public: 
		BufMol();
		BufMol( double nInit );

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _BUF_MOL_H
