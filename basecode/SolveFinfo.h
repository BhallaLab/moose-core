/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SOLVE_FINFO_H
#define _SOLVE_FINFO_H

/**
 * The Solve Finfo substitutes for ThisFinfo when an Element is being solved.
 * Like the ThisFinfo, it represents the entire data object in an Element,
 * and is stored as the first entry in the Finfo list. In addition it
 * intercepts requests for any Finfos that the Solver is interested in.
 * It also steals the ProcessConn slot for handling the solver message.
 */
class SolveFinfo: public ThisFinfo
{
	public:
		/**
		 * This creates a SolveFinfo to replace the
		 * existing ThisFinfo.
		 */
		SolveFinfo( Finfo** finfos, unsigned int nFinfos,
		       	const ThisFinfo* tf, const string& doc="" );

		~SolveFinfo()
		{;}

		/**
		 * This is the key function that is overridden
		 * so that alternate Finfos can be returned for
		 * operations that need to talk to the solver.
		 */
		const Finfo* match( Element* e, const string& name ) const;

		// ThisFinfo must go to the cinfo to build up the list.
		void listFinfos( vector< const Finfo* >& flist ) const;

		/**
		 * This is an interesting case. What happens when
		 * one copies a solved object? Unexpected stuff
		 * if the solver is not included in the copy.
		 * Anyway, for completeness, let's do it right.
		 */
		Finfo* copy() const {
			return new SolveFinfo( *this );
		}

		void addFuncVec( const string& cname )
		{;}

		////////////////////////////////////////////////////
		// Special functions for SolveFinfo
		////////////////////////////////////////////////////

		/**
		* Returns the Conn going from solved 
		* object e to the solver
		* deprecated
		*/
		// const Conn* getSolvedConn( const Element* e ) const;

	private:
		unsigned int procSlot_;
		vector< Finfo* > finfos_;

};

#endif // _SOLVE_FINFO_H
