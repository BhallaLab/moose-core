#ifndef _HHGate2D_h
#define _HHGate2D_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class HHGate2D
{
	public:
		HHGate2D();
		HHGate2D( Id originalChanId, Id originalGateId );
		
		double lookupA( vector< double > v ) const;
		double lookupB( vector< double > v ) const;
		
		// void gateFunc( const Conn* c, double v1, double v2 );
		
		/**
		 * Single call to get both A and B values in a single
		 * lookup
		 */
		void lookupBoth( double v, double c, double* A, double* B) const;

		/**
		 * Checks if the provided Id is the one that the HHGate was created
		 * on. If true, fine, otherwise complains about trying to set the
		 * field.
		 */
		bool checkOriginal( Id id, const string& field ) const;

		/**
		 * isOriginalChannel returns true if the provided Id is the Id of
		 * the channel on which the HHGate was created.
		 */
		bool isOriginalChannel( Id id ) const;

		/**
		 * isOriginalChannel returns true if the provided Id is the Id of
		 * the Gate created at the same time as the original channel.
		 */
		bool isOriginalGate( Id id ) const;

		/**
		 * Returns the Id of the original Channel.
		 */
		Id originalChannelId() const;

		/**
		 * Returns the A interpol
		 */
		Interpol2D* getTableA( unsigned int i );

		/**
		 * Dummy access func for Interpols. Always returns 1.
		 */
		unsigned int getNumTable() const;

		/**
		 * Dummy assignment function for the number of interpols:
		 * We always have both interpols 
		 */
		void setNumTable( unsigned int num );

		/**
		 * Returns the B interpol
		 */
		Interpol2D* getTableB( unsigned int i );

		static const Cinfo* initCinfo();
	private:
		Interpol2D A_;
		Interpol2D B_;

		Id originalChanId_;
		Id originalGateId_;
};

// Used by solver, readcell, etc.

#endif // _HHGate2D_h
