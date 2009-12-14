/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _EREF_H
#define _EREF_H

class Eref
{
	public:
		friend ostream& operator <<( ostream& s, const Eref& e );
		Eref( Element* e, DataId index );

		/**
		 * returns the sum of all valid incoming entries
		 */
		double sumBuf( SyncId slot );

		/**
		 * Returns the product of all valid incoming entries
		 * with v. If there are no entries, returns v
		 */
		double prdBuf( SyncId slot, double v );

		/**
		 * Returns the single specified entry
		 */
		double oneBuf( SyncId slot );

		/**
		 * Returns the memory location specified by slot.
		 * Used for sends.
		 */
		double* getBufPtr( SyncId slot );

		/**
		 * Sends a double argument
		 */
		void ssend1( SyncId src, double v );

		/**
		 * Sends two double arguments
		 */
		void ssend2( SyncId src, double v1, double v2 );

		/**
		 * Asynchronous message send.
		 */
		void asend( ConnId conn, FuncId func, const ProcInfo* p,
			const char* arg, unsigned int size ) const;

		/**
		 * Asynchronous send to a specific target.
		 */
		void tsend( ConnId conn, FuncId func, DataId target, 
			const ProcInfo* p,
			const char* arg, unsigned int size ) const;

		/**
		 * Returns data entry
		 */
		char* data();

		/**
		 * Returns data entry of parent object of field array
		 */
		char* data1();

		/**
		 * Returns Element part
		 */
		Element* element() const {
			return e_;
		}

		/**
		 * Returns index part
		 */
		DataId index() const {
			return i_;
		}
	private:
		Element* e_;
		DataId i_;
};

#endif // _EREF_H
