/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class is used by FieldDataHandlerBase to enable the foreach command
 * to iterate through all field entries, when the foreach command is
 * passed down to the parent DataHandler. It encapsulates the actual OpFunc
 * to be used by the Field class, and performs the iteration. It lets
 * the parent class deal with thread specificity.
 */
class FieldOpFunc: public OpFuncDummy
{
	public:
		FieldOpFunc( const OpFunc* parentOpFunc, Element* localElement,
			unsigned int argIncrement, unsigned int* argOffset_ );
		void op( const Eref& e, const Qinfo* q, const double* buf ) const;
	private:
		const OpFunc* f_;
		Element* e_;
		unsigned int argIncrement_;
		unsigned int* argOffset_;
		FieldDataHandlerBase* fdh_;
};

