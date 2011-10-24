/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class is used by FieldDataHandlerBase to enable the forall command
 * to iterate through all field entries, when the forall command is
 * passed down to the parent DataHandler. It encapsulates the actual OpFunc
 * to be used by the Field class, and performs the iteration. It lets
 * the parent class deal with thread specificity.
 * This variant handles increments to the argument list when there are
 * multiple arguments. It supports tiling, so the number of args does not
 * have to match exactly.
 */
class FieldOpFunc: public OpFuncDummy
{
	public:
		FieldOpFunc( const OpFunc* parentOpFunc, Element* localElement,
			unsigned int argSize, unsigned int numArgs,
			unsigned int* argOffset_ );
		void op( const Eref& e, const Qinfo* q, const double* buf ) const;
	private:
		const OpFunc* f_;
		Element* e_;
		unsigned int argSize_;
		unsigned int maxArgOffset_;
		unsigned int* argOffset_;
		FieldDataHandlerBase* fdh_;
};

/**
 * This variant of FieldOpFunc is used if the iterator has only a single
 * argument to be applied to all targets. So we don't mess with incrementing
 * the argument.
 */
class FieldOpFuncSingle: public OpFuncDummy
{
	public:
		FieldOpFuncSingle( const OpFunc* parentOpFunc, 
			Element* localElement );
		void op( const Eref& e, const Qinfo* q, const double* buf ) const;
	private:
		const OpFunc* f_;
		Element* e_;
		FieldDataHandlerBase* fdh_;
};

