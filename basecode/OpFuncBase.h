/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _OPFUNCBASE_H
#define _OPFUNCBASE_H

class OpFunc
{
	public:
		virtual ~OpFunc()
		{;}
		virtual bool checkFinfo( const Finfo* s) const = 0;
		virtual bool checkSet( const SetGet* s) const = 0;

		/**
		 * Helper function for finding the correct type of SetGet template
		 * in order to do the assignment.
		 */
		virtual bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const = 0;

// 		virtual void op( const Eref& e, const char* buf ) const = 0;

		virtual void op( const Eref& e, const Qinfo* q, const double* buf ) const = 0;

		virtual string rttiType() const = 0;
};

/**
 * This is the base class for all Get OpFuncs. 
 */
template< class A > class GetOpFuncBase: public OpFunc
{
	public: 
		virtual A reduceOp( const Eref& e ) const = 0;

		string rttiType() const {
			return Conv< A >::rttiType();
		}
};

// Should I template these off an integer for generating a family?
class OpFuncDummy: public OpFunc
{
	public:
		OpFuncDummy();
		bool checkFinfo( const Finfo* s) const;
		bool checkSet( const SetGet* s) const;

		bool strSet( const Eref& tgt, 
			const string& field, const string& arg ) const;

		void op( const Eref& e, const Qinfo* q, const double* buf ) const;
		string rttiType() const;
};

/**
 * This class is used in the foreach call to extract a list of all DataIds
 * on the DataHandler.
 */
class DataIdExtractor: public OpFuncDummy
{
	public:
		DataIdExtractor( vector< DataId >* vec )
			: vec_( vec )
		{;}
		void op( const Eref& e, const Qinfo* q, const double* buf) const
		{
			vec_->push_back( e.index() );
		}
	private:
		vector< DataId >* vec_;
};

#endif // _OPFUNCBASE_H
