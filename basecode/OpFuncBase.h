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

		virtual string rttiType() const = 0;
};

class OpFunc0Base: public OpFunc
{
	public:
		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo0* >( s );
		}

		virtual void op( const Eref& e ) const = 0;

		string rttiType() const {
			return "void";
		}
};

template< class A > class OpFunc1Base: public OpFunc
{
	public:

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		virtual void op( const Eref& e, A arg ) const = 0;

		string rttiType() const {
			return Conv< A >::rttiType();
		}
};

template< class A1, class A2 > class OpFunc2Base: public OpFunc
{
	public:
		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo2< A1, A2 >* >( s );
		}

		virtual void op( const Eref& e, A1 arg1, A2 arg2 ) 
				const = 0;


		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType(); 
		}
};

template< class A1, class A2, class A3 > class OpFunc3Base: 
	public OpFunc
{
	public:
		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo3< A1, A2, A3 >* >( s );
		}

		virtual void op( const Eref& e, A1 arg1, A2 arg2, A3 arg3 ) 
				const = 0;

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv< A3 >::rttiType();
		}
};

template< class A1, class A2, class A3, class A4 > 
	class OpFunc4Base: public OpFunc
{
	public:
		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo4< A1, A2, A3, A4 >* >( s );
		}

		virtual void op( const Eref& e, 
						A1 arg1, A2 arg2, A3 arg3, A4 arg4 ) const = 0;

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv<A3>::rttiType() + "," + Conv<A4>::rttiType();
		}
};

template< class A1, class A2, class A3, class A4, class A5 > 
	class OpFunc5Base: public OpFunc
{
	public:
		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo5< A1, A2, A3, A4, A5 >* >( s );
		}

		virtual void op( const Eref& e, 
				A1 arg1, A2 arg2, A3 arg3, A4 arg4, A5 arg5 ) const = 0;

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv<A3>::rttiType() + "," + Conv<A4>::rttiType() +
				"," + Conv<A5>::rttiType();
		}
};

template< class A1, class A2, class A3, class A4, class A5, class A6 > 
		class OpFunc6Base: public OpFunc
{
	public:
		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo6< A1, A2, A3, A4, A5, A6 >* >( s );
		}

		virtual void op( const Eref& e, A1 arg1, A2 arg2, A3 arg3, A4 arg4, 
						A5 arg5, A6 arg6 ) const = 0;

		string rttiType() const {
			return Conv< A1 >::rttiType() + "," + Conv< A2 >::rttiType() +
				"," + Conv<A3>::rttiType() + "," + Conv<A4>::rttiType() +
				"," + Conv<A5>::rttiType() + "," + Conv<A6>::rttiType();
		}
};

/**
 * This is the base class for all GetOpFuncs. 
 */
template< class A > class GetOpFuncBase: public OpFunc
{
	public: 
		bool checkFinfo( const Finfo* s ) const {
			return ( dynamic_cast< const SrcFinfo1< A >* >( s )
			|| dynamic_cast< const SrcFinfo1< FuncId >* >( s ) );
		}

		virtual void op( const Eref& e, ObjId recipient, FuncId fid ) 
				const = 0;

		virtual A returnOp( const Eref& e ) const = 0;

		string rttiType() const {
			return Conv< A >::rttiType();
		}
};

/**
 * This is the base class for all LookupGetOpFuncs. 
 */
template< class L, class A > class LookupGetOpFuncBase: public OpFunc
{
	public: 
		bool checkFinfo( const Finfo* s ) const {
			return ( dynamic_cast< const SrcFinfo1< A >* >( s )
			|| dynamic_cast< const SrcFinfo2< FuncId, L >* >( s ) );
		}

		virtual void op( const Eref& e, L index, 
						ObjId recipient, FuncId fid ) const = 0;

		virtual A returnOp( const Eref& e, const L& index ) const = 0;

		string rttiType() const {
			return Conv< A >::rttiType();
		}
};

#if 0
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
 * This class is used in the forall call to extract a list of all DataIds
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
#endif

#endif // _OPFUNCBASE_H
