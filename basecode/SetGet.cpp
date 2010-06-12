/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SetGet.h"

/*
Eref SetGet::shelle_( 0, 0 );
Element* SetGet::shell_;

void SetGet::setShell()
{
	Id shellid;
	shelle_ = shellid.eref();
	shell_ = shelle_.element();
}
*/

/**
 * completeSet: Confirms that the target function has been executed.
 * Later this has to be much more complete: perhaps await return of a baton
 * to ensure that the operation is complete through varous threading,
 * nesting, and multinode operations.
 * Even single thread, single node this current version is dubious: 
 * suppose the function called by 'set' issues a set command of its own?
 */
void SetGet::completeSet() const
{
	// e_.element()->clearQ();
	Qinfo::clearQ( Shell::procInfo() );
}

bool SetGet::checkSet( const string& field, Eref& tgt, FuncId& fid ) const
{
	// string field = "set_" + destField;
	const Finfo* f = e_.element()->cinfo()->findFinfo( field );
	if ( !f ) { // Could be a child element?
		Id child = ( Neutral::getChild( e_, 0, field ) );
		if ( child != Id() ) {
			f = child()->cinfo()->findFinfo( "setThis" );
			assert( f ); // should always work as Neutral has the field.
			if ( child()->dataHandler()->numData1() == 
				e_.element()->dataHandler()->numData1() )
				tgt = Eref( child(), e_.index() );
			else if ( child()->dataHandler()->numData1() <= 1 )
				tgt = Eref( child(), 0 );
			else {
				cout << "SetGet::checkSet: child index mismatch\n";
				return 0;
			}
		}
	} else {
		tgt = e_;
	}
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	if ( !df )
		return 0;
	
	fid = df->getFid();
	const OpFunc* func = df->getOpFunc();

// 	fid = e_.element()->cinfo()->getOpFuncId( field );
//	const OpFunc* func = e_.element()->cinfo()->getOpFunc( fid );
	if ( !func ) {
		cout << "set::Failed to find " << e_ << "." << field << endl;
		return 0;
	}
	if ( func->checkSet( this ) ) {
		return 1;
	} else {
		cout << "set::Type mismatch" << e_ << "." << field << endl;
		return 0;
	}
}

/**
 * Puts data into target queue for calling functions and setting
 * fields. This is a common core function used by the various
 * type-specialized variants.
 * Called after func checking etc.
void SetGet::iSetInner( FuncId fid, const char* val, unsigned int size )
{
	static unsigned int setBindIndex = 0;
	shell_->clearBinding( setBindIndex );
	Msg* m = new SingleMsg( shelle_, e_ );
	shell_->addMsgAndFunc( m->mid(), fid, setBindIndex );
	// Qinfo( FuncId f, DataId srcIndex, unsigned int size )
	Qinfo q(  fid, 0, size );
	shell_->asend( q, setBindIndex, Shell::procInfo(), val );
}
 */

/*
void SetGet::resizeBuf( unsigned int size )
{
	buf_.resize( size );
}

char* SetGet::buf()
{
	return &buf_[0];
}
*/

/////////////////////////////////////////////////////////////////////////

/**
 * Here we generate the actual opfunc names from the field name.
 * We use the setFid only to do type checking. The actual Fid we
 * need to call is the getFid, which is passed back.
 */
#if 0
bool SetGet::checkGet( const string& field, FuncId& getFid )	
	const
{
	static const SetGet1< FuncId > sgf( shelle_ );

//	string setField = "set_" + field;
	string getField = "get_" + field;

	/*
	const DestFinfo* sf = dynamic_cast< const DestFinfo* >( 
		e_.element()->cinfo()->findFinfo( setField ) );
		*/
	const DestFinfo* gf = dynamic_cast< const DestFinfo* >(
		e_.element()->cinfo()->findFinfo( getField ) );

	if ( !( sf && gf ) ) {
		cout << "get::Failed to find " << e_ << "." << field << endl;
		return 0;
	}

	getFid = gf->getFid();
	// const OpFunc* setFunc = sf->getOpFunc();
	const OpFunc* getFunc = gf->getOpFunc();

/*
	FuncId setFid = e_.element()->cinfo()->getOpFuncId( setField );
	getFid = e_.element()->cinfo()->getOpFuncId( getField );
	const OpFunc* setFunc = e_.element()->cinfo()->getOpFunc( setFid );
	const OpFunc* getFunc = e_.element()->cinfo()->getOpFunc( getFid );
	if ( !( setFunc && getFunc ) ) {
		cout << "get::Failed to find " << e_ << "." << field << endl;
		return 0;
	}
*/
	if ( !getFunc->checkSet( &sgf ) ) {
		 cout << "get::Type mismatch on getFunc" << e_ << "." << field << endl;
		return 0;
	}
	if ( !setFunc->checkSet( this ) ) {
		 cout << "get::Type mismatch on return value" << e_ << "." << field << endl;
		return 0;
	}
	return 1;
}

bool SetGet::iGet( const string& field ) const
{
	// static unsigned int getBindIndex = 0;
	// static FuncId retFunc = shell_->cinfo()->getOpFuncId( "handleGet" );

	const DestFinfo* df = dynamic_cast< const DestFinfo* >( 
		shell_->cinfo()->findFinfo( "handleGet" ) );
	assert( df );
	FuncId retFunc = df->getFid();

	const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >(
		shell_->cinfo()->findFinfo( "requestGet" ) );
	assert( sf );
	BindIndex getBindIndex = sf->getBindIndex();

	FuncId destFid;
	if ( checkGet( field, destFid ) ) {
		shell_->clearBinding( getBindIndex );
		Msg* m = new SingleMsg( shelle_, e_ );
		shell_->addMsgAndFunc( m->mid(), destFid, getBindIndex );
		Qinfo q( destFid, 0, sizeof( FuncId ) );
		shell_->asend( q, getBindIndex, Shell::procInfo(),  
			reinterpret_cast< char* >( &retFunc ) );
		return 1;
	}
	return 0;
}
#endif
