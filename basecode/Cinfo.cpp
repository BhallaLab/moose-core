/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "Dinfo.h"

Cinfo::Cinfo( const string& name,
				const Cinfo* baseCinfo,
				Finfo** finfoArray,
				unsigned int nFinfos,
				DinfoBase* d,
				struct SchedInfo* schedInfo,
				unsigned int nSched
)
		: name_( name ), baseCinfo_( baseCinfo ), dinfo_( d ),
			numBindIndex_( 0 )
{
	if ( cinfoMap().find( name ) != cinfoMap().end() ) {
		cout << "Warning: Duplicate Cinfo name " << name << endl;
	}
	init( finfoArray, nFinfos );
	cinfoMap()[ name ] = this;
}

Cinfo::~Cinfo()
{
	/*
	 * We no longer have to delete the Finfos, as they are all statically
	 * defined.
	// This assumes we don't give the same Finfo two different names.
	for ( map< string, Finfo*>::iterator i = finfoMap_.begin();
		i != finfoMap_.end(); ++i )
		delete i->second;
	*/
	delete dinfo_;
}

////////////////////////////////////////////////////////////////////
// Initialization funcs
////////////////////////////////////////////////////////////////////

/**
 * init: initializes the Cinfo. Must be called just once
 */
void Cinfo::init( Finfo** finfoArray, unsigned int nFinfos )
{
	if ( baseCinfo_ ) {
		// Copy over base Finfos.
		numBindIndex_ = baseCinfo_->numBindIndex_;
		finfoMap_ = baseCinfo_->finfoMap_;
		funcs_ = baseCinfo_->funcs_;
	} 
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		registerFinfo( finfoArray[i] );
	}
}

FuncId Cinfo::registerOpFunc( const OpFunc* f )
{
	FuncId ret = funcs_.size();
	funcs_.push_back( f );
	return ret;
}

BindIndex Cinfo::registerBindIndex()
{
	return numBindIndex_++;
}

void Cinfo::registerFinfo( Finfo* f )
{
		finfoMap_[ f->name() ] = f;
		f->registerFinfo( this );
}

//////////////////////////////////////////////////////////////////////
// Look up operations.
//////////////////////////////////////////////////////////////////////

const Cinfo* Cinfo::find( const string& name )
{
	map<string, Cinfo*>::iterator i = cinfoMap().find(name);
	if ( i != cinfoMap().end() )
		return i->second;
	return 0;
}

/**
 * Looks up Finfo from name.
 */
const Finfo* Cinfo::findFinfo( const string& name ) const
{
	map< string, Finfo*>::const_iterator i = finfoMap_.find( name );
	if ( i != finfoMap_.end() )
		return i->second;
	return 0;
}

/**
 * looks up OpFunc by FuncId
 */
const OpFunc* Cinfo::getOpFunc( FuncId fid ) const {
	if ( fid < funcs_.size () )
		return funcs_[ fid ];
	return 0;
}

/*
FuncId Cinfo::getOpFuncId( const string& funcName ) const {
	map< string, FuncId >::const_iterator i = opFuncNames_.find( funcName );
	if ( i != opFuncNames_.end() ) {
		return i->second;
	}
	return 0;
}
*/

////////////////////////////////////////////////////////////////////////
// Miscellaneous
////////////////////////////////////////////////////////////////////////

const std::string& Cinfo::name() const
{
	return name_;
}

char* Cinfo::createData( unsigned int numEntries ) const 
{
	return reinterpret_cast< char* >(dinfo_->allocData( numEntries ) );
}

/*
bool Cinfo::create( Id id, const string& name, unsigned int numEntries,
	Element::Decomposition decomp ) const
{
	Element* e = new Element( 
			this, 
			reinterpret_cast< char* >(dinfo_->allocData( numEntries ) ),
			numEntries, dinfo_->size(), numBindIndex_, decomp );
	if ( e ) {
		id.bindIdToElement( e );
		return 1;
	}
	return 0;
}
*/

void Cinfo::destroyData( char* d ) const
{
	dinfo_->destroyData( d );
}

unsigned int Cinfo::numBindIndex() const
{
	return numBindIndex_;
}

unsigned int Cinfo::dataSize() const
{
	return dinfo_->size();
}

const map< string, Finfo* >& Cinfo::finfoMap() const
{
	return finfoMap_;
}

////////////////////////////////////////////////////////////////////////
// Private functions.
////////////////////////////////////////////////////////////////////////
map<string, Cinfo*>& Cinfo::cinfoMap()
{
	static map<std::string, Cinfo*> lookup_;
	return lookup_;
}



/*
map< OpFunc, FuncId >& Cinfo::funcMap()
{
	static map< OpFunc, FuncId > lookup_;
	return lookup_;
}
*/

