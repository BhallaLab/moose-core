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
	// This assumes we don't give the same Finfo two different names.
	for ( map< string, Finfo*>::iterator i = finfoMap_.begin();
		i != finfoMap_.end(); ++i )
		delete i->second;

	delete dinfo_;
}

////////////////////////////////////////////////////////////////////
// Initialization funcs
////////////////////////////////////////////////////////////////////

#if 0
void Cinfo::init( Finfo** finfoArray, unsigned int nFinfos )
{
	if ( baseCinfo_ ) {
		// Copy over base Finfos.
		numBindIndex_ = baseCinfo->numBindIndex_
		finfoMap_ = baseCinfo->finfoMap_;
		funcs_ = baseCinfo->funcs_;
		/*
		for ( map< string, Finfo* >::iterator i = 
			baseCinfo_->finfoMap_.begin();
			i != baseCinfo_->finfoMap_.end();
			++i ) {
			Finfo* f = i->second->clone();
			finfo_map_[ i->first ] = f;
		}
		*/
		// Start out by copying base class function array.
		// funcs_ = baseCinfo_->funcs_;
		// opFuncNames_ = baseCinfo_->opFuncNames_;
	} 
	/*
	else {
		// Initialize zero funcId
		// funcs_.push_back( 0 );
		opFuncNames_[ "dummy" ] = 0;
	}
	*/
	// Fill in new finfos. Just overwrite baseCinfo entries.
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		Finfo* f = finfoArray[i];
		finfoMap_[ f->name() ] = f;
		// Register them too.
		f->registerFinfo( this );
		/*
		map< string, Finfo* >::iterator i = finfoMap_.find( f->name() );
		if ( i != finfoMap_.end() ) {
			delete i->second;
			i->second = f;
		} else {
			finfoMap_[ f->name() ] = f;
		}
		*/
	}

/*
	// Register the whole mess.
	numBindIndex_ = 0;
	for ( map< string, Finfo* >::iterator i = finfoMap_.begin();
		i != finfoMap_.end(); ++i ) {
		i->second->registerFinfo( this );
	}
	*/

	/*
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		Finfo* f = finfoArray[i];
		registerFinfo( f );

		SharedFinfo* sf = dynamic_cast< SharedFinfo* >( f );
		if ( sf ) {
			for ( vector< SrcFinfo* >::const_iterator j = sf->src().begin();
				j != sf->src().end(); ++j ) {
				registerFinfo( *j );
			}
			for ( vector< Finfo* >::const_iterator j = sf->dest().begin();
				j != sf->dest().end(); ++j ) {
				registerFinfo( *j );
			}
		}
	}
	*/
}
#endif

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

/*
void Cinfo::addMoreFinfos( Finfo** finfoArray, unsigned int nFinfos )
	// Fill in new finfos. Just overwrite baseCinfo entries.
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		Finfo* f = finfoArray[i];
		finfoMap_[ f->name() ] = f;
		// Register them too.
		f->registerFinfo( this );
	}
}
*/

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
		/*
		f->registerOpFuncs( opFuncNames_, funcs_ );
		numBindIndex_= f->registerBindIndex( numBindIndex_ );
		*/
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

bool Cinfo::create( Id id, const string& name, unsigned int numEntries ) const
{
	Element* e = new Element( 
			this, 
			reinterpret_cast< char* >(dinfo_->allocData( numEntries ) ),
			numEntries, dinfo_->size(), numBindIndex_ );
	if ( e ) {
		id.bindIdToElement( e );
		return 1;
	}
	return 0;
}

void Cinfo::destroy( char* d ) const
{
	dinfo_->destroyData( d );
}

unsigned int Cinfo::numBindIndex() const
{
	return numBindIndex_;
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

