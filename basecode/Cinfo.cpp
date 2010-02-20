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
			numConn_( 0 ), numFuncIndex_( 0 )
{
	if ( cinfoMap().find( name ) != cinfoMap().end() ) {
		cout << "Warning: Duplicate Cinfo name " << name << endl;
	}
	init( finfoArray, nFinfos );
	cinfoMap()[ name ] = this;
}

void Cinfo::init( Finfo** finfoArray, unsigned int nFinfos )
{
	if ( baseCinfo_ ) {
		// Start out by copying base class function array.
		funcs_ = baseCinfo_->funcs_;
		opFuncNames_ = baseCinfo_->opFuncNames_;
	} else {
		// Initialize zero funcId
		funcs_.push_back( 0 );
		opFuncNames_[ "dummy" ] = 0;
	}

	numFuncIndex_ = 0;
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		Finfo* f = finfoArray[i];
		finfoMap_[ f->name() ] = f;
		
		f->registerOpFuncs( opFuncNames_, funcs_ );
		numConn_= f->registerConn( numConn_ );
		numFuncIndex_ = f->registerSrcFuncIndex( numFuncIndex_ );
	}
}

Cinfo::~Cinfo()
{
	// This assumes we don't give the same Finfo two different names.
	for ( map< string, Finfo*>::iterator i = finfoMap_.begin();
		i != finfoMap_.end(); ++i )
		delete i->second;

	delete dinfo_;
}

const std::string& Cinfo::name() const
{
	return name_;
}

const Cinfo* Cinfo::find( const string& name )
{
	map<string, Cinfo*>::iterator i = cinfoMap().find(name);
	if ( i != cinfoMap().end() )
		return i->second;
	return 0;
}

/**
 * Looks up Finfo from name. If it can't find it, tries up the class
 * hierarchy.
 */
const Finfo* Cinfo::findFinfo( const string& name ) const
{
	map< string, Finfo*>::const_iterator i = finfoMap_.find( name );
	if ( i != finfoMap_.end() )
		return i->second;
	else if ( baseCinfo_ )
		return baseCinfo_->findFinfo( name );
	return 0;
}

const OpFunc* Cinfo::getOpFunc( FuncId fid ) const {
	if ( fid < funcs_.size () )
		return funcs_[ fid ];
	return 0;
}

FuncId Cinfo::getOpFuncId( const string& funcName ) const {
	map< string, FuncId >::const_iterator i = opFuncNames_.find( funcName );
	if ( i != opFuncNames_.end() ) {
		return i->second;
	}
	return 0;
}

// Later: make it possible to assign specific new Id.
Id Cinfo::create( const string& name, unsigned int numEntries ) const
{
	return Id::create( 
		new Element( 
			this, 
			reinterpret_cast< char* >(dinfo_->allocData( numEntries ) ),
			numEntries, dinfo_->size(), numFuncIndex_, numConn_ )
	);
}

void Cinfo::destroy( char* d ) const
{
	dinfo_->destroyData( d );
}

unsigned int Cinfo::numConn() const
{
	return numConn_;
}

unsigned int Cinfo::numFuncIndex() const
{
	return numFuncIndex_;
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

