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

Cinfo::Cinfo()
		: name_( "dummy" ), baseCinfo_( 0 ), dinfo_( 0 ),
			numBindIndex_( 0 )
{;}

Cinfo::Cinfo( const Cinfo& other )
		: name_( "dummy" ), baseCinfo_( 0 ), dinfo_( 0 ),
			numBindIndex_( 0 )
{;}

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
		if ( dynamic_cast< DestFinfo* >( f ) ) {
			destFinfos_.push_back( f );
		}
		else if ( dynamic_cast< SrcFinfo* >( f ) ) {
			srcFinfos_.push_back( f );
		}
		else if ( dynamic_cast< ValueFinfoBase* >( f ) ) {
			valueFinfos_.push_back( f );
		}
		else if ( dynamic_cast< LookupValueFinfoBase* >( f ) ) {
			lookupFinfos_.push_back( f );
		}
		else if ( dynamic_cast< SharedFinfo* >( f ) ) {
			sharedFinfos_.push_back( f );
		}
}

void Cinfo::registerPostCreationFinfo( const Finfo* f )
{
	postCreationFinfos_.push_back( f );
}

void Cinfo::postCreationFunc( Id newId, Element* newElm ) const
{
	for ( vector< const Finfo* >::const_iterator i =
		postCreationFinfos_.begin();
		i != postCreationFinfos_.end(); ++i )
		(*i)->postCreationFunc( newId, newElm );
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

unsigned int Cinfo::numBindIndex() const
{
	return numBindIndex_;
}

const map< string, Finfo* >& Cinfo::finfoMap() const
{
	return finfoMap_;
}

const DinfoBase* Cinfo::dinfo() const
{
	return dinfo_;
}

bool Cinfo::isA( const string& ancestor ) const
{
	if ( ancestor == "Neutral" ) return 1;
	const Cinfo* base = this;
	while( base && base != Neutral::initCinfo() ) {
		if ( ancestor == base->name_ )
			return 1;
		base = base->baseCinfo_;
	}
	return 0;
}

void Cinfo::reportFids() const
{
	for ( map< string, Finfo*>::const_iterator i = finfoMap_.begin();
		i != finfoMap_.end(); ++i ) {
		const DestFinfo* df = dynamic_cast< const DestFinfo* >(
			i->second );
		if ( df ) {
			cout << df->getFid() << "	" << df->name() << endl;
		}
	}
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

////////////////////////////////////////////////////////////////////////
// MOOSE class functions.
////////////////////////////////////////////////////////////////////////

const Cinfo* Cinfo::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< Cinfo, string > docs(
			"docs",
			"Documentation",
			&Cinfo::getDocs
		);

		static ReadOnlyValueFinfo< Cinfo, string > baseClass(
			"baseClass",
			"Name of base class",
			&Cinfo::getBaseClass
		);

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo definitions for different kinds of Finfos
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< Cinfo, Finfo > srcFinfo( "srcFinfo",
			"SrcFinfos in this Class",
			Finfo::initCinfo(),
			&Cinfo::getSrcFinfo,
			&Cinfo::setNumFinfo, // Dummy
			&Cinfo::getNumSrcFinfo
		);
		static FieldElementFinfo< Cinfo, Finfo > destFinfo( "destFinfo",
			"DestFinfos in this Class",
			Finfo::initCinfo(),
			&Cinfo::getDestFinfo,
			&Cinfo::setNumFinfo, // Dummy
			&Cinfo::getNumDestFinfo
		);
		static FieldElementFinfo< Cinfo, Finfo > valueFinfo( "valueFinfo",
			"ValueFinfos in this Class",
			Finfo::initCinfo(),
			&Cinfo::getValueFinfo,
			&Cinfo::setNumFinfo, // Dummy
			&Cinfo::getNumValueFinfo
		);
		static FieldElementFinfo< Cinfo, Finfo > lookupFinfo( "lookupFinfo",
			"LookupFinfos in this Class",
			Finfo::initCinfo(),
			&Cinfo::getLookupFinfo,
			&Cinfo::setNumFinfo, // Dummy
			&Cinfo::getNumLookupFinfo
		);
		static FieldElementFinfo< Cinfo, Finfo > sharedFinfo( "sharedFinfo",
			"SharedFinfos in this Class",
			Finfo::initCinfo(),
			&Cinfo::getSharedFinfo,
			&Cinfo::setNumFinfo, // Dummy
			&Cinfo::getNumSharedFinfo
		);

	static Finfo* cinfoFinfos[] = {
		&docs,				// ReadOnlyValue
		&baseClass,		// ReadOnlyValue
		&srcFinfo,			// FieldElementFinfo
		&destFinfo,			// FieldElementFinfo
		&valueFinfo,			// FieldElementFinfo
		&lookupFinfo,		// FieldElementFinfo
		&sharedFinfo,		// FieldElementFinfo
	};

	static Cinfo cinfoCinfo (
		"Cinfo",
		Neutral::initCinfo(),
		cinfoFinfos,
		sizeof( cinfoFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Cinfo >()
	);

	return &cinfoCinfo;
}

static const Cinfo* cinfoCinfo = Cinfo::initCinfo();


///////////////////////////////////////////////////////////////////
// Field functions
///////////////////////////////////////////////////////////////////
string Cinfo::getDocs() const
{
	return "";
}


static DestFinfo dummy( 
		"dummy", 
		"This Finfo is a dummy. If you are reading this you have used an invalid index", 
0 );

string Cinfo::getBaseClass() const 
{
	return baseCinfo_->name();
}

////////////////////////////////////////////////////////////////////
Finfo* Cinfo::getSrcFinfo( unsigned int i )
{
	if ( i < srcFinfos_.size() )
		return srcFinfos_[i];
	return &dummy; // Return a safe dummy on failure
}

unsigned int Cinfo::getNumSrcFinfo() const
{
	return srcFinfos_.size();
}

////////////////////////////////////////////////////////////////////
Finfo* Cinfo::getDestFinfo( unsigned int i )
{
	if ( i < destFinfos_.size() )
		return destFinfos_[i];
	return &dummy; // Return a safe dummy on failure
}

unsigned int Cinfo::getNumDestFinfo() const
{
	return destFinfos_.size();
}

////////////////////////////////////////////////////////////////////
Finfo* Cinfo::getValueFinfo( unsigned int i )
{
	if ( i < valueFinfos_.size() )
		return valueFinfos_[i];
	return &dummy; // Return a safe dummy on failure
}

unsigned int Cinfo::getNumValueFinfo() const
{
	return valueFinfos_.size();
}


////////////////////////////////////////////////////////////////////
Finfo* Cinfo::getLookupFinfo( unsigned int i )
{
	if ( i < lookupFinfos_.size() )
		return lookupFinfos_[i];
	return &dummy; // Return a safe dummy on failure
}

unsigned int Cinfo::getNumLookupFinfo() const
{
	return lookupFinfos_.size();
}

////////////////////////////////////////////////////////////////////
Finfo* Cinfo::getSharedFinfo( unsigned int i )
{
	if ( i < sharedFinfos_.size() )
		return sharedFinfos_[i];
	return &dummy; // Return a safe dummy on failure
}

unsigned int Cinfo::getNumSharedFinfo() const
{
	return sharedFinfos_.size();
}

////////////////////////////////////////////////////////////////////
void Cinfo::setNumFinfo( unsigned int val ) // Dummy function
{
	;
}
