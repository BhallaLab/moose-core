/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <iostream>
#include <fstream>
#include <map>
#include "moose.h"
#include "Cinfo.h"
#include "ThisFinfo.h"

// The include below is commented out because the header file is not
// present in the svn repository. -- Upi
// #include "filecheck.h"

// The three includes below are needed because of the create function
// requiring an instantiable Element class. Could get worse if we
// permit multiple variants of Element, say an array form.

#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"

// These includes are needed to call set
#include <algorithm>
#include "DerivedFtype.h"
#include "Ftype2.h"
#include "setget.h"

//////////////////////////////////////////////////////////////////
// Cinfo is the class info for the MOOSE classes.
//////////////////////////////////////////////////////////////////

Finfo* findMatchingFinfo( const string& name,
				Finfo** finfoArray, unsigned int nFinfos )
{
	for ( unsigned int i = 0; i < nFinfos; i++ )
		if ( name == finfoArray[i]->name() )
				return finfoArray[i];
	return 0;
}

Cinfo::Cinfo(const std::string& name,
				const std::string& author,
				const std::string& description,
				const Cinfo* baseCinfo,
				Finfo** finfoArray,
				unsigned int nFinfos,
				const Ftype* ftype
)
		: name_(name), author_(author), 
		description_(description), baseCinfo_(baseCinfo),
		base_( 0 ), ftype_( ftype ), nSrc_( 0 ), nDest_( 0 )
{
	unsigned int i;
	if ( baseCinfo ) {
		nSrc_ = baseCinfo->nSrc_;
		nDest_ = baseCinfo->nDest_;
		for ( i = 0; i < baseCinfo->finfos_.size(); i++ ) {
			Finfo* f = findMatchingFinfo(
				baseCinfo->finfos_[i]->name(), finfoArray, nFinfos );
			if ( f ) {
				// The inherit operation is true only if the types
				// match.
				assert( f->inherit( baseCinfo->finfos_[i] ) );
				finfos_.push_back( f );
			} else
				finfos_.push_back( baseCinfo->finfos_[i] );
		}
	}
#ifdef GENERATE_WRAPPERS        
        std::string out_dir_name ="generated/";
        std::string swig_name = out_dir_name+"pymoose.i";    
        std::string header_name = out_dir_name+name+".h";
        std::string cpp_name = out_dir_name+name+".cpp";
                    
        ofstream header, cpp, swig;
        bool created = false;
        
        created = open_outfile(header_name, header);

        if (created){
            created = open_outfile(cpp_name, cpp);
            if (!created)
            {
                header.close();
            }else 
            {
                created = open_appendfile(swig_name, swig);
                if (!created)
                {
                    header.close();
                    cpp.close();
                }            
            }        
        }
        if (!created)
        {
            cerr << "Could not create files for " << name << endl;        
        }
    
        if (created)
        {
            /*
              Insert the common class members,
              constructors and the common method getType()
              and the common field className.
            */
            header << "#ifndef _pymoose_" << name << "_h\n"
                   << "#define _pymoose_"<< name << "_h\n"
                   << "#include \"PyMooseBase.h\"\n"
                   << "class " << name << " : public PyMooseBase\n"
                   << "{"
                   << "    public:\n"
                   << "        static const std::string className;\n"
                   << "        " << name << "(Id id);\n"
                   << "        " << name << "(std::string path);\n"
                   << "        " << name << "(std::string name, Id parentId);\n"
                   << "        " << name << "(std::string name, PyMooseBase* parent);\n"
                   << "        ~" << name <<"();\n"
                   << "        const std::string& getType();\n";
        
            cpp << "#ifndef _pymoose_" << name << "_cpp\n"
                << "#define _pymoose_"<< name << "_cpp\n"
                << "#include \"" << name << ".h\"\n"
                << "const std::string "<< name <<"::className = \"" << name <<"\";\n"
                << name << "::" << name << "(Id id):PyMooseBase(id){}\n"
                << name << "::" << name <<"(std::string path):PyMooseBase(className, path){}\n"
                << name << "::" << name << "(std::string name, Id parentId):PyMooseBase(className, name, parentId){}\n"
                << name << "::" << name << "(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}\n"
                << name << "::~" << name <<"(){}\n"
                << "const std::string& "<< name <<"::getType(){ return className; }\n";
            swig << "%include \"" << name << ".h\"\n";        
        }
#endif

	for ( i = 0 ; i < nFinfos; i++ ) {
            finfoArray[i]->countMessages( nSrc_, nDest_ );
            finfos_.push_back( finfoArray[i] );
#ifdef GENERATE_WRAPPERS                
            if ( created )
            {                    
                std::string fieldName = finfoArray[i]->name();
                std::string fieldType = finfoArray[i]->ftype()->getTemplateParameters();
                                
                if (fieldType != "void")
                {
                    /* Insert the getters and setters */                  
                    header << "        " << fieldType << " __get_" << fieldName << "() const;\n";
                    header << "        void" << " __set_" << fieldName << "(" << fieldType << " " << fieldName << ");\n";
                
                    cpp << fieldType <<" " <<  name << "::__get_" <<  fieldName << "() const\n"
                        << "{\n"
                        << "    " <<  fieldType << " " << fieldName << ";\n"
                        << "    get < " << fieldType << " > (Element::element(id_), \""<< fieldName << "\"," << fieldName << ");\n"
                        << "    return " << fieldName << ";\n"
                        << "}" << endl;
                    cpp << "void " << name << "::__set_" << fieldName <<"( " << fieldType << " " << fieldName << " )\n"
                        << "{\n"
                        << "    set < " << fieldType << " > (Element::element(id_), \"" << fieldName << "\", "<< fieldName << ");\n"
                        << "}" << endl;
                    swig << "%attribute(" << name << ", " << fieldType << ", " << fieldName << ", __get_" << fieldName << ", __set_" << fieldName << ")\n";                
                }
            }
#endif
        }
#ifdef GENERATE_WRAPPERS
        if (created)
        {
            header << "};\n"
                   << "#endif" << endl;
            cpp << "#endif" << endl;
            header.close();
            cpp.close();
            swig.close();        
        }
#endif
        
	thisFinfo_ = new ThisFinfo( this );
	noDelFinfo_ = new ThisFinfo( this, 1 );
	///\todo: here need to put in additional initialization stuff from base class
	lookup()[name] = this;
	// This funny call is used to ensure that the root element is
	// created at static initialization time.
	// Element::root();
}

Cinfo::~Cinfo()
{
	unsigned int i;
	unsigned int start = 0;
	if ( baseCinfo_ )
		start = baseCinfo_->finfos_.size();
	for ( i = start; i < finfos_.size(); i++ )
		delete finfos_[i];
	/*
	vector< Finfo* >::iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ )
		delete *i;
		*/
	delete thisFinfo_;
	delete noDelFinfo_;
}

const Cinfo* Cinfo::find( const string& name )
{
	map<string, Cinfo*>::iterator i;
	i = lookup().find(name);
	if ( i != lookup().end() )
		return i->second;
	return 0;
}

const Finfo* Cinfo::findFinfo( Element* e, const string& name ) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		const Finfo* ret = (*i)->match( e, name );
		if ( ret )
			return ret;
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != 0 && base_ != this)
		return base_->findFinfo( e, name );

	return 0;
}

const Finfo* Cinfo::findFinfo( 
		const Element* e, unsigned int connIndex) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		const Finfo* ret = (*i)->match( e, connIndex );
		if ( ret )
			return ret;
	}

	// Fallthrough. No matches were found, so ask the base class.
	// This could be problematic, if the base class indices disagree
	// with the child class.
	///\todo: Figure out how to manage base class index alignment here
	if ( base_ && base_ != this)
		return base_->findFinfo( e, connIndex );

	return 0;
}

const Finfo* Cinfo::findFinfo( const string& name ) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		if ( (*i)->name() == name )
				return (*i);
	}

	// Fallthrough. No matches were found, so ask the base class.
	if (base_ != 0 && base_ != this)
		return base_->findFinfo( name );

	return 0;
}

/*
void Cinfo::listFinfos( vector< Finfo* >& ret ) const
{
	ret.insert( ret.end(), finfos_.begin(), finfos.end() );
	
	// for ( unsigned int i = 0; i < nFinfos; i++ )
		// ret.push_back( finfoArray_[ i ] );

	if (base_ != this)
		const_cast< Cinfo* >( base_ )->listFinfos( ret );
}
*/


// Called by main() when starting up.
void Cinfo::initialize()
{
		/*
	map<string, Cinfo*>::iterator i;
	for (i = lookup().begin(); i != lookup().end(); i++) {
		// Make the Cinfo object on /classes
		Element* e = new CinfoWrapper( i->first, i->second );
		Element::classes()->adoptChild( e );

		// Identify base classes
		const Cinfo* c = find(i->second->baseName_);
		if (!c) {
			cerr << "Error: Cinfo::initalize(): Invalid base name '" <<
				i->second->baseName_ << "'\n";
			exit(0);
		}
		i->second->base_ = c;
	}

	// Do the field inits after the classes are inited, otherwise
	// there are unfilled base_ pointers.
	for (i = lookup().begin(); i != lookup().end(); i++) {
		// Initialize field equivalences
		for (unsigned int k = 0; k < i->second->nFields_; k++) {
			i->second->fieldArray_[k]->initialize( i->second );
		}
	}
	*/
}

std::map<std::string, Cinfo*>& Cinfo::lookup()
{
	static std::map<std::string, Cinfo*> lookup_;
	return lookup_;
}

/**
 * Create a new element with provided data, a set of Finfos and
 * the MsgSrc and MsgDest allocated.
 */
Element* Cinfo::create( const std::string& name, void* data,
				bool noDeleteFlag ) const
{
	SimpleElement* ret = 
		new SimpleElement( name, nSrc_, nDest_, data );
	if ( noDeleteFlag )
		ret->addFinfo( noDelFinfo_ );
	else
		ret->addFinfo( thisFinfo_ );
	set( ret, "postCreate" );
	return ret;
}

/**
 * Create a new element, complete with data, a set of Finfos and
 * the MsgSrc and MsgDest allocated.
 */
Element* Cinfo::create( const std::string& name ) const
{
	return create( name, ftype_->create( 1 ) );
}

/**
 * listFinfo fills in the finfo list onto the flist.
 * \todo: Should we nest the finfos in Cinfo? Or should we only show
 * the deepest layer?
 */
void Cinfo::listFinfos( vector< const Finfo* >& flist ) const
{
	flist.insert( flist.end(), finfos_.begin(), finfos_.end() );
}

/**
 * Looks up the slotIndex for the finfo specified by name.
 * This is either the destIndex_ or srcIndex_, depending on the
 * Finfo class. Used to set up named static indices for various
 * finfos, for use in the send() functions
 */
unsigned int Cinfo::getSlotIndex( const string& name ) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin() ; i < finfos_.end(); i++ ) {
		if ( (*i)->name() == name )
			return (*i)->getSlotIndex();
	}
	return 0;
}
