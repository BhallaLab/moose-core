/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "moose.h"
#include "ThisFinfo.h"
#include "../shell/Shell.h"
#ifdef GENERATE_WRAPPERS
#include "filecheck.h"
#endif
// The three includes below are needed because of the create function
// requiring an instantiable Element class. Could get worse if we
// permit multiple variants of Element, say an array form.

#include "../element/Neutral.h"

// These includes are needed to call set

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
				const Ftype* ftype,
				struct SchedInfo* schedInfo,
				unsigned int nSched
)
		: baseCinfo_(baseCinfo),
		ftype_( ftype ), nMsg_( 0 ), numSrc_( 0 )
{
	const unsigned int nDoc = 2 * 3;
	static string doc[ nDoc ] =
	{
		"Name", "",
		"Author", "",
		"Description", ""
	};
	
	doc[ 1 ] = name;
	doc[ 3 ] = author;
	doc[ 5 ] = description;
	
	init( doc, nDoc, finfoArray, nFinfos, schedInfo, nSched );
}

Cinfo::Cinfo( const string* doc,
				unsigned int nDoc,
				const Cinfo* baseCinfo,
				Finfo** finfoArray,
				unsigned int nFinfos,
				const Ftype* ftype,
				struct SchedInfo* schedInfo,
				unsigned int nSched
)
		: baseCinfo_(baseCinfo),
		ftype_( ftype ), nMsg_( 0 ), numSrc_( 0 )
{
	init( doc, nDoc, finfoArray, nFinfos, schedInfo, nSched );
}

void Cinfo::init( const string* doc,
				unsigned int nDoc,
				Finfo** finfoArray,
				unsigned int nFinfos,
				struct SchedInfo* schedInfo,
				unsigned int nSched )
{
	// Doc-string array should have key/value pairs as entries
	assert( nDoc % 2 == 0 );
	unsigned int i;
	for ( i = 0; i < nDoc; i += 2 )
		doc_[ doc[ i ] ] = doc[ i + 1 ];

	map< const Finfo*, const Finfo* > baseFinfoCopy;
	if ( baseCinfo_ ) {
		nMsg_ = baseCinfo_->nMsg_;
		for ( i = 0; i < baseCinfo_->finfos_.size(); i++ ) {
			Finfo* f = findMatchingFinfo(
				baseCinfo_->finfos_[i]->name(), finfoArray, nFinfos );
			if ( f ) {
				// The inherit operation is true only if the types
				// match.
				bool ret = f->inherit( baseCinfo_->finfos_[i] );
				assert( ret );
				finfos_.push_back( f );
			} else {
				Finfo* orig = baseCinfo_->finfos_[ i ];
				Finfo* copy = orig->copy();
				baseFinfoCopy[ orig ] = copy;
				
				finfos_.push_back( copy );
			}
		}
	}

	// Here we set up the scheduling.
	// First inherit scheduling information from base class
	if ( baseCinfo_ ) {
		vector< SchedInfo >::const_iterator isched;
		for ( isched = baseCinfo_->scheduling_.begin();
			  isched != baseCinfo_->scheduling_.end();
			  isched++ )
		{
			SchedInfo schedCopy = *isched;
			schedCopy.finfo = baseFinfoCopy[ schedCopy.finfo ];
			scheduling_.push_back( schedCopy );
		}
	}

	// Next store own scheduling information
	for ( i = 0; i < nSched; i++ )
		scheduling_.push_back( schedInfo[i] );
	
	/*
	 * Uncomment if you want a list of Cinfo's stored in a text file. Commented
	 * out because it keeps creating the file everywhere you run moose.
	 */
	//~ #ifndef NDEBUG /* If compiling in DEBUG mode. */
	//~ // Printing list of Cinfos
	//~ string filename = "Cinfolist.txt";
	//~ ofstream fout( filename.c_str(), ios_base::app );
	//~ cout << "Printing: " << name() << "with nSched =" << nSched << endl;
	//~ for ( unsigned int i = 0; i < nSched; i++ ){
		//~ fout << name() << endl;
	//~ }
	//~ cout << "Wrote list of Cinfos to " << filename << ".\n";
	//~ fout.close();
	//~ #endif // NDEBUG
	

#ifdef GENERATE_WRAPPERS        
        std::string out_dir_name ="generated/";
        std::string swig_name = out_dir_name+"pymoose.i";    
        std::string header_name = out_dir_name+name()+".h";
        std::string cpp_name = out_dir_name+name()+".cpp";
        std::string initf_name = out_dir_name + "initCinfos.cpp";
        ofstream header, cpp, swig, init_cinfo;
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
            cerr << "Could not create files for " << name() << endl;        
        }
    
        if (created)
        {
            /*
              Insert the common class members,
              constructors and the common method getType()
              and the common field className.
            */
            string baseClass = baseCinfo_? baseCinfo_->name() : "PyMooseBase";
            header << "#ifndef _pymoose_" << name() << "_h\n"
                   << "#define _pymoose_"<< name() << "_h\n"
                   << "#include \"" << baseClass << ".h\"\n"
                   << "namespace pymoose{\n"
                   << "    class PyMooseBase;\n"
                   << "    class " << baseClass << ";\n"
                   << "    class " << name() << " : public " << baseClass << endl
                   << "    {\n"
                   << "      public:\n"
                   << "        static const std::string className_;\n"
                   << "        " << name() << "(std::string className, std::string objectName, Id parentId);\n"
                   << "        " << name() << "(std::string className, std::string path);\n"
                   << "        " << name() << "(std::string className, std::string objectName, PyMooseBase& parent);\n"
                   << "        " << name() << "(Id id);\n"
                   << "        " << name() << "(std::string path);\n"
                   << "        " << name() << "(std::string name, Id parentId);\n"
                   << "        " << name() << "(std::string name, PyMooseBase& parent);\n"
                   << "        " << name() << "( const " << name() << "& src, std::string name, PyMooseBase& parent);\n"
                   << "        " << name() << "( const " << name() << "& src, std::string name, Id& parent);\n"
                   << "        " << name() << "( const " << name() << "& src, std::string path);\n"
                   << "        " << name() << "( const Id& src, std::string name, Id& parent);\n"
                   << "        " << name() << "( const Id& src, std::string path);\n"
                   << "        ~" << name() <<"();\n"
                   << "        const std::string& getType();\n";

            cpp << "#ifndef _pymoose_" << name() << "_cpp\n"
                << "#define _pymoose_"<< name() << "_cpp\n"
                << "#include \"" << name() << ".h\"\n"
                << "using namespace pymoose;\n"             
                << "const std::string "<< name() <<"::className_ = \"" << name() <<"\";\n"
                << name() << "::" << name() << "(std::string className, std::string objectName, Id parentId):" << baseClass << "(className, objectName, parentId){}\n"
                << name() << "::" << name() << "(std::string className, std::string path):" << baseClass << "(className, path){}\n"
                << name() << "::" << name() << "(std::string className, std::string objectName, PyMooseBase& parent):" << baseClass << "(className, objectName, parent){}\n"
                << name() << "::" << name() << "(Id id):" << baseClass << "(id){}\n"
                << name() << "::" << name() <<"(std::string path):" << baseClass << "(className_, path){}\n"
                << name() << "::" << name() << "(std::string name, Id parentId):" << baseClass << "(className_, name, parentId){}\n"
                << name() << "::" << name() << "(std::string name, PyMooseBase& parent):" << baseClass << "(className_, name, parent){}\n"
                << name() << "::" << name() << "(const " << name() << "& src, std::string objectName, PyMooseBase& parent):" << baseClass << "(src, objectName, parent){}\n"
                << name() << "::" << name() << "(const " << name() << "& src, std::string objectName, Id& parent):" << baseClass << "(src, objectName, parent){}\n"
                << name() << "::" << name() << "(const " << name() << "& src, std::string path):" << baseClass << "(src, path){}\n"
                << name() << "::" << name() << "(const Id& src, std::string name, Id& parent):" << baseClass << "(src, name, parent){}\n"
                << name() << "::" << name() << "(const Id& src, std::string path):" << baseClass << "(src, path){}\n"
                << name() << "::~" << name() <<"(){}\n"
                << "const std::string& "<< name() <<"::getType(){ return className_; }\n";
            swig << "%include \"" << name() << ".h\"\n";        
        }
#endif // GENERATE_WRAPPERS
	for ( i = 0 ; i < nFinfos; i++ ) {
            finfoArray[i]->countMessages( nMsg_ );
			// This sends in the new Cinfo name needed to set up the
			// FuncVecs within the finfoArray.
			finfoArray[i]->addFuncVec( name() );

            finfos_.push_back( finfoArray[i] );
#ifdef GENERATE_WRAPPERS                
            if ( created )
            {                    
                std::string fieldName = finfoArray[i]->name();
                std::string fieldType = finfoArray[i]->ftype()->getTemplateParameters();
                                
                if (fieldType != "void" && dynamic_cast<ValueFinfo*>(finfoArray[i]) != NULL)
                {
                    /* Insert the getters and setters
                       Strings require special treatment as they may
                       get deleted from the stack, so we use
                       PyMooseBase::getField() to return a reference
                       to the context_->fieldValue member.
                     */                  
                    header << "            " << ((fieldType == "string") ? "const string& " : fieldType) << " __get_" << fieldName << "() const;\n";
                
                    cpp << ((fieldType == "string") ? "const string& " : fieldType) <<" " <<  name() << "::__get_" <<  fieldName << "() const\n"
                        << "{\n";
                    if (fieldType != "string"){
                        cpp << "    " << fieldType << " " << fieldName << ";\n"
                            << "    get < " << fieldType << " > (id_(), \""<< fieldName << "\"," << fieldName << ");\n"
                            << "    return " << fieldName << ";\n";
                    } else {
                        cpp << "return this->getField(\"" << fieldName << "\");\n";
                    }
                    cpp << "}" << endl;
                    swig << "%attribute(pymoose::" << name() << ", " << fieldType << ", " << fieldName << ", __get_" << fieldName;
                    
                    if ( finfoArray[i]->recvFunc() != &dummyFunc) // read-write field - put setter
                    {
                        header << "            void" << " __set_" << fieldName << "(" << fieldType << " " << fieldName << ");\n";
                    
                        cpp << "void " << name() << "::__set_" << fieldName <<"( " << fieldType << " " << fieldName << " )\n"
                            << "{\n"
                            << "    set < " << fieldType << " > (id_(), \"" << fieldName << "\", "<< fieldName << ");\n"
                            << "}" << endl;
                        swig << ", __set_" << fieldName;                        
                    }                    
                    swig << ")\n";                
                }
            }
#endif // GENERATE_WRAPPERS
        }
#ifdef GENERATE_WRAPPERS
        if (created)
        {
            header << "    };\n\n"
                   << "}\n"
                   << "#endif" << endl;
            cpp << "#endif" << endl;
            header.close();
            cpp.close();
            swig.close();        
        }
#endif // GENERATE_WRAPPERS
	
	// Now we shift the DestFinfos to the back of the set.
	// Have to maintain ordering here
	// because the base classes will need consistency.
	// At this time we also assign the msg numbering for all Finfos.
	shuffleFinfos();
        
	thisFinfo_ = new ThisFinfo( this );
	noDelFinfo_ = new ThisFinfo( this, 1 );
	///\todo: here need to put in additional initialization stuff from base class
	if ( name() != "" )
		lookup()[ name() ] = this;
}

Cinfo::~Cinfo()
{
	map< string, Cinfo* >::iterator pos = lookup().find( name() );
	assert( pos != lookup().end() );
	lookup().erase( pos );
	pos = lookup().find( name() );
	assert( pos == lookup().end() );

	/*
	 * I won't delete the allocated Finfos. This is because
	 * they get shuffled around so it is hard to figure out which
	 * belongs to a base class and which is local.
	 * The number of Finfos that accumulate is small, and does not
	 * go up once setup is over, so it can 
	 * wait till the end of the simulation.
	 */
	delete thisFinfo_;
	delete noDelFinfo_;
}

const std::string& Cinfo::name() const
{
	static string blank = "";
	map< string, string >::const_iterator i;
	i = doc_.find( "Name" );
	if ( i != doc_.end() )
		return i->second;
	return blank;
}

const std::string& Cinfo::author() const
{
	static string blank = "";
	map< string, string >::const_iterator i;
	i = doc_.find( "Author" );
	if ( i != doc_.end() )
		return i->second;
	return blank;
}

const std::string& Cinfo::description() const
{
	static string blank = "";
	map< string, string >::const_iterator i;
	i = doc_.find( "Description" );
	if ( i != doc_.end() )
		return i->second;
	return blank;
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

	return 0;
}

const Finfo* Cinfo::findFinfo( 
		const Element* e, const ConnTainer* c ) const
{
	// if ( c->msg2() < finfos_.size() && finfos_[ c->msg2() ].isDest() )
	// 	return finfos_[ c->msg2() ];
	// else if ( c->msg1() < finfos_.size() )
	// 	return finfos_[ c->msg1() ];
	// else
	// 	return 0;
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		const Finfo* ret = (*i)->match( e, c );
		if ( ret )
			return ret;
	}

	/*
	// Fallthrough. No matches were found, so ask the base class.
	// This could be problematic, if the base class indices disagree
	// with the child class.
	///\todo: Figure out how to manage base class index alignment here
	if ( base_ && base_ != this)
		return base_->findFinfo( e, connIndex );
		*/

	return 0;
}

/**
* Returns the Finfo identified by the specified msg number.
* Source Finfos should have a positive index
* pure Dest finfos have a negative index.
* Not all Finfos will have a msgNum, but any valid msgNum 
* should have a Finfo.
*/
const Finfo* Cinfo::findFinfo( int msgNum ) const
{
	if ( msgNum >= 0 && msgNum < static_cast< int >( numSrc_ ) )
		return finfos_[msgNum];
	if ( msgNum < 0 && ( -msgNum < static_cast< int >( finfos_.size() ) ) )
		return finfos_[ -msgNum ];
	return 0;
}

const Finfo* Cinfo::findFinfo( const string& name ) const
{
	vector< Finfo* >::const_iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ ) {
		if ( (*i)->name() == name )
				return (*i);
	}

	return 0;
}

/**
 * Puts SrcFinfos in front, DestFinfos next, and finally ValueFinfos
 * while keeping relative order the same. SharedFinfos can be either
 * Src or Dest depending on their quota of functions.
 *
 * Must be called after the initialization of FuncVecs.
 *
 * Side-effects:
 * - Assign numSrc_.
 * - Put msg_ numbers on each, where the msg_ is just the index.
 *   Note that ValueFinfos do not get messages. That is left to the
 *   DynamicFinfos.
 */
unsigned int Cinfo::shuffleFinfos()
{
	vector< Finfo* > temp;
	vector< Finfo* >::iterator i;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ )
		if ( !( *i )->isDestOnly() ) // Is Src.
			temp.push_back( *i );
	numSrc_ = temp.size();
	for ( i = finfos_.begin(); i != finfos_.end(); i++ )
		if ( ( *i )->isDestOnly() && ( *i )->msg() != INT_MAX )
			temp.push_back( *i );
	for ( i = finfos_.begin(); i != finfos_.end(); i++ )
		if ( ( *i )->isDestOnly() && ( *i )->msg() == INT_MAX )
			temp.push_back( *i );

	assert( temp.size() == finfos_.size() );

	unsigned int j = 0;
	finfos_ = temp;
	for ( i = finfos_.begin(); i != finfos_.end(); i++ )
		( *i )->countMessages( j );
	return numSrc_;
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
Element* Cinfo::create( Id id, const std::string& name, 
			void* data, bool noDeleteFlag ) const
{
	SimpleElement* ret = 
		new SimpleElement( id, name, data, numSrc_ );
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
Element* Cinfo::create( Id id, const std::string& name ) const
{
	return create( id, name, ftype_->create( 1 ) );
}

/**
 * Create a new element with provided data, a set of Finfos and
 * the MsgSrc and MsgDest allocated.
 */
Element* Cinfo::createArray( Id id, const std::string& name, 
			void* data, unsigned int numEntries, size_t objectSize,
			bool noDeleteFlag ) const
{
	ArrayElement* ret = 
		new ArrayElement( id, name,  data, numSrc_, numEntries, objectSize);
		//new ArrayElement( id, name, numSrc_, data, numEntries, objectSize );
	if ( noDeleteFlag )
		ret->addFinfo( noDelFinfo_ );
	else
		ret->addFinfo( thisFinfo_ );
	set( ret, "postCreate" );
	
	return ret;
}

/**
 * Create a new arrayElement, complete with data, a set of Finfos and
 * the MsgSrc and MsgDest allocated.
 */
Element* Cinfo::createArray( Id id, const std::string& name,
	unsigned int numEntries)	const
{
	return createArray( id, name, ftype_->create( numEntries ), 
		numEntries, ftype_->size() );
}

/**
 * Connect up new element to the clock ticks that will control its
 * runtime operations.
 */
bool Cinfo::schedule( Element* e, unsigned int connTainerOption ) const
{
	if ( scheduling_.size() == 0 ) 
		return 1;
        // Subhasis - 2010-04-22 12:13:42 (+0530)
        // Adding autoschedule enabling and disabling code.
        // Right now it is done by querying the autoschedule field of
        // ClockJob object. Not sure it should go into Property.
        Id cjId = Id::localId( "/sched/cj" );
        assert( cjId.good() );
        int autoschedule;
        get<int>(cjId(), "autoschedule", autoschedule);
        if (!autoschedule){
            return 1;
        }
        
	string tickClass = "Tick";
	if ( Shell::numNodes() > 1 )
		tickClass = "ParTick";
	const Cinfo* tickCinfo = find( tickClass );
	const Finfo* procFinfo = tickCinfo->findFinfo( "process" );
	assert( tickCinfo != 0 ); // Not sure about execution order here.
	assert( procFinfo != 0 );

	// Don't bother to schedule objects sitting on /library or /proto
        const Id& libId = Id::localId( "/library" );
        if (!libId.bad()){
            const Element* library = libId();
            if ( e->isDescendant( library )){
                return 1;
            }
        }
        const Id& protoId = Id::localId( "/proto" );
	if (!protoId.bad())
        {
            const Element* proto = protoId();
            if( e->isDescendant( proto ) ){
		return 1;
            }
        }

        

	vector< SchedInfo >::const_iterator i;
	for ( i = scheduling_.begin(); i != scheduling_.end(); i++ ) {
		char line[20];
		sprintf( line, "/sched/cj/t%d", i->tick * 2 + i->stage );
		Id tick = Id::localId( line );
		if ( !tick.good() ) { // Make the clock tick
			sprintf( line, "t%d", i->tick * 2 + i->stage );
			Element* t = Neutral::create( tickClass, line, cjId, 
				Id::scratchId() );
			assert( t != 0 );
			tick = t->id();
		}
		assert( tick.good() );
		tick.eref().add( procFinfo->msg(), e, i->finfo->msg(),
			connTainerOption );
		// procFinfo->add( tick(), e, i->finfo );
	}
	
	// cout << "scheduling new object " << e->name() << endl;
	return 1;
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
Slot Cinfo::getSlot( const string& name ) const
{
	vector< Finfo* >::const_iterator i;
	Slot ret;
	for ( i = finfos_.begin() ; i < finfos_.end(); i++ ) {
		if ( (*i)->getSlot( name, ret ) )
			return ret;
//		if ( (*i)->name() == name )
//			return (*i)->getSlotIndex();
	}
	cout << "Warning: Failed to find slot: " << name << endl;
	return ret;
}

bool Cinfo::isA( const Cinfo* other ) const {
	if ( ftype_->isSameType( other->ftype_ ) )
		return 1;
	if ( baseCinfo_ )
		return baseCinfo_->isA( other );
	else
		return 0;
}
