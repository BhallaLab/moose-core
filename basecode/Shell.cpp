#include "header.h"
#include "../builtins/Interpol.h"
#include "Shell.h"
#include "ShellWrapper.h"
#include "ExtFieldFinfo.h"

// bool splitFieldString( const string& field, string& e, string& f );
//bool splitField( const string& fieldstr, Field& f );

//////////////////////////////////////////////////////////////////
// SimDumpInfo functions
//////////////////////////////////////////////////////////////////
unsigned int parseArgs( const string& in, vector< string >& out )
{
	static const char* separator = " ,";
	string back = in;
	unsigned long pos = 0;

	while ( pos != string::npos ) {
		back = back.substr( pos );
		pos = back.find_first_not_of( separator );
		if ( pos == string::npos )
			break;
		back = back.substr( pos );
		pos = back.find_first_of( separator );
		string front = back.substr( 0, pos );
		out.push_back( front );
	}
	return out.size();
}

SimDumpInfo::SimDumpInfo(
	const string& oldObject, const string& newObject,
			const string& oldFields, const string& newFields)
			: oldObject_( oldObject ), newObject_( newObject )
{
	vector< string > oldList;
	vector< string > newList;

	parseArgs( oldFields, oldList );
	parseArgs( newFields, newList );

	if ( oldList.size() != newList.size() ) {
		cout << "Error: SimDumpInfo::SimDumpInfo: field list length diffs:\n" << oldFields << "\n" << newFields << "\n";
		return;
	}
	for ( unsigned int i = 0; i < oldList.size(); i++ )
		fields_[ oldList[ i ] ] = newList[ i ];
}

// Takes info from simobjdump
void SimDumpInfo::setFieldSequence( int argc, const char** argv )
{
	string blank = "";
	fieldSequence_.resize( 0 );
	for ( int i = 0; i < argc; i++ ) {
		string temp( argv[ i ] );
		map< string, string >::iterator j = fields_.find( temp );
		if ( j != fields_.end() )
			fieldSequence_.push_back( j->second );
		else 
			fieldSequence_.push_back( blank );
	}
}

bool SimDumpInfo::setFields( Element* e, int argc, const char** argv )
{
	if ( static_cast< unsigned int >(argc) != fieldSequence_.size() ) {
		cout << "Error: SimDumpInfo::setFields:: Number of argument mismatch\n";
		return 0;
	}
	for ( int i = 0; i < argc; i++ ) {
		if ( fieldSequence_[ i ].length() > 0 )
		{
			if ( Field( e, fieldSequence_[ i ] ).set( argv[ i ] ) == 0 )
			{
				cout << "Error: SimDumpInfo::setFields:: Failed to set '";
				cout << e->path() << "/" << 
					fieldSequence_[ i ] << " = " << argv[ i ] << "'\n";
				return 0;
			}
		}
	}
	return 1;
}

//////////////////////////////////////////////////////////////////
// Shell functions
//////////////////////////////////////////////////////////////////


Shell::Shell( Element* wrapper )
	: workingElement_( "/" ), wrapper_( wrapper ), recentElement_( 0 )
{
	string className = "molecule";
	vector< SimDumpInfo *> sid;
	// Here we initialize some simdump conversions. Several things here
	// are for backward compatibility. Need to think about how to
	// eventually evolve out of these. Perhaps SBML.
	sid.push_back( new SimDumpInfo(
		"kpool", "Molecule", 
		"n nInit vol slave_enable", 
		"n nInit volumeScale slaveEnable") );
	sid.push_back( new SimDumpInfo(
		"kreac", "Reaction", "kf kb", "kf kb") );
	sid.push_back( new SimDumpInfo( "kenz", "Enzyme",
		"k1 k2 k3 usecomplex",
		"k1 k2 k3 mode") );
	sid.push_back( new SimDumpInfo( "xtab", "Table",
	"input output step_mode stepsize",
	"input output mode stepsize" ) );

	sid.push_back( new SimDumpInfo( "group", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xgraph", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xplot", "Plot", "", "" ) );

	sid.push_back( new SimDumpInfo( "geometry", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xcoredraw", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xtree", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xtext", "Neutral", "", "" ) );

	sid.push_back( new SimDumpInfo( "kchan", "ConcChan",
		"perm Vm",
		"permeability Vm" ) );

	for (unsigned int i = 0 ; i < sid.size(); i++ ) {
		dumpConverter_[ sid[ i ]->oldObject() ] = sid[ i ];
	}
}

///////////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////////
void Shell::addFuncLocal( const string& src, const string& dest )
{
	Field s;
	Field d;
	if ( splitField( src, s ) && splitField ( dest, d ) ) {
		if ( s.add ( d ) ) {
			ok();
			return;
		}
	}
	error( string ("Unable to add msg from ") + src + " to " + dest);
}

void Shell::dropFuncLocal( const string& src, const string& dest )
{
	Field s;
	Field d;
	if ( splitField( src, s ) && splitField ( dest, d ) ) {
		s.drop ( d );
		ok();
	} else {
		error(
			string ("Unable to drop msg from ") + src + " to " + dest
		);
	}
}

void Shell::setFuncLocal( const string& field, const string& value )
{
	vector < Field > f;
	if ( wildcardField (field, f ) > 0 ) {
		vector< Field >::iterator i;
		for ( i = f.begin(); i != f.end(); i++ )
			i->set( value );
		ok();
		return;
	}
	error( "Unable to set field ", field );
}

string Shell::getFuncLocal( const string& field ) {
	Field f;
	if ( splitField (field, f ) ) {
		string value;
		if ( f.get( value ) )
			return value;
		else 
			error( "unable to get field", field );
	}
	return "";
}

void Shell::createFuncLocal( const string& type, const string& path )
{
//	cout << "doing create with " << type << ", " << path <<"\n";
	Element* parent;
	string name;
	const Cinfo* ci = Cinfo::find( type );

	if ( ci ) {
		Element* cwe = checkWorkingElement( );
		unsigned long pos = path.rfind( '/' );

		if ( pos == string::npos ) { // no slash
			parent = cwe;
			name = path;
		} else {
			parent = cwe->relativeFind( path.substr( 0, pos + 1 ) );
			name = path.substr( pos + 1 );
		}
		if ( isalpha( name[0] ) && 
			name.find_first_of( " 	$%&;,{}()!#\'\\\"\n`~?") ==
			string::npos )
		{
			if ( parent ) {
				if ( parent->relativeFind( name ) ) {
					error( "create: object already exists", path );
					return;
				}
				Element* ret = ci->create( name, parent );
				if ( !ret ) {
					error( "create: Failed to create object", path );
				} else {
					recentElement_ = ret;
					ok();
				}
			} else {
				error( "create: Failed to find parent object ", 
					path.substr( 0, pos ) );
			}
		} else {
			error( "create: Illegal object name", name );
		}
	} else {
		error( "create: Failed to find class", type );
	}
}

void Shell::deleteFuncLocal( const string& path )
{
	Element* e = findElement( path );
	if ( e && e != Element::root() ) {
		delete e ;
		ok();
	} else {
		error( "delete: cannot delete", path );
	}
}

// This function parses the dest string and checks whether it can
// handle children. It passes back the name of the dest child if
// this is to be renamed, otherwise the length of the name is zero.
// It returns the ptr to the dest parent on success.
Element* Shell::findDest( const string& dest, string& destChildName )
{
	Element* d = findElement( dest );
	destChildName = "";

	if ( !d ) { // Perhaps it is being moved to a new name.
		string parentName;
		splitFieldString( dest, parentName, destChildName);
		if ( destChildName.length() == 0) {
			error("FindDest::Failed to find dest element ", dest );
			return 0;
		}
		if ( parentName == "" )
			d = checkWorkingElement();
		else 
			d = findElement( parentName );
	}

	if ( !d ) {
		error("FindDest::Failed to find dest element ", dest );
		return 0;
	}

	Field newParent = d->field( "child_out" );
	if ( !newParent.good() ) {
		error ( "Dest element ", dest + " does not like children" );
		return 0;
	}
	return d;
}

void Shell::moveFuncLocal( const string& src, const string& dest ) {
	Element* s = findElement( src );

	if ( !s ) {
		error( "move: Failed to find src element ", src );
		return;
	}
	string newName = "";
	Element* d;
	if ( dest.find( "/") == string::npos ) {
		if ( dest == "." ) {
			d = checkWorkingElement();
		} else if ( dest == ".." ) {
			d = checkWorkingElement()->parent();
		} else {
			d = checkWorkingElement();
			newName = dest;
		}
	} else {
		d = findDest( dest, newName );
	}
	if ( !d ) {
		error( "Move: destination element", dest + "does not exist" );
		return;
	}
	if ( d->descendsFrom( s ) ) {
		error( "move: cannot move '", 
			src + "' onto itself, '" + d->path() + "'" );
		return;
	}

	Field childIn = s->field( "child_in" );
	Field childOut = s->parent()->field( "child_out" );
	Field newParent = d->field( "child_out" );

	if ( !childOut.drop( childIn ) ) {
		error( "move: Failed to remove",
			src + " from original parent");
		return;
	}
	if ( !newParent.add( childIn ) ) {
		error( "move: Failed to add", src + " to " + dest );
		return;
	}
	if ( newName.length() > 0 )
		s->field( "name" ).set( newName );
	ok();
}

void Shell::copyFuncLocal( const string& src, const string& dest )
{
	Element* s = findElement( src );

	if ( !s ) {
		error( "copy: Failed to find src element ", src );
		return;
	}
	string newName = "";
	Element* d = findDest( dest, newName );
	if ( !d ) return;
	if ( d->descendsFrom( s ) ) {
		error( "copy: cannot copy '", 
			src + "' onto itself, ' " + d->path() + "'" );
		return;
	}

	Element* e = s->deepCopy( d );
	if ( newName.length() > 0 )
		e->field( "name" ).set( newName );
	ok();
}

void Shell::copyShallowFuncLocal( const string& src, const string& dest)
{
	Element* s = findElement( src );

	if ( !s ) {
		error( "copy: Failed to find src element ", src );
		return;
	}
	string newName = "";
	Element* d = findDest( dest, newName );
	if ( !d ) return;

	Element* e = s->shallowCopy( d );
	if ( newName.length() > 0 )
		e->field( "name" ).set( newName );
	ok();
}

void Shell::copyHaloFuncLocal( const string& src, const string& dest ) {
}

Element* Shell::shellRelativeFind( const string& path )
{
	Element* cwe = findElement( workingElement_ );
	if ( !cwe )
		cwe = Element::root();

	return cwe->relativeFind( path );
}

void Shell::ceFuncLocal( const string& newpath ) {
	// Element* nwe = shellRelativeFind( newpath );
	Element* nwe = findElement( newpath );
	if ( nwe )
		workingElement_ = nwe->path();
	else
		workingElement_ = "/";
}

void Shell::pweFuncLocal( ) {
	cout << workingElement_ << "\n";;
}

void Shell::pusheFuncLocal( const string& newpath ) {
	// Element* nwe = shellRelativeFind( newpath );
	Element* nwe = findElement( newpath );
	if ( nwe ) {
		workingElementStack_.push_back( workingElement_ );
		workingElement_ = nwe->path();
		cout << workingElement_ << "\n";
	} else {
		cout << "Can't find element " << newpath << "\n";
	}
}

void Shell::popeFuncLocal(  ) {
	if ( workingElementStack_.size() > 0 ) {
		workingElement_ = workingElementStack_.back();
		workingElementStack_.pop_back();
		cout << workingElement_ << "\n";
	} else {
		cout << "** Error: Empty element stack\n";
	}
}

void Shell::aliasFuncLocal( 
	const string& origfunc, const string& newfunc )
{
	Field f;
	splitField( parser_ + "/alias", f );
	f.set( origfunc + ", " + newfunc );
	aliasMap_[ origfunc ] = newfunc;
}

void Shell::quitFuncLocal(  ) {
	cout << "Quitting Moose\n";
	exit( 0 );
}

void Shell::stopFuncLocal(  ) {
}
void Shell::resetFuncLocal(  ) {
	cout << "doing reset\n";
	Field( "/sched/cj/resetIn" ).set( "" );
}

void Shell::stepFuncLocal( const string& stepTime,
	const string& option )
{
	Field( "/sched/cj/runTime").set( stepTime );
	ProcInfoBase b( wrapper_->path() );
	Field f( "/sched/startIn" );
	f.set( "/sched/cj/startIn, " + wrapper_->path() );
}

// Assigns a dt to a clock. If the clock does not exist, create it.
// Only clock0 exists by default
void Shell::setClockFuncLocal( 
	const string& clockNo, const string& dt, const string& stage )
{
	string clockName = string( "/sched/cj/ct" ) + clockNo;

	Field f( clockName + "/dt" );
	if ( !f.good() ) {
		createFuncLocal( "ClockTick", clockName );
		f = Field( clockName + "/dt" );
		if ( !f.good() ) {
			cerr << "Error: Shell::setClockFuncLocal(): Failed to create clock/dt " << clockName << "\n";
			return;
		}
	}
	f.set( dt );
	Field ( clockName + "/stage" ).set( stage );
	ok();
}

void Shell::showClocksFuncLocal()
{
	vector< Field > f;
	Field( "/sched/cj/child_out" ).dest( f );
	vector< Field >::iterator i;
	cout << "ACTIVE CLOCKS\n";
	cout << "-------------\n";
	for ( i = f.begin(); i != f.end(); i++ ) {
		string dt;
		string stage;
		i->getElement()->field( "dt" ).get( dt );
		i->getElement()->field( "stage" ).get( stage );
		cout << i->getElement()->name() << "	:	" << dt << 
			"	( stage " << stage << " )\n";
	}
	cout << "\n";
}

void Shell::useClockFuncLocal( const string& path, 
	const string& clockNo )
{
	Field f( string( "/sched/cj/ct" ) + clockNo + "/path" );
	if ( f.good() ) {
		if ( f.set( path ) ) {
			ok();
			return;
		}
	}

	cerr << "Error: Shell::useClockFuncLocal( " << path << ", " <<
		clockNo << " ): failed\n"; 
}

void Shell::callFuncLocal( const string& args ) {
}

// This function is considerably different from the old GENESIS version,
// because the structure of messaging has changed.
// The field looks up the message and object being queried.
// The index looks up which of the src/dest to use
// The useSrc flag tells it to look for Src vs Dest.
// Still need to work out mapping to TYPE for SLI
string Shell::getmsgFuncLocal(
	const string& field, const string& options )
{
	Field f;
	if ( splitField( field, f ) ) {
		bool useSrc = ( options.find( "-in" ) != string::npos ) ;
		bool doCount = ( options.find( "-c" ) );

		vector< Field > list;
		if ( useSrc )
			f.src( list );
		else 
			f.dest( list );

		if ( doCount ) {
			char ret[20];
			sprintf( ret, "%d", static_cast< int >( list.size() ) );
			return ret;
		} else { // Assume options is just a number with the index
			unsigned int index;
			sscanf( options.c_str(), "%u", &index );
			if ( index < list.size() )
				return list[ index ].path();
		}
	}
	error( "getmsgFuncLocal: Unknown options: ", options );
	return "";
}

// Need to elaborate on this. For example, handling the elm itself.
int Shell::isaFuncLocal( const string& type, const string& field ) {
	Field f;
	if ( splitField( field, f ) ) {
		if ( f->name() == type )
			return 1;
		Field other;
		if ( splitField( type, other ) )
			if ( f->name() == other->name() )
				return 1;
	}
	return 0;
}

int Shell::existsFuncLocal( const string& fieldstr ) {
	Field f;
	return splitField( fieldstr, f );
}

// Here we refer to the composite elm.finfo as the field.
void Shell::showFuncLocal( const string& field )
{
	string ename, fname;
	splitFieldString( field, ename, fname );
	Element* e = findElement( ename );
	/*
	Element* e = checkWorkingElement();
	if ( ename != "" )
		e = e->relativeFind( ename ); // later expand to elist.
		*/
	if ( e ) {
		// const Cinfo* ci = e->cinfo();
		cout << "\n[ " << e->path() << " ]\n";
		if ( fname == "*" ) {
			vector< Finfo* > flist;
			vector< Finfo* >::iterator i;
			// ci->listFields( flist );
			e->listFields( flist );
			for ( i = flist.begin(); i != flist.end(); i++ ) {
				string value;
				if ( ( *i )->strGet( e, value ) ) 
				cout << ( *i )->name() << "			= " <<
					value << "\n";
			}
		} else {
			// Field f = ci->field( fname );
			Field f = e->field( fname );
			if ( f.good() ) {
				string value;
				if ( f->strGet( e, value ) ) 
				cout << ( f )->name() << "			= " <<
					value << "\n";
			}
		}
	}
}

void Shell::showmsgFuncLocal( const string& field ) {
}
void Shell::showobjectFuncLocal( const string& classname ) {
}

void Shell::leFuncLocal( const string& start ) {
	Element* s = findElement( start );
	/*
	Element* s = checkWorkingElement();
	s = s->relativeFind( start );
	*/
	if ( s ) {
		vector< Field > f;
		s->field( "child_out" ).dest( f );
		vector< Field >::iterator i;
		for ( i = f.begin(); i != f.end(); i++ )
			cout << i->getElement()->name() << "\n";
	} else {
		cout << "Warning: '" << start <<
			"' is not a valid element path\n";
	}
}

// Just do a call onto the parser.
void Shell::listCommandsFuncLocal( ) {
	Field f;
	splitField( parser_ + "/listcommands", f );
	f.set( "" );
}

void Shell::listClassesFuncLocal( ) {
	leFuncLocal( "/classes" );
}

// For now, option 0 -> ordinary newline, option 1 -> no newline. 
void Shell::echoFuncLocal( vector< string >& s, int options ) {
	vector< string >::iterator i;

	if ( isInteractive_ ) {
		for ( i = s.begin(); i != s.end(); i++ )
			cout << *i << " ";
		if ( options != 1 )
			cout << "\n";
	} else {
		for ( i = s.begin(); i != s.end(); i++ )
			response_ += *i + " ";
		if ( options != 1 )
			response_ += "\n";
	}
}

void Shell::commandFuncLocal( int argc, const char** argv )
{
	if ( argc == 0 )
		return;

	string funcname;
	map< string, string >:: iterator i = aliasMap_.find( argv[ 0 ] );
	if ( i != aliasMap_.end() )
		funcname = i->second;
	else
		funcname = argv[ 0 ];

	if ( funcname == "simundump" )
		simundumpFunc( argc, argv );
	if ( funcname == "simobjdump" )
		simobjdumpFunc( argc, argv );
	if ( funcname == "loadtab" )
		loadtabFunc( argc, argv );
	if ( funcname == "readcell" )
		readcellFunc( argc, argv );
	if ( funcname == "setupalpha" )
		setupAlphaFunc( argc, argv, 0 );
	if ( funcname == "setuptau" )
		setupAlphaFunc( argc, argv, 1 );
	if ( funcname == "tweakalpha" )
		tweakFunc( argc, argv, 0 );
	if ( funcname == "tweaktau" )
		tweakFunc( argc, argv, 1 );
	if ( funcname == "addfield" )
		addFieldFunc( argc, argv );
}

void Shell::simobjdumpFunc( int argc, const char** argv )
{
	if ( argc < 3 )
		return;
	string name = argv[ 1 ];
	map< string, SimDumpInfo* >::iterator i = 
		dumpConverter_.find( name );
	if ( i != dumpConverter_.end() ) {
		i->second->setFieldSequence( argc - 2, argv + 2 );
	}
}

void Shell::simundumpFunc( int argc, const char** argv )
{
	// use a map to associate class with sequence of fields, 
	// as set up in default and also with simobjdump
	if (argc < 4 ) {
		error( string("usage: ") + argv[ 0 ] +
			" class path clock [fields...]");
		return;
	}
	string oldClassName = argv[ 1 ];
	string path = argv[ 2 ];
	map< string, SimDumpInfo*  >::iterator i;

	i = dumpConverter_.find( oldClassName );
	if ( i == dumpConverter_.end() ) {
		error( string("simundumpFunc: old class name '") + 
			oldClassName + "' not entered into simobjdump" );
		return;
	}

	Element* e = Element::root()->relativeFind( path );
	if ( !e ) {
		string epath;
		string f;
		if ( !splitFieldString( path , epath, f ) ) {
			error( "simundumpFunc: bad path" );
			return;
		}
		Element* parent = Element::root()->relativeFind( epath );
		if ( !parent ) {
			error( "simundumpFunc: bad parent path" );
			return;
		}

		string newClassName = i->second->newObject();

		const Cinfo* ci = Cinfo::find( newClassName );
		if ( !ci ) {
			error( string("simundumpFunc: no class ") + argv[ 1 ]);
			return;
		}
		e = ci->create( f, parent );
	}
	if ( !e ) {
			error( string( "simundumpFunc: Failed to create element" ) +
				argv[ 2 ] );
			return;
	}

	i->second->setFields( e, argc - 4, argv + 4 );
}

void Shell::loadtabFunc( int argc, const char** argv )
{
	static Interpol ip; 
	static int lastIndex = 0;
	static Element* e = 0;
	// Static because the -continue flag in loadtab requires that
	// the old interpol remain available.

	int i = 0;
	int j = 0;
	int xdivs = 0;


	if (argc < 2 ) {
		error( string("usage: ") + argv[ 0 ] +
		"loadtab element table calc_mode xdivs xmin xmax [values...]\nor, for continuation loadtabs: -cont [values...]");
		return;
	}

	if ( strncmp( argv[1], "-c", 2 ) == 0 ) { // loadtab -continue
		if ( e == 0 || e == Element::root() ) {
			error( "loadtab -continue called without inital loadtab\n");
			return;
		}
		xdivs = ip.localGetXdivs();
		j = lastIndex;
		for ( i = 1; i < argc && j <= xdivs; i++ )
			ip.setTableValue( atof( argv[ i ] ), j++ );
	} else { // Start of loadtab.
		if ( argc < 8 ) {
			error( string("usage: ") + argv[ 0 ] +
			"loadtab element table calc_mode xdivs xmin xmax [values...]\nor, for continuation loadtabs: -cont [values...]");
			return;
		}
		e = Element::root()->relativeFind( argv[1] );
		if ( !e || e == Element::root() ) {
			error( string( argv[ 0 ] ) +
			": could not find element '" + argv[1] + "'");
			return;
		}
		ip.localSetMode( atoi( argv[3] ) );
		// e->field( "mode" ).set( argv[3] );
		xdivs = atoi( argv[4] );
		if ( xdivs < 1 ) {
			error( e->path() + ": loadtab: Must specify xdivs > 0 \n" );
			e = 0;
			return;
		}
		ip.localSetXdivs( xdivs );
		ip.localSetXmin( atoi( argv[5] ) );
		ip.localSetXmax( atoi( argv[6] ) );
		// e->field( "xdivs" ).set( argv[4] );
		// e->field( "xmin" ).set( argv[5] );
		// e->field( "xmax" ).set( argv[6] );
		j = 0;
		for ( i = 7; i < argc && j <= xdivs; i++ )
			ip.setTableValue( atof( argv[ i ] ), j++ );
	}
	if ( j == xdivs + 1 ) { // Presumably it is done
		Field tabip( e, "table" );
		Ftype1< Interpol >::set( e, tabip.getFinfo(), ip );
		lastIndex = 0;
		e = 0;
	} else {
		lastIndex = j;
	}
}

void Shell::addFieldFunc( int argc, const char** argv )
{
	if ( argc < 3 ) {
		error( "usage: ", string( argv[0] ) + "element field_name [field_type]");
		return;
	}
	Element* e = findElement( argv[ 1 ] );
	if ( !e ) {
		error( "Shell::addFieldFunc: Failed to find element", argv[1] );
		return;
	}
	string fieldType = "String";
	if ( argc == 4 )
		fieldType = argv[3];

	Finfo* f;
	if ( 
		fieldType == "Bool" || fieldType == "bool" ||
		fieldType == "Int" || fieldType == "int" ||
		fieldType == "Long" || fieldType == "long" ||
		fieldType == "Short" || fieldType == "short"
		) {
			f = new ExtFieldFinfo< int >( argv[2], "Int" );
	} else if (
		fieldType == "Float" || fieldType == "float" ||
		fieldType == "Double" || fieldType == "double"
		) {
			f = new ExtFieldFinfo< double >( argv[2], "Double" );
	} else {
			f = new ExtFieldFinfo< string >( argv[2], "String" );
	}

	e->appendRelay( f );
}


///////////////////////////////////////////////////////
// Utility functions.
///////////////////////////////////////////////////////

// Returns 1 if it can find a / to split the string on.
// Splits string field into e: element part and f: finfo part.
// Does so by finding the last / and splitting the string there.
bool Shell::splitFieldString( const string& field, string& e, string& f)
{
	unsigned long pos = field.rfind( '/' );
	if ( pos == string::npos ) {
		e = "";
		return 0;
	} else {
		if ( pos == 0 )
			e = "/";
		else 
			e = field.substr( 0, pos );
		f = field.substr( pos + 1 );
		return 1;
	}
}

// Builds a field list out of the wildcard + field name path.
// Returns number of fields found.
int Shell::wildcardField( const string& fieldstr, vector< Field >& f )
{
	string ename, fname;

	string path;
	if ( fieldstr[0] == '.' && fieldstr[1] == '/' ) {
		path = workingElement_ + fieldstr.substr( 1 );
	} else if ( fieldstr[0] != '/' && fieldstr[0] != '^' ) {
		if ( workingElement_ == "/" )
			path = workingElement_ + fieldstr;
		else
			path = workingElement_ + "/" + fieldstr;
	} else {
		path = fieldstr;
	}
	splitFieldString( path, ename, fname );

	vector< Element* > elist;
	f.resize( 0 );

	if ( ename == "^" ) {
		if ( recentElement_ != 0 )
			elist.push_back( recentElement_ );
	} else if ( ename == "." ) {
		elist.push_back( checkWorkingElement() );
	} else if ( ename != "" ) {
		Element::wildcardFind( ename, elist );
	} else
		return 0;

	if ( elist.size() > 0 ) {
		vector< Element* >::iterator i;
		for ( i = elist.begin(); i != elist.end(); i++ ) {
			Field temp = (*i)->field( fname );
			if ( temp.good() )
				f.push_back( temp );
		}
	}
	return f.size();
}

// Returns 1 if it succeeds.
// Figures out element and finfo parts.
bool Shell::splitField( const string& fieldstr, Field& f )
{
	string ename, fname;
	splitFieldString( fieldstr, ename, fname );
	Element* e = checkWorkingElement();
	if ( ename == "^" )
		e = recentElement_;
	if ( ename != "" )
		e = e->relativeFind( ename ); // later expand to elist.
	if ( e ) {
		f = e->field( fname );
		if ( f.good() ) {
			return 1;
		}
	}
	return 0;
}

Element* Shell::checkWorkingElement( )
{
	Element* cwe = findElement( workingElement_ );
	if ( cwe )
		return cwe;
	
	workingElement_ = "/";
	return Element::root();
}

Element* Shell::findElement( const string& path )
{
	if ( path == "" || path == "/" || path == "/root" )
		return Element::root();
	if ( path == "root" && workingElement_ == "/" )
		return Element::root();
	if ( path == "^" )
		return recentElement_;
	if ( path.find( '/' ) == 0 )
		 return Element::root()->relativeFind( path.substr(1) );
	else if ( path.find( "../" ) == 0 )
		 return checkWorkingElement()->relativeFind( path );
	else if ( path.find( "./" ) == 0 ) 
		 return checkWorkingElement()->relativeFind( path.substr( 2 ) );
	else
		 return findElement( workingElement_ + "/" + path );
}
void Shell::ok()
{
	if ( isInteractive_ )
		cout << "OK\n";
}
void Shell::error( const string& report )
{
	if ( isInteractive_ )
		cout << "Error: " << report << "\n";
}
void Shell::error( const string& s1, const string& s2 )
{
	if ( isInteractive_ )
		cout << "Error: " << s1 << " " << s2 << "\n";
}

#ifdef DO_UNIT_TESTS
void testParseArgs()
{
	cout << "Testing Shell: parseArgs";
	string in = " a , b , c,d,e f g        hohohohoho    ";
	vector< string > out;
	int ret = parseArgs( in, out );
	if ( ret == 8 ) {
		cout << ".";
	} else {
		cout << "!\nFailed: Number of args = " << ret << " != 8\n";
		return;
	}
	if ( out[ 0 ] == "a" ) cout << ".";
	else cout << "!\nFailed 1\n";
	if ( out[ 1 ] == "b" ) cout << ".";
	else cout << "!\nFailed 2\n";
	if ( out[ 2 ] == "c" ) cout << ".";
	else cout << "!\nFailed 3\n";
	if ( out[ 3 ] == "d" ) cout << ".";
	else cout << "!\nFailed 4\n";
	if ( out[ 4 ] == "e" ) cout << ".";
	else cout << "!\nFailed 5\n";
	if ( out[ 5 ] == "f" ) cout << ".";
	else cout << "!\nFailed 6\n";
	if ( out[ 6 ] == "g" ) cout << ".";
	else cout << "!\nFailed 7\n";
	if ( out[ 7 ] == "hohohohoho" ) cout << ".";
	else cout << "!\nFailed 8\n";

	cout << " done\n"; 
}
#endif

