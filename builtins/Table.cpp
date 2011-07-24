/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <fstream>
#include "TableEntry.h"
#include "Table.h"

static SrcFinfo1< double > output (
	"output",
	"Sends out single pass of data in vector"
);

static SrcFinfo1< double > outputLoop (
	"outputLoop",
	"Sends data in vector in a loop, repeating as often as run continues"
);

static SrcFinfo1< FuncId > requestData(
	"requestData",
	"Sends request for a field to target object"
);

static DestFinfo recvDataBuf( "recvData",
	"Handles data sent back following request",
	new OpFunc1< Table, PrepackedBuffer >( &Table::recvData )
);

const Cinfo* Table::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Table, double > threshold(
			"threshold",
			"threshold used when Table acts as a buffer for spikes",
			&Table::setThreshold,
			&Table::getThreshold
		);

		static ValueFinfo< Table, vector< double > > vec(
			"vec",
			"vector with all table entries",
			&Table::setVec,
			&Table::getVec
		);

		static ReadOnlyValueFinfo< Table, double > outputValue(
			"outputValue",
			"Output value holding current table entry or output of a calculation",
			&Table::getOutputValue
		);

		static ReadOnlyValueFinfo< Table, unsigned int > outputIndex(
			"outputIndex",
			"Index of current table entry",
			&Table::getOutputIndex
		);

		static ReadOnlyValueFinfo< Table, unsigned int > size(
			"size",
			"size of table",
			&Table::getVecSize
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo input( "input",
			"Fills data into the Table.",
			new OpFunc1< Table, double >( &Table::input ) );

		static DestFinfo spike( "spike",
			"Fills spike timings into the Table. Signal has to exceed thresh",
			new OpFunc1< Table, double >( &Table::spike ) );

		static DestFinfo xplot( "xplot",
			"Dumps table contents to xplot-format file. "
			"Argument 1 is filename, argument 2 is plotname",
			new OpFunc2< Table, string, string >( &Table::xplot ) );

		static DestFinfo loadCSV( "loadCSV",
			"Reads a single column from a CSV file. "
			"Arguments: filename, column#, starting row#, separator",
			new OpFunc4< Table, string, int, int, char >( 
				&Table::loadCSV ) );

		static DestFinfo loadXplot( "loadXplot",
			"Reads a single plot from an xplot file. "
			"Arguments: filename, plotname"
			"When the file has 2 columns, the 2nd column is loaded.",
			new OpFunc2< Table, string, string >( 
				&Table::loadXplot ) );

		static DestFinfo compareXplot( "compareXplot",
			"Reads a plot from an xplot file and compares with contents of Table."
			"Result is put in 'output' field of table."
			"If the comparison fails (e.g., due to zero entries), the "
			"return value is -1."
			"Arguments: filename, plotname, comparison_operation"
			"Operations: rmsd (for RMSDifference), rmsr (RMSratio ), "
			"dotp (Dot product, not yet implemented).",
			new OpFunc3< Table, string, string, string >( 
				&Table::compareXplot ) );

		static DestFinfo compareVec( "compareVec",
			"Compares contents of Table with a vector of doubles."
			"Result is put in 'output' field of table."
			"If the comparison fails (e.g., due to zero entries), the "
			"return value is -1."
			"Arguments: Other vector, comparison_operation"
			"Operations: rmsd (for RMSDifference), rmsr (RMSratio ), "
			"dotp (Dot product, not yet implemented).",
			new OpFunc2< Table, vector< double >, string >(
				&Table::compareVec ) );

		static DestFinfo process( "process",
			"Handles process call, updates internal time stamp.",
			new ProcOpFunc< Table >( &Table::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call.",
			new ProcOpFunc< Table >( &Table::reinit ) );
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

		//////////////////////////////////////////////////////////////
		// Field Element for the vector data
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< Table, double > tableEntryFinfo( 
			"table", 
			"Field Element for Table entries",
			TableEntry::initCinfo(),
			&Table::lookupVec,
			&Table::setVecSize,
			&Table::getVecSize
		);

	static Finfo* tableFinfos[] = {
		&threshold,		// Value
		&vec,			// Value
		&outputValue,	// ReadOnlyValue
		&outputIndex,	// ReadOnlyValue
		&size,			// ReadOnlyValue
		&group,			// DestFinfo
		&input,			// DestFinfo
		&spike,			// DestFinfo
		&xplot,			// DestFinfo
		&loadCSV,			// DestFinfo
		&loadXplot,			// DestFinfo
		&compareXplot,		// DestFinfo
		&compareVec,		// DestFinfo
		&recvDataBuf,	// DestFinfo
		&output,		// SrcFinfo
		&outputLoop,		// SrcFinfo
		&requestData,		// SrcFinfo
		&tableEntryFinfo,	// FieldElementFinfo
		&proc,			// SharedFinfo
	};

	static Cinfo tableCinfo (
		"Table",
		Neutral::initCinfo(),
		tableFinfos,
		sizeof( tableFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Table >()
	);

	return &tableCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* tableCinfo = Table::initCinfo();

Table::Table()
	: threshold_( 0.0 ), lastTime_( 0.0 ), outputIndex_( 0 ), input_( 0.0 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Table::process( const Eref& e, ProcPtr p )
{
	lastTime_ = p->currTime;
	// send out a request for data. This magically comes back in the
	// RecvDataBuf and is handled.
	requestData.send( e, p, recvDataBuf.getFid() );
	if ( vec_.size() == 0 ) {
		output.send( e, p, 0.0 );
		outputLoop.send( e, p, 0.0 );
		output_ = 0;
		return;
	}

	if ( outputIndex_ < vec_.size() )
		output_ = vec_[ outputIndex_ ];
	else
		output_ = vec_.back();

	output.send( e, p, output_ );

	outputLoop.send( e, p, vec_[ outputIndex_ % vec_.size() ] );

	outputIndex_++;
}

void Table::reinit( const Eref& e, ProcPtr p )
{
	input_ = 0.0;
	vec_.resize( 0 );
	outputIndex_ = 0;
	lastTime_ = 0;
	// cout << "tabReinit on :" << p->groupId << ":" << p->threadIndexInGroup << endl << flush;
	requestData.send( e, p, recvDataBuf.getFid() );
}

void Table::input( double v )
{
	// input_ = v;
	vec_.push_back( v );
}

void Table::spike( double v )
{
	if ( v > threshold_ )
		vec_.push_back( lastTime_ );
}

void Table::xplot( string fname, string plotname )
{
	ofstream fout( fname.c_str(), ios_base::app | ios_base::out );
	fout << "/newplot\n";
	fout << "/plotname " << plotname << "\n";
	for ( vector< double >::iterator i = vec_.begin(); i != vec_.end(); ++i)
		fout << *i << endl;
	fout << "\n";
}

bool isNamedPlot( const string& line, const string& plotname )
{
	static const unsigned int len = strlen( "/plotname" ) ;
	if ( line.size() < len + 2 )
		return 0;
	if ( line[0] == '/' && line[1] == 'p' ) {
		string name = line.substr( strlen( "/plotname" ) );
		string::size_type pos = name.find_first_not_of( " 	" );
		if ( pos == string::npos ) {
			cout << "Table::loadXplot: Malformed plotname line '" <<
				line << "'\n";
			return 0;
		}
		name = name.substr( pos );
		if ( plotname == name )
			return 1;
	}
	return 0;
}

/*
bool lineHasExactlyTwoNumericalColumns( const string& line )
{
	istringstream sstream( line );
	double y;

	if ( sstream >> y ) {
		if ( sstream >> y ) {
			if ( sstream >> y ) {
				return 0;
			} else {
				return 1;
			}
		}
	}
	return 0;
}
*/

double getYcolumn( const string& line )
{
	istringstream sstream( line );
	double y1 = 0.0;
	double y2;
	double y3;

	if ( sstream >> y1 ) {
		if ( sstream >> y2 ) {
			if ( sstream >> y3 ) {
				return y1;
			} else {
				return y2;
			}
		}
	}
	return y1;
}

bool innerLoadXplot( string fname, string plotname, vector< double >& v )
{
	ifstream fin( fname.c_str() );
	if ( !fin.good() ) {
		cout << "Table::innerLoadXplot: Failed to open file " << fname <<endl;
		return 0;
	}

	string line;
	// Here we advance to the first numerical line.
	if ( plotname == "" ) // Just load starting from the 1st numerical line.
	{
		while ( fin.good() ) { // Break out of this loop if we find a number
			getline( fin, line );
			if ( isdigit( line[0] ) )
				break;;
			if ( line[0] == '-' && line.length() > 1 && isdigit( line[1] ) )
				break;
		}
	} else { // Find plotname and then begin loading.
		while ( fin.good() ) {
			getline( fin, line );
			if ( isNamedPlot ( line, plotname ) ) {
				if ( !getline ( fin, line ) )
					return 0;
				break;
			}
		}
	}

	v.resize( 0 );
	do {
		if ( line.length() == 0 || line == "/newplot" || line == "/plotname" )
			break;
		v.push_back( getYcolumn( line ) );
		getline( fin, line );
	} while ( fin.good() );

	return ( v.size() > 0 );
}

void Table::loadXplot( string fname, string plotname )
{
	if ( !innerLoadXplot( fname, plotname, vec_ ) ) {
		cout << "Table::loadXplot: unable to load data from file " << fname <<endl;
		return;
	}
}

void Table::loadCSV( 
	string fname, int startLine, int colNum, char separator )
{
}

double getRMSDiff( const vector< double >& v1, const vector< double >& v2 )
{
	unsigned int size = ( v1.size() < v2.size() ) ? v1.size() : v2.size();
	if ( size == 0 )
		return -1;

	double sumsq = 0;
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = v1[i] - v2[i];
		sumsq += temp * temp;
	}
	return sqrt( sumsq / size );
}

double getRMS( const vector< double >& v )
{
	double sumsq = 0;
	unsigned int size = v.size();
	if ( size == 0 )
		return -1;
	for ( vector< double >::const_iterator i = v.begin(); i != v.end(); ++i)
		sumsq += *i * *i;
	
	return sqrt( sumsq / size );
}

double getRMSRatio( const vector< double >& v1, const vector< double >& v2 )
{
	double r1 = getRMS( v1 );
	double r2 = getRMS( v2 );
	if ( v1.size() == 0 || v2.size() == 0 )
		return -1;
	if ( r1 + r2 > 1e-20 )
		return getRMSDiff( v1, v2 ) / (r1 + r2);
	return -1;
}

string headop( const string& op )
{
	const unsigned int len = 5;
	char temp[len];
	unsigned int i = 0;
	for ( i = 0; i < op.length() && i < len-1 ; ++i )
		temp[i] = tolower( op[i] );
	temp[i] = '\0';
	return string( temp );
}

void Table::compareXplot( string fname, string plotname, string op )
{
	vector< double > temp;
	if ( !innerLoadXplot( fname, plotname, temp ) ) {
		cout << "Table::compareXplot: unable to load data from file " << fname <<endl;
	}

	string hop = headop( op );

	if ( hop == "rmsd" ) { // RMSDifference
		output_ = getRMSDiff( vec_, temp );
	}

	if ( hop == "rmsr" ) { // RMS ratio
		output_ = getRMSRatio( vec_, temp );
	}

	if ( hop == "dotp" )
		cout << "Table::compareXplot: DotProduct not yet done\n";
}

void Table::compareVec( vector< double > temp, string op )
{
	// Note that this line below is illegal: it causes a race condition
	// vector< double > temp = Field< vector< double > >::get( other, "vec" );

	string hop = headop( op );

	if ( hop == "rmsd" ) { // RMSDifference
		output_ = getRMSDiff( vec_, temp );
	}

	if ( hop == "rmsr" ) { // RMS ratio
		output_ = getRMSRatio( vec_, temp );
	}

	if ( hop == "dotp" )
		cout << "Table::compareVec: DotProduct not yet done\n";
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Table::setThreshold( double v )
{
	threshold_ = v;
}

double Table::getThreshold() const
{
	return threshold_;
}

double Table::getOutputValue() const
{
	return output_;
}

unsigned int Table::getOutputIndex() const
{
	return outputIndex_;
}

//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

double* Table::lookupVec( unsigned int index )
{
	if ( index < vec_.size() )
		return &( vec_[index] );
	cout << "Error: Table::lookupTableEntry: Index " << index << 
		" >= vector size " << vec_.size() << endl;
	return 0;
}

void Table::setVecSize( unsigned int num )
{
	assert( num < 1000 ); // Pretty unlikely upper limit
	vec_.resize( num );
}

unsigned int Table::getVecSize() const
{
	return vec_.size();
}

vector< double > Table::getVec() const
{
	return vec_;
}

void Table::setVec( vector< double >  val )
{
	vec_ = val;
}

//////////////////////////////////////////////////////////////
// Test for 'get'
//////////////////////////////////////////////////////////////

void Table::recvData( PrepackedBuffer pb )
{
	assert ( pb.dataSize() == sizeof( double ) );
	double ret = *reinterpret_cast< const double* >( pb.data() );

	vec_.push_back( ret );
}


//////////////////////////////////////////////////////////////
