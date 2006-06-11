#include <math.h>
#include "header.h"
#include "Interpol.h"
#include "InterpolWrapper.h"
#include "Table.h"
#include "TableWrapper.h"


Finfo* TableWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ObjFinfo< Interpol >(
		"table", &TableWrapper::getTable, 
		&TableWrapper::setTable,
		&TableWrapper::lookupTable, "Interpol" ),
	new ValueFinfo< int >(
		"mode", &TableWrapper::getMode, 
		&TableWrapper::setMode, "int" ),
	new ValueFinfo< double >(
		"stepsize", &TableWrapper::getStepsize, 
		&TableWrapper::setStepsize, "double" ),
	new ValueFinfo< double >(
		"input", &TableWrapper::getInput, 
		&TableWrapper::setInput, "double" ),
	new ValueFinfo< double >(
		"output", &TableWrapper::getOutput, 
		&TableWrapper::setOutput, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc1Finfo< double >(
		"returnLookupOut", &TableWrapper::getReturnLookupSrc, 
		"returnLookupIn", 1 ),
	new NSrc1Finfo< double >(
		"out", &TableWrapper::getOutSrc, 
		"processIn, reinitIn, lookupIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"returnLookupIn", &TableWrapper::returnLookupFunc,
		&TableWrapper::getReturnLookupConn, "returnLookupOut", 1 ),
	new Dest1Finfo< double >(
		"lookupIn", &TableWrapper::lookupFunc,
		&TableWrapper::getLookupInConn, "out" ),
	new Dest1Finfo< double >(
		"in", &TableWrapper::inFunc,
		&TableWrapper::getInInConn, "" ),
	new Dest2Finfo< double, int >(
		"tabFillIn", &TableWrapper::tabFillFunc,
		&TableWrapper::getTabFillInConn, "" ),
	new Dest1Finfo< int >(
		"tabOpIn", &TableWrapper::tabOpFunc,
		&TableWrapper::getTabOpInConn, "" ),
	new Dest3Finfo< int, double, double >(
		"tabOpRangeIn", &TableWrapper::tabOpRangeFunc,
		&TableWrapper::getTabOpRangeInConn, "" ),
	new Dest1Finfo< double >(
		"sumIn", &TableWrapper::sumFunc,
		&TableWrapper::getSumInConn, "" ),
	new Dest1Finfo< double >(
		"prdIn", &TableWrapper::prdFunc,
		&TableWrapper::getPrdInConn, "" ),
	new Dest1Finfo< double >(
		"bufferIn", &TableWrapper::bufferFunc,
		&TableWrapper::getBufferInConn, "" ),
	new Dest2Finfo< double, int >(
		"assignIn", &TableWrapper::assignFunc,
		&TableWrapper::getAssignInConn, "" ),
	new Dest3Finfo< double, double, int >(
		"tabcreateIn", &TableWrapper::tabcreateFunc,
		&TableWrapper::getTabcreateInConn, "" ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &TableWrapper::processFunc,
		&TableWrapper::getProcessConn, "out", 1 ),
	new Dest0Finfo(
		"reinitIn", &TableWrapper::reinitFunc,
		&TableWrapper::getProcessConn, "out", 1 ),
	new Dest0Finfo(
		"dumpIn", &TableWrapper::dumpFunc,
		&TableWrapper::getDumpInConn, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"returnLookup", &TableWrapper::getReturnLookupConn,
		"returnLookupOut, returnLookupIn" ),
	new SharedFinfo(
		"process", &TableWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo TableWrapper::cinfo_(
	"Table",
	"Upinder S. Bhalla, 2006, NCBS",
	"Table: Table class. Handles lookup in several modes: \nmode 0: TAB_IO: Instantaneous lookup and return of output\nmode 1: TAB_LOOP: Looks up based on simulation time, looping.\nmode 2: TAB_ONCE: Looks up based on simulation time, nonlooping.\nmode 3: TAB_BUF: Buffers incoming data. Output holds index.\nmode 4: TAB_SPIKE: Buffers spike time. Spike thresh in stepsize\nmode 5: TAB_FIELDS: Buffers multiple input messages.\nmode 6: TAB_DELAY: Ring buffer. Delay = xdivs * dt.\nFurther refinement by stepsize:\nIn mode 1 and 2, stepsize is the increment to be applied\neach dt.\nIn mode 4, stepsize is the spike threshold.\nThe table can do internal interpolation using the function\ntabFill xdivs\nThe table can perform operations on its contents using the\ntabOp op\nwhere op can be:\na: average\nm: min\nM: max\nr: range = max - min\ns: slope\ni: intercept\nf: frequency\nS: Sqrt( sum of sqares )\nFor backward compatibility, we retain tabcreate:\ntabcreate xmin xmax xdivs",
	"Neutral",
	TableWrapper::fieldArray_,
	sizeof(TableWrapper::fieldArray_)/sizeof(Finfo *),
	&TableWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void TableWrapper::inFuncLocal( double value )
{
			if ( mode_ != 6 ) { 
				input_ = value;
			} else {
				x_ = value; 
			}
}
void TableWrapper::tabcreateFuncLocal( double xmin, double xmax, int xdivs )
{
			table_.localSetXdivs( xdivs );
			table_.localSetXmin( xmin );
			table_.localSetXmax( xmax );
}
void TableWrapper::processFuncLocal( ProcInfo info )
{
			switch ( mode_ ) {
				case 0: 
					output_ = table_.doLookup( input_ ) * py_ + sy_;
					py_ = 1.0;
					sy_ = 0.0;
					break;
				case 1: 
					if ( stepsize_ == 0.0 ) {
						double looplen = table_.localGetXmax() - table_.localGetXmin();
						double temp = input_ + info->currTime_;
						temp = temp - looplen * floor( temp / looplen );
						output_ = table_.doLookup( temp );
					} else {
						input_ += stepsize_;
						if ( input_ > table_.localGetXmax() )
							input_ = table_.localGetXmin();
						output_ = table_.doLookup( input_ );
					}
					break;
				case 2: 
					if ( stepsize_ == 0.0 ) {
						output_ = table_.doLookup( input_ + info->currTime_ );
					} else {
						input_ += stepsize_;
						output_ = table_.doLookup( input_ );
					}
					break;
				case 3: 
					{
						int i = static_cast< int >( output_ );
						if ( i < table_.localGetXdivs() && i >= 0 ) {
							table_.setTableValue( input_, i );
							output_ += 1.0;
							table_.localSetXmax( output_ );
						}
					}
					break;
				case 4: 
					if ( input_ > stepsize_ ) {
						int i = static_cast< int >( output_ );
						if ( i < table_.localGetXdivs() && i >= 0 ) {
							if ( x_ < stepsize_ ) {
								table_.setTableValue( info->currTime_, i );
								output_ = i + 1;
								table_.localSetXmax( output_ );
							}
						}
					}
					x_ = input_;
					break;
				case 5: 
					break;
				case 6: 
					int i = static_cast< int >( round( input_ ) );
					if ( i < table_.localGetXdivs() && i >= 0 ) {
						output_ = table_.getTableValue( i );
						table_.setTableValue( x_, i++ );
						input_ = ( i >= table_.localGetXdivs() ) ? 0 : i;
					}
					break;
			}
			outSrc_.send( output_ );
}


void TableWrapper::reinitFuncLocal(  ) {
	if ( mode_ <= 2 )
		output_ = table_.doLookup( input_ );
	else 
		output_ = 0.0;
	outSrc_.send( output_ );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* lookupInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, lookupInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* returnLookupConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, returnLookupConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* processConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, processConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* inInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, inInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* tabFillInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, tabFillInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* tabOpInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, tabOpInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* tabOpRangeInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, tabOpRangeInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* sumInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, sumInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* prdInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, prdInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* bufferInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, bufferInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* assignInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, assignInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* tabcreateInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, tabcreateInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* dumpInConnTableLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( TableWrapper, dumpInConn_ );
	return reinterpret_cast< TableWrapper* >( ( unsigned long )c - OFFSET );
}

Element* TableWrapper::lookupTable( Element* e, unsigned long index )
{
	return reinterpret_cast< InterpolWrapper* >(
		( unsigned long )
		( &static_cast< const TableWrapper* >( e )->table_ ) - 
		InterpolOffset()
	);
}

#ifdef DO_UNIT_TESTS
void ASSERT_TAB( bool test, const string& report )
{
	if ( test )
		cout << ".";
	else {
		cout << "!\nError in table: " << report << "\n";
		exit( 0 );
	}
}

void tabprocess( Field& proc, double time, double value, 
	const string& report )
{
	static ProcInfoBase info( "/sli_shell" );
	info.dt_ = 0.1;
	info.currTime_ = time;

	string res;

	Ftype1< ProcInfo >::set( proc.getElement(), proc.getFinfo(), &info);

	Field( "/tab/output" ).get( res );
	ASSERT_TAB( fabs( atof( res.c_str() ) - value ) < 1.0e-6, 
		res + " != " + report );
}

void testTable()
{
	cout << "Testing table: ";
	Element* telm = Cinfo::find( "Table" )->create( "tab", 
		Element::root() );
	TableWrapper* t = dynamic_cast< TableWrapper* >( telm );
//	TableWrapper* ttest = new TableWrapper( "/test" );
	t->tabcreateFuncLocal( 0.0, 100.0, 100 );
	t->localSetMode( 0 );
	ASSERT_TAB( t->table_.localGetXmin() == 0.0, "xmin assignment" );
	ASSERT_TAB( t->table_.localGetXmax() == 100.0, "xmax assignment" );
	ASSERT_TAB( t->table_.localGetXdivs() == 100, "xdivs assignment" );
	double x;
	int i = 0;
	for ( x = 0.0; x <= 100.0 ; x += 1.0 ) {
		t->assignFuncLocal( sin( x / 10.0 ), i++ );
	}
	ASSERT_TAB( 
		fabs( t->table_.getTableValue( 50 ) - sin( 5.0 ) ) < 1.0e-10,
		"assignFuncLocal" );
	Element* dbl = Cinfo::find( "Double" )->create( "dbl", Element::root() );
	Field fDbl( "/dbl/value" );
	ASSERT_TAB( 
		Field( "/tab/out").add( fDbl ), "Adding out msg\n" );
	Field( "/tab/lookupIn" ).set( "50.0" );
	string res;
	// fDbl.set( "1.234" );
	//fDbl.get( res );
	Field tabout( "/tab/output" );
	tabout.get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - sin( 5.0 ) ) < 1.0e-6,
		res + " != table lookup output" );
	fDbl.get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - sin( 5.0 ) ) < 1.0e-6,
		res + " != table out message" );

	Field( "/tab/table.table[27]" ).set( "9845.3");
	Field( "/tab/table.table[27]" ).get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - 9845.3 ) < 1.0e-6,
		res + " != table assignment through indexing" );

	Field( "/tab/assignIn" ).set( "1234.5, 17");
	Field( "/tab/table.table[17]" ).get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - 1234.5 ) < 1.0e-6,
		res + " != assign and table indexing" );

//////////////////////////////////////////////////////////////////

	Field tabproc( "/tab/processIn" );
	Field( "/tab/in" ).set( "27.0" );
	tabprocess( tabproc, 1.0, 9845.3, "Process mode 0, simple input" );

	Field( "/tab/in" ).set( "17.0" );
	Field( "/tab/sumIn" ).set( "10.0" );
	Field( "/tab/prdIn" ).set( "100.0" );
	tabprocess( tabproc, 1.0, 123460.0, "Process mode 0, sum and prd" );
//////////////////////////////////////////////////////////////////

	Field( "/tab/stepsize" ).set( "0.0" );
	Field( "/tab/mode" ).set( "1" );
	Field( "/tab/input" ).set( "0.0" );
	tabprocess( tabproc, 10.0, sin( 1.0 ),
		"Process mode 1=TAB_LOOP, zero stepsize, 1st cycle" );
	tabprocess( tabproc, 210.0, sin( 1.0 ),
		"Process mode 1=TAB_LOOP, zero stepsize, after 2 cycles" );

	Field( "/tab/input" ).set( "98.0" );
	Field( "/tab/stepsize" ).set( "1.0" );
	tabprocess( tabproc, 0.0, sin( 9.9 ),
		"Process mode 1=TAB_LOOP, stepsize=1, step 1 " );
	tabprocess( tabproc, 00.0, sin( 10.0 ),
		"Process mode 1=TAB_LOOP, stepsize=1, step 2" );
	tabprocess( tabproc, 00.0, sin( 0.0 ),
		"Process mode 1=TAB_LOOP, stepsize=1, step 3" );
	tabprocess( tabproc, 00.0, sin( 0.1 ),
		"Process mode 1=TAB_LOOP, stepsize=1, step 4" );
//////////////////////////////////////////////////////////////////

	Field( "/tab/input" ).set( "98.0" );
	Field( "/tab/mode" ).set( "2" );
	Field( "/tab/stepsize" ).set( "1.0" );
	tabprocess( tabproc, 0.0, sin( 9.9 ),
		"Process mode 2=TAB_ONCE, stepsize=1, step 1 " );
	tabprocess( tabproc, 00.0, sin( 10.0 ),
		"Process mode 2=TAB_ONCE, stepsize=1, step 2" );
	tabprocess( tabproc, 00.0, sin( 10.0 ),
		"Process mode 2=TAB_ONCE, stepsize=1, step 3" );
	tabprocess( tabproc, 00.0, sin( 10.0 ),
		"Process mode 2=TAB_ONCE, stepsize=1, step 4" );

//////////////////////////////////////////////////////////////////

	Field( "/tab/mode" ).set( "3" );
	Field( "/tab/output" ).set( "0.0" );
	Field( "/tab/input" ).set( "0.1" );
	tabprocess( tabproc, 0.0, 1.0,
		"Process mode 3=TAB_BUF, step 0 " );
	Field( "/tab/table.table[0]" ).get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - 0.1 ) < 1.0e-6,
		res + " != buffer input 0 " );

	Field( "/tab/input" ).set( "0.4" );
	tabprocess( tabproc, 0.0, 2.0,
		"Process mode 3=TAB_BUF, step 1 " );
	Field( "/tab/table.table[1]" ).get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - 0.4 ) < 1.0e-6,
		res + " != buffer input 1 " );

	Field( "/tab/input" ).set( "0.9" );
	tabprocess( tabproc, 0.0, 3.0,
		"Process mode 3=TAB_BUF, step 2 " );
	Field( "/tab/table.table[2]" ).get( res );
	ASSERT_TAB( 
		fabs( atof( res.c_str() ) - 0.9 ) < 1.0e-6,
		res + " != buffer input 2 " );

//////////////////////////////////////////////////////////////////

	Field( "/tab/mode" ).set( "4" );
	Field( "/tab/output" ).set( "0.0" );
	Field( "/tab/stepsize" ).set( "1.0" );
	Field( "/tab/input" ).set( "0.9" );
	tabprocess( tabproc, 0.0, 0.0, "Process mode 4=TAB_SPIKE, step 0");
	Field( "/tab/input" ).set( "0.8" );
	tabprocess( tabproc, 0.1, 0.0, "Process mode 4=TAB_SPIKE, step 1");
	Field( "/tab/input" ).set( "1.1" );
	tabprocess( tabproc, 0.2, 1.0, "Process mode 4=TAB_SPIKE, step 2");
	Field( "/tab/table.table[0]" ).get( res );
	ASSERT_TAB( fabs( atof( res.c_str() ) - 0.2 ) < 1.0e-6,
		res + " != spike over thresh at t=0.2 " );

	Field( "/tab/input" ).set( "-100.1" );
	tabprocess( tabproc, 0.3, 1.0, "Process mode 4=TAB_SPIKE, step 3");

	Field( "/tab/input" ).set( "100.1" );
	tabprocess( tabproc, 0.4, 2.0, "Process mode 4=TAB_SPIKE, step 4");
	Field( "/tab/table.table[1]" ).get( res );
	ASSERT_TAB( fabs( atof( res.c_str() ) - 0.4 ) < 1.0e-6,
		res + " != spike over thresh at t=0.4 " );

//////////////////////////////////////////////////////////////////

	Field( "/tab/mode" ).set( "6" ); // TAB_DELAY
	Field( "/tab/output" ).set( "0.0" );
	Field( "/tab/input" ).set( "0.0" );

	ProcInfoBase info( "/sli_shell" );
	Field tabin( "/tab/in" );
	// Pre-fill in vector
	for ( x = 0.0; x < 10.0; x += 0.1 ) {
		info.dt_ = 0.1;
		info.currTime_ = x;
		Ftype1< double >::set( tabin.getElement(), tabin.getFinfo(), 
			sin( x ) );
		Ftype1< ProcInfo >::set( tabproc.getElement(), tabproc.getFinfo(), &info );
	}
	// Read out after delay.
	for ( ; x < 30.0; x += 0.1 ) {
		double ret;
		info.dt_ = 0.1;
		info.currTime_ = x;
		Ftype1< double >::set( tabin.getElement(), tabin.getFinfo(), 
			sin( x ) );
		Ftype1< ProcInfo >::set( 
			tabproc.getElement(), tabproc.getFinfo(), &info );
		Ftype1< double >::get( 
			tabout.getElement(), tabout.getFinfo(), ret );
		if ( fabs( ret - sin( x - 10.0 ) ) > 1.0e-6 ) {
			cout << "!\nError in table: TAB_DELAY: ret (" << ret <<
				") != sin( x - 10.0 ): (" << sin( x - 10.0 ) << ")\n";
			exit( 0 );
		}
	}
	cout << ".";

//////////////////////////////////////////////////////////////////

	cout << " done\n";
	delete telm;
	delete dbl;
}
#endif
