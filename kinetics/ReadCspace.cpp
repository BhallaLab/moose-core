/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include "header.h"
#include "../shell/Shell.h"

#define DO_CSPACE_DEBUG 0

#include "ReadCspace.h"

const double ReadCspace::SCALE = 1.0;
const double ReadCspace::DEFAULT_CONC = 1.0;
const double ReadCspace::DEFAULT_RATE = 0.1;
const double ReadCspace::DEFAULT_KM = 1.0;
const bool ReadCspace::USE_PIPE = 1;

ReadCspace::ReadCspace()
	:
		base_(),
		fout_( &cout )
{;}

void ReadCspace::printHeader()
{
	reaclist_.resize( 0 );
	mollist_.resize( 0 );
}

void ReadCspace::printFooter()
{
	string separator = ( USE_PIPE ) ? "|" : "" ;
	// We do this in all cases, regardless of the doOrdering flag.
	sort( mollist_.begin(), mollist_.end() );
	sort( reaclist_.begin(), reaclist_.end() );
	unsigned int i;
	*fout_ << separator;
	for ( i = 0; i < reaclist_.size(); i++ )
		*fout_ << reaclist_[ i ].name() << separator;

	for ( i = 0; i < mollist_.size(); i++ )
		*fout_ << " " << mollist_[i].conc();
	for ( i = 0; i < reaclist_.size(); i++ )
		*fout_ << " " << reaclist_[i].r1() << " " << reaclist_[i].r2();
	*fout_ << "\n";
}

void ReadCspace::printMol( Id id, double conc, double concinit, double vol)
{

	// Skip explicit enzyme complexes.
	ObjId parent = Neutral::parent( id.eref() );
	if ( parent.element()->cinfo()->isA( "Enzyme" ) &&
		id.element()->getName() == ( parent.element()->getName() + "_cplx" )
	)
		return;

	CspaceMolInfo m( id.element()->getName()[ 0 ], concinit );
	mollist_.push_back( m );
	// Need to look up concs in a final step so that the sorted order
	// is maintained.
}

void ReadCspace::printReac( Id id, double kf, double kb)
{
	CspaceReacInfo r( id.element()->getName(), kf, kb );
	reaclist_.push_back( r );
}

void ReadCspace::printEnz( Id id, Id cplx, double k1, double k2, double k3)
{
	CspaceReacInfo r( id.element()->getName(), k3, (k3 + k2) / k1 );
	reaclist_.push_back( r );
}

// Model string always includes topology, after that the parameters
// are filled in according to how many there are.
void ReadCspace::readModelString( const string& model,
	const string& modelname, Id pa, const string& solverClass )
{
	unsigned long pos = model.find_first_of( "|" );
	if ( pos == string::npos ) {
		cerr << "ReadCspace::readModelString: Error: model undefined in\n";
		cerr << model << "\n";
		return;
	}
	mol_.resize( 0 );
	molseq_.resize( 0 );
	reac_.resize( 0 );
	molparms_.resize( 0 );
	parms_.resize( 0 );
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1,1 );

	base_ = s->doCreate( solverClass, pa, modelname, dims, false );
	assert( base_ != Id() );

	string temp = model.substr( pos + 1 );
	pos = temp.find_first_of( " 	\n" );
	
	for (unsigned long i = 0 ; i < temp.length() && i < pos; i += 5 ) {
		build( temp.c_str() + i );
		if ( temp[ i + 4 ] != '|' )
			break;
	}

	parms_.insert( parms_.begin(), molparms_.begin(), molparms_.end() );

	pos = model.find_last_of( "|" ) + 1;
	double val = 0;
	int i = 0;
	while ( pos < model.length() ) {
		if ( model[ pos ] == ' ' ) {
			val = atof( model.c_str() + pos + 1 );
			parms_[ i++ ] = val;
		}
		pos++;
	}

	deployParameters();
}
/////////////////////////////////////////////////////////////////////
//	From reacdef.cpp in CSPACE:
//             if (line == "A <==> B") type = 'A';
//        else if (line == "2A <==> B") type = 'B';
//        else if (line == "A --A--> B") type = 'C';
//        else if (line == "A --B--> B") type = 'D';
//        else if (line == "A <==> B + C") type = 'E';
//        else if (line == "2A <==> B + C") type = 'F';
//        else if (line == "2A + B <==> C") type = 'G';
//        else if (line == "2A + B <==> 2C") type = 'H';
//        else if (line == "4A + B <==> C") type = 'I';
//        else if (line == "A --B--> C") type = 'J';
//        else if (line == "A --A--> B + C") type = 'K';
//        else if (line == "A --B--> B + C") type = 'L';
/////////////////////////////////////////////////////////////////////

void ReadCspace::expandEnzyme( 
	const char* name, int e, int s, int p, int p2 )
{
	static Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	static vector< unsigned int > dims( 1,1 );

	Id enzMolId = mol_[e];
	
	Id enzId = shell->doCreate( "Enz", enzMolId, name, dims, false );

	MsgId ret = shell->doAddMsg( "OneToOne", 
		enzMolId, "reac", enzId, "enz" );

	ret = shell->doAddMsg( "OneToOne", 
		mol_[ name[ s ] - 'a' ], "reac", enzId, "sub" );

	ret = shell->doAddMsg( "OneToOne", 
		mol_[ name[ p ] - 'a' ], "reac", enzId, "prd" );

	if ( p2 != 0 )
		ret = shell->doAddMsg( "OneToOne", 
			mol_[ name[ p2 ] - 'a' ], "reac", enzId, "prd" );

	reac_.push_back( enzId );
	parms_.push_back( DEFAULT_RATE );
	parms_.push_back( DEFAULT_KM );
}

void ReadCspace::expandReaction( const char* name, int nm1 )
{
	static Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	static vector< unsigned int > dims( 1,1 );

	if ( name[0] == 'C' || name[0] == 'D' || name[0] >= 'J' ) // enzymes
		return;
	int i;

	Id reacId = s->doCreate( "Reac", base_, name, dims, false );
	
	// A is always a substrate
	for (i = 0; i < nm1; i++ ) {
		s->doAddMsg( "OneToOne", reacId, "sub", mol_[ name[1] - 'a' ], "reac" );
	}
		
	if ( name[0] < 'G' ) { // B is a product
		s->doAddMsg( "OneToOne", reacId, "prd", mol_[ name[2] - 'a' ], "reac" );
	} else { // B is a substrate
		s->doAddMsg( "OneToOne", reacId, "sub", mol_[ name[2] - 'a' ], "reac" );
	}

	if ( name[0] > 'D' ) { // C is a product
		s->doAddMsg( "OneToOne", reacId, "prdout", mol_[ name[3] - 'a' ], "reac" );
	}

	if ( name[0] == 'H' ) { // C is a dual product
		s->doAddMsg( "OneToOne", reacId, "prd", mol_[ name[3] - 'a' ], "reac" );
	}
	reac_.push_back( reacId );
	parms_.push_back( DEFAULT_RATE );
	parms_.push_back( DEFAULT_RATE );
}

void ReadCspace::build( const char* name )
{
	makeMolecule( name[1] );
	makeMolecule( name[2] );
	makeMolecule( name[3] );
	char tname[6];
	strncpy( tname, name, 4 );
	tname[4] = '\0';

	switch ( tname[0] ) {
		case 'A':
		case 'E':
			expandReaction( tname, 1 );
			break;
		case 'B':
		case 'F':
		case 'G':
		case 'H':
			expandReaction( tname, 2 );
			break;
		case 'I':
			expandReaction( tname, 4 );
			break;
		case 'C':
			expandEnzyme( tname, 1, 1, 2 );
			break;
		case 'D':
			expandEnzyme( tname, 2, 1, 2 );
			break;
		case 'J':
			expandEnzyme( tname, 2, 1, 3 );
			break;
		case 'K':
			expandEnzyme( tname, 1, 1, 2, 3 );
			break;
		case 'L':
			expandEnzyme( tname, 2, 1, 2, 3 );
			break;
		default:
			break;
	}
}

void ReadCspace::makeMolecule( char name )
{
	static Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	static vector< unsigned int > dims( 1,1 );

	if ( name == 'X' ) // silently ignore it, as it is a legal state
		return;
	if ( name < 'a' || name > 'z' ) {
		cerr << "ReadCspace::makeMolecule Error: name '" << name <<
			"' out of range 'a' to 'z'\n";
		return;
	}

	unsigned int index = 1 + name - 'a';

	// Put in molecule if it is a new one.
	if ( find( molseq_.begin(), molseq_.end(), index - 1 ) == 
					molseq_.end() )
			molseq_.push_back( index - 1 );

	for ( unsigned int i = mol_.size(); i < index; i++ ) {
		stringstream ss( "a" );
		ss << i ;
		string molname = ss.str();
		Id temp = s->doCreate( "Pool", base_, molname, dims, false );
		mol_.push_back( temp );
		molparms_.push_back( DEFAULT_CONC );
	}
}

void ReadCspace::deployParameters( )
{
	unsigned long i, j;
	if ( parms_.size() != mol_.size() + 2 * reac_.size() ) {
		cerr << "ReadCspace::deployParameters: Error: # of parms mismatch\n";
		return;
	}
	for ( i = 0; i < mol_.size(); i++ ) {
		// SetField(mol_[ i ], "volscale", volscale );
		// SetField(mol_[ molseq_[i] ], "ninit", parms_[ i ] );
		Field< double >::set( mol_[i], "nInit", parms_[i] );
	}
	for ( j = 0; j < reac_.size(); j++ ) {
		if ( reac_[ j ].element()->cinfo()->isA( "Reac" ) ) {
			Field< double >::set( reac_[j], "kf", parms_[i++] );
			Field< double >::set( reac_[j], "kb", parms_[i++] );
		} else {
			Field< double >::set( reac_[j], "k3", parms_[i] );
			Field< double >::set( reac_[j], "k2", 4.0 * parms_[i++] );
			Field< double >::set( reac_[j], "Km", parms_[i++] );
		}
	}
}

void ReadCspace::testReadModel( )
{
	cout << "Testing ReadCspace::readModelString()\n";
	readModelString( "|Habc|Lbca|", "mod1", Id(), "Stoich" );
	assert( mol_.size() == 3 );
	assert( reac_.size() == 2 );
	readModelString( "|AabX|BbcX|CcdX|DdeX|Eefg|Ffgh|Gghi|Hhij|Iijk|Jjkl|Kklm|Llmn| 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 101 102 201 202 301 302 401 402 501 502 601 602 701 702 801 802 901 902 1001 1002 1101 1102 1201 1202",
		"kinetics", Id(), "Stoich" );
	assert( mol_.size() == 14 );
	assert( reac_.size() == 12 );
	double concInit;
	int i;
	cout << "\nTesting ReadCspace:: conc assignment\n";
	for ( i = 0; i < 14; i++ ) {
		stringstream ss( "/kinetics/" );
		// const char* molname = ss.str();
		ss << 'a' + i;
		Id temp( ss.str() );
		concInit = Field< double >::get( temp, "concInit" );
		assert( doubleEq( concInit, i + 1 ) );
	}

	cout << "\nTesting ReadCspace:: reac construction\n";
	assert( reac_[ 0 ].path() == "/kinetics/AabX" );
	assert( reac_[ 1 ].path() == "/kinetics/BbcX" );
	assert( reac_[ 2 ].path() == "/kinetics/c/CcdX" );
	assert( reac_[ 3 ].path() == "/kinetics/e/DdeX" );
	assert( reac_[ 4 ].path() == "/kinetics/Eefg" );
	assert( reac_[ 5 ].path() == "/kinetics/Ffgh" );
	assert( reac_[ 6 ].path() == "/kinetics/Gghi" );
	assert( reac_[ 7 ].path() == "/kinetics/Hhij" );
	assert( reac_[ 8 ].path() == "/kinetics/Iijk" );
	assert( reac_[ 9 ].path() == "/kinetics/k/Jjkl" );
	assert( reac_[ 10 ].path() == "/kinetics/k/Kklm" );
	assert( reac_[ 11 ].path() == "/kinetics/m/Llmn" );

	cout << "\nTesting ReadCspace:: reac rate assignment\n";
	Id tempA( "/kinetics/AabX" );
	double r1 = Field< double >::get( tempA, "kf");
	double r2 = Field< double >::get( tempA, "kb");
	assert( doubleEq( r1, 101 ) && doubleEq( r2, 102 ) );

	Id tempB( "/kinetics/BbcX" );
	r1 = Field< double >::get( tempB, "kf");
	r2 = Field< double >::get( tempB, "kb");
	assert( doubleEq( r1, 201 ) && doubleEq( r2, 202 ) );

	Id tempC( "/kinetics/c/CcdX" );
	r1 = Field< double >::get( tempC, "k3");
	r2 = Field< double >::get( tempC, "km");
	assert( doubleEq( r1, 301 ) && doubleEq( r2, 302 ) );

	Id tempD( "/kinetics/e/DdeX" );
	r1 = Field< double >::get( tempD, "k3");
	r2 = Field< double >::get( tempD, "km");
	assert( doubleEq( r1, 401 ) && doubleEq( r2, 402 ) );

	for ( i = 4; i < 9; i++ ) {
		stringstream ss( "/kinetics/A" );
		ss << i << 'a' + i << 'b' + i << 'c' + i;
		// sprintf( ename, "/kinetics/%c%c%c%c", 'A' + i, 'a' + i, 'b' + i, 'c' + i );
		Id temp( ss.str() );
		r1 = Field< double >::get( temp, "kf");
		r2 = Field< double >::get( temp, "kb");
		assert( doubleEq( r1, i* 100 + 101 ) && 
			doubleEq( r2, i * 100 + 102 ) );
	}

	Id tempJ( "/kinetics/k/Jjkl" );
	r1 = Field< double >::get( tempJ, "k3");
	r2 = Field< double >::get( tempJ, "km");
	assert( doubleEq( r1, 1001 ) && doubleEq( r2, 1002 ) );

	Id tempK( "/kinetics/k/Kklm" );
	r1 = Field< double >::get( tempK, "k3");
	r2 = Field< double >::get( tempK, "km");
	assert( doubleEq( r1, 1101 ) && doubleEq( r2, 1102 ) );

	Id tempL( "/kinetics/m/Llmn" );
	r1 = Field< double >::get( tempL, "k3");
	r2 = Field< double >::get( tempL, "km");
	assert( doubleEq( r1, 1201 ) && doubleEq( r2, 1202 ) );

	cout << ".";
}
