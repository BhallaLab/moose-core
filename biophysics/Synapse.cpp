#include <queue>
#include "header.h"
#include "Synapse.h"
#include "IntFire.h"
#include "Dinfo.h"
#include "UpFunc.h"

const Cinfo* Synapse::initCinfo()
{
	static Finfo* synapseFinfos[] = {
		new ValueFinfo< Synapse, double >(
			"weight",
			"Synaptic weight",
			&Synapse::setWeight,
			&Synapse::getWeight
		),

		new ValueFinfo< Synapse, double >(
			"delay",
			"Axonal propagation delay to this synapse",
			&Synapse::setDelay,
			&Synapse::getDelay
		),

		new DestFinfo( "addSpike",
			"Handles arriving spike messages, by redirecting up to parent "
			"Int Fire object",
			new UpFunc1< IntFire, double >( &IntFire::addSpike ) ),
	};

	static Cinfo synapseCinfo (
		"Synapse",
		0, // No base class, but eventually I guess it will be neutral.
		synapseFinfos,
		sizeof( synapseFinfos ) / sizeof ( Finfo* ),
		new FieldDinfo()
	);

	return &synapseCinfo;
}

static const Cinfo* synapseCinfo = Synapse::initCinfo();

Synapse::Synapse()
	: weight_( 1.0 ), delay_( 0.0 )
{
	;
}

Synapse::Synapse( double w, double d ) 
	: weight_( w ), delay_( d )
{
	;
}

Synapse::Synapse( const Synapse& other, double time )
	: weight_( other.weight_ ), delay_( time + other.delay_ )
{
	;
}

/*
void Synapse::clearQ( Eref e )
{
	const char* i = e.generalQ.begin();
	while i != e.generalQ.end() {
		// FuncId* fi = static_cast< FuncId* >( i );
		// i += sizeof( FuncId );
		// i += fi->doOperation( e, i );
		// i += doOperation( *fi, e, i );
		unsigned int op = *static_cast< const unsigned int* >( i );
		i += sizeof( unsigned int );
		i += this->opVec_[ op ]( e, i );
			// opVec is set up statically, has the function ptrs.
			// All are of the form f( Eref e, const char* i ).
	}
}
*/

/*
unsigned int FuncId::doOperation( Eref e, char* i )
{
	unsigned int op = *static_cast< unsigned int* >( i );
	i += sizeof( unsigned int );
	return opVec_[ op ]( i ) + sizeof( unsigned int );
}
*/


void Synapse::setWeight( const double v )
{
	weight_ = v;
}

void Synapse::setDelay( const double v )
{
	delay_ = v;
}

double Synapse::getWeight() const
{
	return weight_;
}

double Synapse::getDelay() const
{
	return delay_;
}

///////////////////////////////////////////////////////////////////////

SynElement::SynElement( const Cinfo* c, const Element* other )
	: 	Element( c, other )
{ ; }

void SynElement::process( const ProcInfo* p )
{;}

char* SynElement::data( DataId index )
{
	assert( index.data() < numData_ );
	Synapse* s = 
		( reinterpret_cast< IntFire* >( d_ + index.data() * dataSize_ ) )->
		synapse( index.field() );
	return reinterpret_cast< char* >( s );
}

char* SynElement::data1( DataId index )
{
	assert( index.data() < numData_ );
	return d_ + index.data() * dataSize_;
}

unsigned int SynElement::numData() const
{
	unsigned int ret = 0;;
	char* endData = d_ + numData_ * dataSize_;

	for ( char* data = d_; data < endData; data += dataSize_ )
		ret += ( reinterpret_cast< IntFire* >( data ) )->getNumSynapses();

	return ret;
}

unsigned int SynElement::numDimensions() const
{
	return 2;
}
