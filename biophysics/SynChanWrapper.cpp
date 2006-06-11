#include <math.h>
#include "header.h"
#include <queue>
#include "SynInfo.h"
#include "SynChan.h"
#include "SynChanWrapper.h"

#ifndef M_E
#define M_E   2.7182818284590452354
#endif


Finfo* SynChanWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"Ek", &SynChanWrapper::getEk, 
		&SynChanWrapper::setEk, "double" ),
	new ValueFinfo< double >(
		"Gk", &SynChanWrapper::getGk, 
		&SynChanWrapper::setGk, "double" ),
	new ValueFinfo< double >(
		"Ik", &SynChanWrapper::getIk, 
		&SynChanWrapper::setIk, "double" ),
	new ValueFinfo< double >(
		"Gbar", &SynChanWrapper::getGbar, 
		&SynChanWrapper::setGbar, "double" ),
	new ValueFinfo< double >(
		"tau1", &SynChanWrapper::getTau1, 
		&SynChanWrapper::setTau1, "double" ),
	new ValueFinfo< double >(
		"tau2", &SynChanWrapper::getTau2, 
		&SynChanWrapper::setTau2, "double" ),
	new ValueFinfo< int >(
		"normalizeWeights", &SynChanWrapper::getNormalizeWeights, 
		&SynChanWrapper::setNormalizeWeights, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc2Finfo< double, double >(
		"channelOut", &SynChanWrapper::getChannelSrc, 
		"channelIn", 1 ),
	new SingleSrc1Finfo< double >(
		"IkOut", &SynChanWrapper::getIkSrc, 
		"channelIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< double, ProcInfo >(
		"channelIn", &SynChanWrapper::channelFunc,
		&SynChanWrapper::getChannelConn, "channelOut, IkOut", 1 ),
	new Dest1Finfo< double >(
		"EkIn", &SynChanWrapper::EkFunc,
		&SynChanWrapper::getEkInConn, "" ),
	new Dest1Finfo< double >(
		"activationIn", &SynChanWrapper::activationFunc,
		&SynChanWrapper::getActivationInConn, "" ),
	new Dest1Finfo< double >(
		"modulatorIn", &SynChanWrapper::modulatorFunc,
		&SynChanWrapper::getModulatorInConn, "" ),
	new Dest1Finfo< double >(
		"reinitIn", &SynChanWrapper::reinitFunc,
		&SynChanWrapper::getChannelConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
	new ArrayFinfo< SynInfo >(
		"synapsesValue", &SynChanWrapper::getSynapsesValue,
		&SynChanWrapper::setSynapsesValue, "multi" ),
	new Synapse1Finfo< double >(
		"synapsesIn", &SynChanWrapper::synapsesFunc,
		&SynChanWrapper::getSynapsesConn, &SynChanWrapper::newSynapsesConn, "" ),

///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"channel", &SynChanWrapper::getChannelConn,
		"channelOut, channelIn, reinitIn" ),
};

const Cinfo SynChanWrapper::cinfo_(
	"SynChan",
	"Upinder S. Bhalla, 2006, NCBS",
	"SynChan: Synaptic channel incorporating weight and delay. Does not\nhandle activity-dependent modification, see HebbSynChan for \nthat. Very similiar to the old synchan from GENESIS.",
	"Neutral",
	SynChanWrapper::fieldArray_,
	sizeof(SynChanWrapper::fieldArray_)/sizeof(Finfo *),
	&SynChanWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SynChanWrapper::channelFuncLocal( double Vm, ProcInfo info )
{
			while ( !pendingEvents_.empty() &&
				pendingEvents_.top().delay <= info->currTime_ ) {
				activation_ += pendingEvents_.top().weight / info->dt_;
				pendingEvents_.pop();
			}
			X_ = modulation_ * activation_ * xconst1_ + X_ * xconst2_;
			Y_ = X_ * yconst1_ + Y_ * yconst2_;
			Gk_ = Y_ * norm_;
			Ik_ = ( Ek_ - Vm ) * Gk_;
			activation_ = 0.0;
			modulation_ = 1.0;
			channelSrc_.send( Gk_, Ek_ );
			IkSrc_.send( Ik_ );
}
void SynChanWrapper::reinitFuncLocal( double Vm )
{
			double dt = 1.0;
			if ( channelConn_.nTargets() > 0 ) {
				Element* compt = channelConn_.target( 0 )->parent();
				if ( compt ) {
					Field f = compt->field( "processIn" );
					if ( f.good() ) {
						Conn* tick = f->inConn( compt );
						if ( tick && tick->nTargets() > 0 ) {
							Ftype1< double >::get( 
								tick->target( 0 )->parent(), "dt", dt );
						}
					}
				}
			}
			activation_ = 0.0;
			modulation_ = 1.0;
			xconst1_ = tau1_ * ( 1.0 - exp( -dt / tau1_ ) );
			xconst2_ = exp( -dt / tau1_ );
			yconst1_ = tau2_ * ( 1.0 - exp( -dt / tau2_ ) );
			yconst2_ = exp( -dt / tau2_ );
			if ( tau1_ == tau2_ ) {
				norm_ = Gbar_ * M_E / tau1_;
			} else {
				double tpeak = tau1_ * tau2_ * log( tau1_ / tau2_ ) / 
					( tau1_ - tau2_ );
				norm_ = Gbar_ * ( tau1_ - tau2_ ) / 
					( tau1_ * tau2_ * ( 
						exp( -tpeak / tau1_ ) - exp( tpeak / tau2_ )
					) );
			}
			if ( normalizeWeights_ && synapsesConn_.size() > 0 )
				norm_ /= static_cast< double >( synapsesConn_.size() );
			while ( !pendingEvents_.empty() )
				pendingEvents_.pop();
}
///////////////////////////////////////////////////
// Synapse function definitions
///////////////////////////////////////////////////
void SynChanWrapper::setSynapsesValue(
	Element* e , unsigned long index, SynInfo value )
{
	SynChanWrapper* f = static_cast< SynChanWrapper* >( e );
	if ( f->synapsesConn_.size() > index )
		f->synapsesConn_[ index ]->value_ = value;
}

SynInfo SynChanWrapper::getSynapsesValue(
	const Element* e , unsigned long index )
{
	const SynChanWrapper* f = static_cast< const SynChanWrapper* >( e );
	if ( f->synapsesConn_.size() > index )
		return f->synapsesConn_[ index ]->value_;
	return SynInfo();
}

void SynChanWrapper::synapsesFunc( Conn* c, double time )
{
	SynConn< SynInfo >* s = static_cast< SynConn< SynInfo >* >( c );
	SynChanWrapper* temp = static_cast< SynChanWrapper* >( c->parent() );
	// Here we do the synaptic function
			temp->pendingEvents_.push( s->value_.event( time ) );
}

unsigned long SynChanWrapper::newSynapsesConn( Element* e ) {
	SynChanWrapper* temp = static_cast < SynChanWrapper* >( e );
	SynConn< SynInfo >* s = new SynConn< SynInfo >( e );
	temp->synapsesConn_.push_back( s );
	return temp->synapsesConn_.size( ) - 1;
 }
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* channelConnSynChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( SynChanWrapper, channelConn_ );
	return reinterpret_cast< SynChanWrapper* >( ( unsigned long )c - OFFSET );
}

Element* IkOutConnSynChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( SynChanWrapper, IkOutConn_ );
	return reinterpret_cast< SynChanWrapper* >( ( unsigned long )c - OFFSET );
}

Element* EkInConnSynChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( SynChanWrapper, EkInConn_ );
	return reinterpret_cast< SynChanWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// SynInfo conversion template specializations.
///////////////////////////////////////////////////

template<> string val2str< SynInfo >( SynInfo val ) {
	char ret[40];
	sprintf( ret, "%g,%g", val.weight, val.delay );
	return ret;
}

// Converts strings of the form <weight> or <weight, delay>
template<> SynInfo str2val< SynInfo >( const string& s ) {
	if ( s.length() < 1 )
		return SynInfo();

	size_t pos = s.find(",");
	if ( pos == string::npos ) {
		return SynInfo( atof( s.c_str() ), 0.0 );
	} else {
		double weight = atof( s.substr( 0, pos ).c_str() );
		double delay = atof( s.substr( pos + 1 ).c_str() );
		return SynInfo( weight, delay );
	}
}
