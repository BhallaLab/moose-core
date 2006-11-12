/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "header.h"
#include <deque>
#include <fstream>
#include "CrossCorr.h"
#include "CrossCorrWrapper.h"


Finfo* CrossCorrWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"threshold", &CrossCorrWrapper::getThreshold, 
		&CrossCorrWrapper::setThreshold, "double" ),
	new ValueFinfo< int >(
		"binCount", &CrossCorrWrapper::getBinCount, 
		&CrossCorrWrapper::setBinCount, "int" ),
	new ValueFinfo< double >(
		"binWidth", &CrossCorrWrapper::getBinWidth, 
		&CrossCorrWrapper::setBinWidth, "double" ),
	new ValueFinfo< int >(
		"aSpikeCount", &CrossCorrWrapper::getASpikeCount, 
		&CrossCorrWrapper::setASpikeCount, "int" ),
	new ValueFinfo< int >(
		"bSpikeCount", &CrossCorrWrapper::getBSpikeCount, 
		&CrossCorrWrapper::setBSpikeCount, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< double, double >(
		"aSpikeIn", &CrossCorrWrapper::aSpikeFunc,
		&CrossCorrWrapper::getASpikeInConn, "" ),
	new Dest2Finfo< double, double >(
		"bSpikeIn", &CrossCorrWrapper::bSpikeFunc,
		&CrossCorrWrapper::getBSpikeInConn, "" ),
	new Dest2Finfo< string, int >(
		"printIn", &CrossCorrWrapper::printFunc,
		&CrossCorrWrapper::getPrintInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &CrossCorrWrapper::reinitFunc,
		&CrossCorrWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &CrossCorrWrapper::processFunc,
		&CrossCorrWrapper::getProcessConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &CrossCorrWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo CrossCorrWrapper::cinfo_(
	"CrossCorr",
	"",
	"CrossCorr: Cross-correlation histogram.\nAccepts spike-amplitudes from spike trains A and B, and\ncomputes their cross-correlogram. Spikes are recorded only\nif their amplitudes cross the given threshold.\nImportant fields:\nbinCount:  Number of bins, preferably odd.\nbinWidth:  Width of one bin in units of time.\nIf binCount = 3, binWidth = 1.0, the bins appear as:\n|   Bin 0   |   Bin 1   |   Bin 2   |\n|-----+-----|-----+-----|-----+-----|\n-1.5  -1.0  -0.5  -0.0   0.5   1.0   1.5",
	"Neutral",
	CrossCorrWrapper::fieldArray_,
	sizeof(CrossCorrWrapper::fieldArray_)/sizeof(Finfo *),
	&CrossCorrWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void CrossCorrWrapper::aSpikeFuncLocal( double amplitude, double cTime )
{
			if( amplitude < threshold_ )
				return;
			++aSpikeCount_;
			aSpikeTime_.push_back( cTime );
			deque<double>::iterator ib;
			for( ib = bSpikeTime_.begin(); ib != bSpikeTime_.end(); ++ib )
				if ( cTime - *ib > ccWidth_ )
					bSpikeTime_.pop_front();
				else
					++correlogram_ [
						static_cast<int>
						(( *ib - cTime + ccWidth_ ) / binWidth_)
					];
}
void CrossCorrWrapper::bSpikeFuncLocal( double amplitude, double cTime )
{
			if( amplitude < threshold_ )
				return;
			++bSpikeCount_;
			bSpikeTime_.push_back( cTime );
			if ( aSpikeTime_.empty() )
				return;
			deque<double>::iterator ia = aSpikeTime_.begin();
			while ( ia != aSpikeTime_.end() && cTime - *ia >= ccWidth_ )
				++ia;
			aSpikeTime_.erase( aSpikeTime_.begin(), ia );
			for( ia = aSpikeTime_.begin(); ia != aSpikeTime_.end(); ++ia )
				++correlogram_ [
					static_cast<int>
					(( cTime - *ia + ccWidth_ ) / binWidth_)
				];
}
void CrossCorrWrapper::printFuncLocal( string fileName, int plotMode )
{
			ofstream fout( fileName.c_str(), plotMode==1 ? ios::out : ios::app );
			fout << "/Cross-correlogram\n/Correlogram name " << name() << endl;
			long ic = - ( correlogram_.size() / 2 );
			for( long ii = 0;
			     ii < static_cast<signed long> (correlogram_.size());
			     ++ii, ++ic )
				fout << ic * binWidth_ << '\t' << correlogram_[ii] << endl;
			fout << endl;
			fout.close();
}
void CrossCorrWrapper::reinitFuncLocal(  )
{
			correlogram_.resize( binCount_, 0 );
			aSpikeTime_.clear();
			bSpikeTime_.clear();
			aSpikeCount_ = 0;
			bSpikeCount_ = 0;
			ccWidth_ = binCount_ * binWidth_ / 2.0;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnCrossCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CrossCorrWrapper, processConn_ );
	return reinterpret_cast< CrossCorrWrapper* >( ( unsigned long )c - OFFSET );
}

Element* aSpikeInConnCrossCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CrossCorrWrapper, aSpikeInConn_ );
	return reinterpret_cast< CrossCorrWrapper* >( ( unsigned long )c - OFFSET );
}

Element* bSpikeInConnCrossCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CrossCorrWrapper, bSpikeInConn_ );
	return reinterpret_cast< CrossCorrWrapper* >( ( unsigned long )c - OFFSET );
}

Element* printInConnCrossCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CrossCorrWrapper, printInConn_ );
	return reinterpret_cast< CrossCorrWrapper* >( ( unsigned long )c - OFFSET );
}

