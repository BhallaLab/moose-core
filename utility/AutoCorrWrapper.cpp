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
#include "AutoCorr.h"
#include "AutoCorrWrapper.h"


Finfo* AutoCorrWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"threshold", &AutoCorrWrapper::getThreshold, 
		&AutoCorrWrapper::setThreshold, "double" ),
	new ValueFinfo< int >(
		"binCount", &AutoCorrWrapper::getBinCount, 
		&AutoCorrWrapper::setBinCount, "int" ),
	new ValueFinfo< double >(
		"binWidth", &AutoCorrWrapper::getBinWidth, 
		&AutoCorrWrapper::setBinWidth, "double" ),
	new ValueFinfo< int >(
		"spikeCount", &AutoCorrWrapper::getSpikeCount, 
		&AutoCorrWrapper::setSpikeCount, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< double, double >(
		"spikeIn", &AutoCorrWrapper::spikeFunc,
		&AutoCorrWrapper::getSpikeInConn, "" ),
	new Dest2Finfo< string, int >(
		"printIn", &AutoCorrWrapper::printFunc,
		&AutoCorrWrapper::getPrintInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &AutoCorrWrapper::reinitFunc,
		&AutoCorrWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &AutoCorrWrapper::processFunc,
		&AutoCorrWrapper::getProcessConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &AutoCorrWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo AutoCorrWrapper::cinfo_(
	"AutoCorr",
	"",
	"AutoCorr: Auto-correlation histogram.\nAccepts spike-amplitudes from a spike train and computes its\nauto-correlogram. Spikes are recorded only if their amplitudes\ncross the given threshold.\nImportant fields:\nbinCount:  Number of bins, preferably odd.\nbinWidth:  Width of one bin in units of time.\nIf binCount = 3, binWidth = 1.0, the bins appear as:\n|   Bin 0   |   Bin 1   |   Bin 2   |\n|-----+-----|-----+-----|-----+-----|\n-1.5  -1.0  -0.5  -0.0   0.5   1.0   1.5",
	"Neutral",
	AutoCorrWrapper::fieldArray_,
	sizeof(AutoCorrWrapper::fieldArray_)/sizeof(Finfo *),
	&AutoCorrWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void AutoCorrWrapper::spikeFuncLocal( double amplitude, double cTime )
{
			if( amplitude < threshold_ )
				return;
			deque<double>::iterator i = spikeTime_.begin();
			while ( i != spikeTime_.end() && cTime - *i >= ccWidth_ )
				++i;
			spikeTime_.erase( spikeTime_.begin(), i );
			for( i = spikeTime_.begin(); i != spikeTime_.end(); ++i )
				++correlogram_ [
					static_cast<int>
					(( cTime - *i + ccWidth_ ) / binWidth_)
				];
			++spikeCount_;
			spikeTime_.push_back( cTime );
}
void AutoCorrWrapper::printFuncLocal( string fileName, int plotMode )
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
void AutoCorrWrapper::reinitFuncLocal(  )
{
			correlogram_.resize( binCount_, 0 );
			spikeTime_.clear();
			spikeCount_ = 0;
			ccWidth_ = binCount_ * binWidth_ / 2.0;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnAutoCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( AutoCorrWrapper, processConn_ );
	return reinterpret_cast< AutoCorrWrapper* >( ( unsigned long )c - OFFSET );
}

Element* spikeInConnAutoCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( AutoCorrWrapper, spikeInConn_ );
	return reinterpret_cast< AutoCorrWrapper* >( ( unsigned long )c - OFFSET );
}

Element* printInConnAutoCorrLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( AutoCorrWrapper, printInConn_ );
	return reinterpret_cast< AutoCorrWrapper* >( ( unsigned long )c - OFFSET );
}

