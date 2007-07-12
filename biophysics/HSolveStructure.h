/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_STRUCTURE_H
#define _HSOLVE_STRUCTURE_H

struct HSolveStructure
{
	unsigned long            N_;
	vector< unsigned long >  checkpoint_;
	vector< unsigned char >  channelCount_;
	vector< unsigned char >  gateCount_;
	vector< unsigned char >  gateCount1_;
	vector< unsigned char >  gateFamily_;
	vector< double >         M_;
	vector< double >         V_;
	vector< double >         CmByDt_;
	vector< double >         EmByRm_;
	vector< double >         inject_;
	vector< double >         Gbar_;
	vector< double >         GbarEk_;
	vector< double >         state_;
	vector< double >         power_;
	vector< double >         lookup_;
	int                      lookupBlocSize_;
	int                      NDiv_;
	double                   VLo_;
	double                   VHi_;
	double                   dV_;        
};

#endif // _HSOLVE_STRUCTURE_H
