/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_PASSIVE_H
#define _HSOLVE_PASSIVE_H

class HSolvePassive: public HinesMatrix
{
#ifdef DO_UNIT_TESTS
	friend void testHSolvePassive();
#endif
	
public:
	void setup( Id seed, double dt );
	void solve();
	
protected:
	// Integration
	void updateMatrix();
	void forwardEliminate();
	void backwardSubstitute();
	
	vector< CompartmentStruct >       compartment_;
	vector< Id >                      compartmentId_;
	vector< double >                  V_;
	vector< TreeNodeStruct >          tree_;
	map< unsigned int, InjectStruct > inject_;
	
private:
	// Setting up of data structures
	void clear();
	void walkTree( Id seed );
	void initialize();
	void storeTree();
	
	// Used for unit tests.
	double getV( unsigned int row ) const;
};

#endif // _HSOLVE_PASSIVE_H
