/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _KineticHub_h
#define _KineticHub_h
class KineticHub
{
	friend class KineticHubWrapper;
	public:
		KineticHub()
		{
		}

	private:
		vector< double >* S_;
		vector< double >* Sinit_;
		bool rebuildFlag_;
		unsigned long nMol_;
		unsigned long nBuf_;
		unsigned long nSumTot_;
};
#endif // _KineticHub_h
