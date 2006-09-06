/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Ksolve_h
#define _Ksolve_h
class Ksolve
{
	friend class KsolveWrapper;
	public:
		Ksolve()
		{
			rebuildFlag_ = 0;
			bufOffset_ = 0;
			sumTotOffset_ = 0;
		}

	private:
		string path_;
		vector< double >S_;
		vector< double >Sinit_;
		bool rebuildFlag_;
		long bufOffset_;
		long sumTotOffset_;
		void setPath( const string& path, Element* wrapper );
		void zombify( Element* e, Field& solveSrc );
};
#endif // _Ksolve_h
