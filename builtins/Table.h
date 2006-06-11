#ifndef _Table_h
#define _Table_h
/************************************************************************ This program is part of 'MOOSE', the** Messaging Object Oriented Simulation Environment,** also known as GENESIS 3 base code.**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS** It is made available under the terms of the** GNU Lesser General Public License version 2.1** See the file COPYING.LIB for the full notice.**********************************************************************/
class Table
{
	friend class TableWrapper;
#ifdef DO_UNIT_TESTS
	friend void testTable();
#endif
	public:
		Table()
			: mode_( 0 ), stepsize_( 0.0 ), input_( 0.0 ),
				output_( 0.0 ), sy_( 0.0 ), py_( 1.0 ), x_ ( 0.0 )
		{
			;
		}

	private:
		Interpol table_;
		int mode_;
		double stepsize_;
		double input_;
		double output_;
		double sy_;
		double py_;
		double x_;
};
#endif // _Table_h
