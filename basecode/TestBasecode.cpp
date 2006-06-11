/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include <fstream>
#include "header.h"
#include "Shell.h"
#include "ShellWrapper.h"

extern void testWildcard();
extern void	testParseArgs(); // Defined in Shell.cpp


// Here is a simple test for Element access from the shell
void testFindElement( Shell* shell, 
	const string& path, const Element* ans ) 
{
	static unsigned int counter_ = 0;
	++counter_;
	if ( ans == shell->findElement( path ) ) {
		cout << ".";
	} else {
		cout << "!\n";
		cout << "Assert " << counter_ << ": shell->findElement " <<
			"(" << path << ") fails.\n";
	}
}

// Assertions are of the form that a field (e, f) should have a value y.
// Also need assertions for messaging. The above form may work still.

class Tester1 {
	public:
		Tester1( Shell* s, 
			void ( Shell::* func )( const string& arg ), 
			const string& funcname )
				:	s_( s ),
					func_( func ),
					funcname_( funcname ),
					counter_ ( 0 )
		{
			cout << "\nChecking shell::" << funcname_ << "(arg)";
		}

		void test( const string& arg, const string& field,
			const string& test ) {
			++counter_;
			( s_->*func_ )( arg );
			if ( s_->getFuncLocal( field ) == test ) {
				cout << ".";
			} else {
				cout << "!\n";
				cout << "Assert " << counter_ << ": " << funcname_ <<
					"(" << arg << ") fails to give\n";
				cout << field << " == " << test << "\n";
			}
		}
	private:
		Shell* s_;
		void ( Shell::* func_ )( const string& arg );
		const string funcname_;
		unsigned int counter_;
};

class Tester2 {
	public:
		Tester2(
		Shell* s, 
		void ( Shell::* func )( const string& arg1, const string& arg2),
		const string& funcname
		)
					: 	s_( s ),
					func_( func ),
					funcname_( funcname ),
					counter_( 0 )
		{
			cout << "Checking shell::" << funcname_ << "(arg1, arg2)";
		}
		void test( const string& arg1, const string& arg2,
			const string& field, const string& test ) {
			++counter_;
			( s_->*func_ )( arg1, arg2 );
			if ( s_->getFuncLocal( field ) == test ) {
				cout << ".";
			} else {
				cout << "!\n";
				cout << "Assert " << counter_ << ": " << funcname_ <<
					"(" << arg1 << ", " << arg2 << ") fails to give\n";
				cout << field << " == " << test << "\n";
			}
		}
	private:
		Shell* s_;
		void ( Shell::*func_ )( const string& arg1, const string& arg2);
		const string funcname_;
		unsigned int counter_;
};

void testShell()
{
	Element* tshell = Element::root()->relativeFind( "sli_shell" );

	cout << "Checking shell->findElement";

	ShellWrapper* shell = dynamic_cast< ShellWrapper* >( tshell );
	if ( shell ) {
		cout << "."; 
	} else  {
		cout << "Failed to create shell\n";
		return;
	}
	
	testFindElement( shell, "/", Element::root() );
	testFindElement( shell, "", Element::root() );
	testFindElement( shell, "/root", Element::root() );
	testFindElement( shell, "root", Element::root() );
	testFindElement( shell, ".", Element::root() );
	testFindElement( shell, "./", Element::root() );
	testFindElement( shell, "/.", Element::root() );
	testFindElement( shell, "/classes", Element::classes() );
	testFindElement( shell, "classes", Element::classes() );
	testFindElement( shell, "./classes", Element::classes() );
	testFindElement( shell, "/classes/..", Element::root() );
	testFindElement( shell, "classes/..", Element::root() );
	testFindElement( shell, "..", Element::root() );
	testFindElement( shell, "/..", Element::root() );
	testFindElement( shell, "classes/../..", Element::root() );
	cout << " done\n";

	Tester2 testCreate(
		shell, &Shell::createFuncLocal, "createFuncLocal");
	testCreate.test( "Int", "/i1", "/i1/value", "0");
	testCreate.test( "Int", "i2", "/i2/value", "0");
	testCreate.test( "Int", "./i3", "/i3/value", "0");
	testCreate.test( "Int", "../i4", "/i4/value", "0");
	testCreate.test( "Neutral", "/n1", "/n1/name", "n1");
	testCreate.test( "Int", "/n1/i5", "/n1/i5/value", "0");
	testCreate.test( "Int", "n1/i6", "/n1/i6/value", "0");
	testCreate.test( "Int", "n1/../i7", "/i7/value", "0");
	cout << " done\n";

	Tester2 testSet( shell, &Shell::setFuncLocal, "setFuncLocal");
	testSet.test( "/i1/value", "1", "/i1/value", "1");
	testSet.test( "/i1/value", "2", "/i1/value", "2");
	testSet.test( "/i1/strval", "3", "/i1/value", "3");
	testSet.test( "/i1/strval", "4", "/i1/strval", "4");
	testSet.test( "/i1/value", "5", "/i1/strval", "5");
	testSet.test( "/i1/name", "new_i1", "/new_i1/value", "5");
	cout << " done\n";

	Tester2 testCopy( shell, &Shell::copyFuncLocal, "copyFuncLocal");
	shell->setFuncLocal( "/n1/i5/value", "55" );
	shell->setFuncLocal( "/n1/i6/value", "66" );
	testCopy.test( "/n1", "/n2", "/n2/i5/value", "55");
	testCopy.test( "/n1", "/n3", "/n3/name", "n3");
	testCopy.test( "/n2", "/n3", "/n3/n2/i6/value", "66");
	testCopy.test( "/n2", "/n3/n4", "/n3/n4/i5/value", "55");
	testCopy.test( "/n3", "/n2/n4", "/n3/n4/i5/value", "55");
	testCopy.test( "/n2", "/n3/n4/n5", "/n3/n4/n5/i6/value", "66");
	cout << " done\n";
	cout << "Shell Tests complete\n\n";
}

void testBasecode()
{
	testShell();
	testWildcard();
	testParseArgs();
	delete Element::find( "/n1" );
	delete Element::find( "/n2" );
	delete Element::find( "/n3" );
	delete Element::find( "/new_i1" );
	delete Element::find( "/i2" );
	delete Element::find( "/i3" );
	delete Element::find( "/i4" );
	delete Element::find( "/i7" );
}

#endif
