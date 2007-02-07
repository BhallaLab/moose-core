%module pymoose 
%include "std_string.i"
%{
#include <string>
#include "Swig.h"
%}

// Now list ANSI C/C++ declarations
extern void pwe();
extern void ce( const std::string& dest );
extern void create( const std::string& type, const std::string& path );
extern void destroy( const std::string& path );
extern void le ( const std::string& dest );
extern void le ( );
