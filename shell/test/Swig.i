%module shell 
%include "std_string.i"
%{
#include <string>
#include "Swig.h"
%}

// Now list ANSI C/C++ declarations
extern const std::string& pwe();
extern void ce( const std::string& dest );
extern void create( const std::string& path );
extern void remove( const std::string& path );
extern void le ( const std::string& dest );
extern void le ( );
