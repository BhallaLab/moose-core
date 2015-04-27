#ifndef _CONVERSIONS_HPP_
#define _CONVERSIONS_HPP_

#include <string>
#include "Python.h"

using namespace std;


unsigned
stou(string const& str, size_t * idx = 0, int base = 10);

bool
equalf( float first_value
      , float second_value
      , float tolerance_percentage
      );

string
float_to_string(const float& value);

unsigned long
get_unsigned_long_from_pydictobject( PyObject * python_dict_object
                             , const char * key
                             );
unsigned int
get_unsigned_int_from_pydictobject( PyObject * python_dict_object
                            , const char * key
                            );

const char *
get_c_string_from_pydictobject( PyObject * python_dict_object
                        , const char * key
                        );

string
get_cpp_string_from_pydictobject( PyObject * python_dict_object
                          , const char * key
                          );

double
get_double_from_pydictobject( PyObject * python_dict_object
                      , const char * key
                      );

float
get_float_from_pydictobject( PyObject * python_dict_object
                     , const char * key
                     );

#endif /* _CONVERSIONS_HPP_ */
