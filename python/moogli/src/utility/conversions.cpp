#include <string>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <Python.h>
#include <iostream>

using namespace std;


unsigned
stou(string const& str, size_t * idx, int base)
{
    unsigned long number = stoul(str, idx, base);
    if (number > numeric_limits<unsigned>::max())
    {
        throw out_of_range("stou");
    }
    return number;
}

bool
equalf( float first_value
      , float second_value
      , float tolerance_percentage
      )
{
    return ( abs(first_value - second_value)
           < abs(first_value * tolerance_percentage)
           );
}


string
float_to_string(const float& value)
{
    //(numeric_limits<float>::digits10 + 1)
    ostringstream out;
    out << showpos << scientific << value;
    return out.str();
}


unsigned long
get_unsigned_long_from_pydictobject( PyObject * python_dict_object
                             , const char * key
                             )
{
    return PyInt_AsUnsignedLongMask(PyDict_GetItemString( python_dict_object
                                                        , "id"
                                                        )
                                   );

}

unsigned int
get_unsigned_int_from_pydictobject( PyObject * python_dict_object
                            , const char * key
                            )
{
    return static_cast<unsigned int>(
        get_unsigned_long_from_pydictobject(python_dict_object, key)
                                    );
}


const char *
get_c_string_from_pydictobject( PyObject * python_dict_object
                        , const char * key
                        )
{
    return PyString_AsString( PyDict_GetItemString( python_dict_object
                                                  , key
                                                  )
                            );
}

string
get_cpp_string_from_pydictobject( PyObject * python_dict_object
                          , const char * key
                          )
{
    return string(get_c_string_from_pydictobject(python_dict_object, key));
}

double
get_double_from_pydictobject( PyObject * python_dict_object
                      , const char * key
                      )
{
    return PyFloat_AsDouble(PyDict_GetItemString( python_dict_object
                                                , key
                                                )
                           );
}

float
get_float_from_pydictobject( PyObject * python_dict_object
                     , const char * key
                     )
{
    return static_cast<float>(get_double_from_pydictobject( python_dict_object
                                                    , key
                                                    )
                             );
}

