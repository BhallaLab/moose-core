%module moose
%include "attribute.i"
%include "std_string.i"
%include "std_vector.i"
%{
	#include "../basecode/header.h"
	#include "../basecode/moose.h"
	#include "PyMooseContext.h"
	#include "PyMooseBase.h"
	#include "Neutral.h"
	#include "Compartment.h"
%}


%template(uint_vector) std::vector<unsigned int>;
%template(int_vector) std::vector<int>;
%template(double_vector) std::vector<double>;
%template(string_vector) std::vector<std::string>;
%include "../basecode/header.h"
%include "../basecode/moose.h"
%include "PyMooseContext.h"
%include "PyMooseBase.h"
%include "Compartment.h"
%include "Neutral.h"
%attribute(PyMooseBase, unsigned int, id, __get_id)
%attribute(PyMooseBase, unsigned int, parent, __get_parent)
%attribute(PyMooseBase, vector <unsigned int>&, children, __get_children)
%attribute(Compartment, double, Em, __get_Em, __set_Em)
%attribute(Compartment, double, Vm, __get_Vm, __set_Vm)
%attribute(Compartment, double, Rm, __get_Rm, __set_Rm)
%attribute(Compartment, double, Ra, __get_Ra, __set_Ra)
%attribute(Compartment, double, Cm, __get_Cm, __set_Cm)
%attribute(Compartment, double, initVm, __get_initVm, __set_initVm)
%attribute(Compartment, double, inject, __get_inject, __set_inject)
%attribute(Compartment, double, diameter, __get_diameter, __set_diameter)
%attribute(Compartment, double, length, __get_length, __set_length)
%attribute(Compartment, double, Im, __get_Im, __set_Im)
