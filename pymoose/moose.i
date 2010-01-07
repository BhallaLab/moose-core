%module moose
%include "attribute.i"
%include "std_string.i"
%include "std_vector.i"
%{
	#define SWIG_FILE_WITH_INIT
	#include "../basecode/header.h"
	#include "../basecode/moose.h"
	#include "../utility/utility.h"
	#include "PyMooseContext.h"
	#include "PyMooseBase.h"
	#include "Neutral.h"
	#include "Class.h"
	#include "Cell.h"
	#include "Compartment.h"
	#include "Tick.h" 
	#include "ClockJob.h" 
	#include "Interpol.h"
	#include "TableIterator.h"
	#include "Table.h"
	#include "SynChan.h"
	#include "BinSynchan.h"
	#include "StochSynchan.h"
	#include "SpikeGen.h"
	#include "PulseGen.h"
	#include "RandomSpike.h"
	#include "Nernst.h"
	#include "CaConc.h"
	#include "HHGate.h"
	#include "Leakage.h"
	#include "HHChannel.h"
	#include "Mg_block.h"
	#include "Compartment.h"
	#include "NeuroScan.h"
	#include "HSolve.h"
	#include "Enzyme.h"
	#include "KineticHub.h"		
	#include "Kintegrator.h"
	#include "MathFunc.h"
	#include "Molecule.h"
	#include "Reaction.h"
	#include "Stoich.h"
	#include "KineticManager.h"
	#include "KinCompt.h"
	#include "Panel.h"
	#include "DiskPanel.h"
	#include "CylPanel.h"
	#include "HemispherePanel.h"
	#include "SpherePanel.h"
	#include "TriPanel.h"
	#include "RectPanel.h"
	#include "Surface.h"
	#include "Geometry.h"
 	#include "Adaptor.h"
 	#include "SigNeur.h"
	#include "AscFile.h"
	#include "DifShell.h"
	#include "GssaStoich.h"
	#include "TauPump.h"
        #include "GLcell.h"
	#include "TimeTable.h"
	#include "PIDController.h"
	#include "DiffAmp.h"
	#include "RC.h"
	#include "IntFire.h"
	#include "IzhikevichNrn.h"
	#include "GHK.h"
//	#include "../kinetics/SparseMatrix.h"
	#include "../utility/utility.h"
	/* Random number related utilities */
	#include "../randnum/randnum.h"
	/* These are the raw generic C++ classes - without any dependency on MOOSE */
	#include "../randnum/Probability.h"
	#include "../randnum/Binomial.h"
	#include "../randnum/Gamma.h"
	#include "../randnum/Normal.h"
	#include "../randnum/Poisson.h"
	#include "../randnum/Exponential.h"
	/* The following are moose classes */
	#include "RandGenerator.h"
	#include "BinomialRng.h"
	#include "GammaRng.h"
	#include "NormalRng.h"
	#include "PoissonRng.h"
	#include "ExponentialRng.h"
	#include "UniformRng.h"
	#include "HHGate2D.h"
	#include "HHChannel2D.h"
#ifdef USE_NUMPY
#include <algorithm>
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_pymoose_
#endif
#include <numpy/arrayobject.h>
#endif //!USE_NUMPY
#include "../builtins/Interpol.h"

%}


%init %{
#ifdef USE_NUMPY
      import_array();
#endif
%}

%feature("autodoc", "1");
%template(uint_vector) std::vector<unsigned int>;
%template(int_vector) std::vector<int>;
%template(double_vector) std::vector<double>;
%template(string_vector) std::vector<std::string>;
%template(Id_vector) std::vector<Id>;


%include "../basecode/header.h"
%include "../basecode/moose.h"
%ignore mooseInit;
%include "../utility/utility.h"
%ignore main; // this does not work, friend main() seems to interfere inspite of otherwise being stated in documentation
//%ignore ConnTainer;
%ignore Id::operator();
%ignore operator<<;
%ignore operator>>;
%ignore Id::eref;   

%include "../basecode/Id.h"
%include "../utility/Property.h"
%include "../utility/PathUtility.h"

%include "PyMooseContext.h"
%include "PyMooseBase.h"
%attribute(pymoose::PyMooseBase, Id*, id, __get_id)
%attribute(pymoose::PyMooseBase, Id*, parent, __get_parent)
%attribute(pymoose::PyMooseBase, const std::string, className, __get_className)
%attribute(pymoose::PyMooseBase, const std::string, author, __get_author)
%attribute(pymoose::PyMooseBase, const std::string, description, __get_description)

%pythoncode %{
from inspect import isclass

def doc(cls):
    """Return documentation string from MOOSE"""
    if isclass(cls):
        return PyMooseBase.getContext().doc(cls.__name__)
    elif isinstance(cls, PyMooseBase):
        return PyMooseBase.getContext().doc(cls.className)
    elif isinstance(cls, str):
        return PyMooseBase.getContext().doc(cls)		
%} // !pythoncode
	    
%attribute(pymoose::PyMooseBase, const std::string, name, __get_name)
%attribute(pymoose::PyMooseBase, const std::string, path, __get_path)

%include "Neutral.h"
%attribute(pymoose::Neutral, int, child, __get_child, __set_child)

%include "Class.h"
%attribute(pymoose::Class, std::string, name, __get_name, __set_name)
%attribute(pymoose::Class, std::string, author, __get_author)
%attribute(pymoose::Class, std::string, description, __get_description)
%attribute(pymoose::Class, unsigned int, tick, __get_tick, __set_tick)
%attribute(pymoose::Class, unsigned int, stage, __get_stage, __set_stage)

%include "Cell.h"
%attribute(pymoose::Cell, string, method, __get_method, __set_method)
%attribute(pymoose::Cell, bool, variableDt, __get_variableDt)
%attribute(pymoose::Cell, bool, implicit, __get_implicit)
%attribute(pymoose::Cell, string, description, __get_description)

%include "Tick.h"
%attribute(pymoose::Tick, double, dt, __get_dt, __set_dt)
%attribute(pymoose::Tick, int, stage, __get_stage, __set_stage)
%attribute(pymoose::Tick, int, ordinal, __get_ordinal, __set_ordinal)
%attribute(pymoose::Tick, double, nextTime, __get_nextTime, __set_nextTime)
//%attribute(pymoose::Tick, string&, path, __get_path, __set_path) 
%attribute(pymoose::Tick, double, updateDtSrc, __get_updateDtSrc, __set_updateDtSrc)

%include "ClockJob.h"
%attribute(pymoose::ClockJob, double, runTime, __get_runTime, __set_runTime)
%attribute(pymoose::ClockJob, double, currentTime, __get_currentTime, __set_currentTime)
%attribute(pymoose::ClockJob, int, nsteps, __get_nsteps, __set_nsteps)
%attribute(pymoose::ClockJob, int, currentStep, __get_currentStep, __set_currentStep)
%attribute(pymoose::ClockJob, double, start, __get_start, __set_start)
%attribute(pymoose::ClockJob, int, step, __get_step, __set_step)

%include "Interpol.h"
/* Numpy interface for Interpol */
%extend pymoose::Interpol{
#ifdef USE_NUMPY
PyObject* get__array_struct__()
{
    PyArrayObject* result;
    int dimensions[1];
    static vector <double> data;
    printf("In __array_struct__\n");
    data = Interpol::getTableVector((*($self->__get_id()))());
    dimensions[0] = data.size();
    // avoiding shared data copy - dangerous and the vector returned by getTableVector is a temporary copy 
    // result = (PyArrayObject*)PyArray_FromDimsAndData(1, dimensions, NPY_DOUBLE, (char*)(&data[0]));
    // instead create a PyArrayObject initialized with zero and then copy data
    result = (PyArrayObject*)PyArray_FromDims(1, dimensions, NPY_DOUBLE);
    memcpy(result->data, &data[0], dimensions[0]*sizeof(double));
    return PyArray_Return(result);
} // !get__array_struct__

/**
   This function fills a table object using a Python sequence type object
*/
#if 0
void fillData(PyObject* args)
{
    PyObject* input_seq;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O", &input_seq))
        return;
    array = (PyArrayObject *) PyArray_ContiguousFromObject(input_seq, PyArray_DOUBLE, 1, 1);
    if (array == NULL)
        return;
    this->__set_xdivs(array->dimensions[0] - 1);
    std::vector <double> data(array->dimensions[0], 0.0);
    double min = *std::min_element(data.begin(), data.end());
    this->__set_xmin(min);
    double max = *std::max_element(data.begin(), data.end());
    this->__set_xmax(max);
    // todo: should use stl copy
    memcpy(&data[0], array->data, sizeof(double)*(array->dimensions[0]));
    set <vector <double> > (id_(), "tableVector", data);
}
#endif
  %pythoncode %{
  __array_struct__ = property(get__array_struct__,
                                                 doc='Array protocol')
  %} // end pythoncode
#endif // !USE_NUMPY	     
}; // end of extend

%attribute(pymoose::Interpol, double, xmin, __get_xmin, __set_xmin)
%attribute(pymoose::Interpol, double, xmax, __get_xmax, __set_xmax)
%attribute(pymoose::Interpol, int, xdivs, __get_xdivs, __set_xdivs)
%attribute(pymoose::Interpol, int, mode, __get_mode, __set_mode)
%attribute(pymoose::Interpol, double, dx, __get_dx, __set_dx)
%attribute(pymoose::Interpol, double, sy, __get_sy, __set_sy)
%attribute(pymoose::Interpol, int, calcMode, __get_calcMode, __set_calcMode)

%include "TableIterator.h"
%extend pymoose::TableIterator
{	%insert("python")%{
		def _generator_(self):
			if self.__hasNext__():
				yield self.__next__()
		
		def next(self):
			return self._generator_().next()
			
	%}
}
/* The following does not work
%pythoncode %{
	pymoose::Interpol.__setitem__ = Interpol.__set_table
	pymoose::Interpol.__getitem__ = Interpol.__get_table
%} // end pythoncode
*/
//%attribute(pymoose::Interpol, string&, dumpFile, __get_print, __set_print) 

%include "Table.h"
%attribute(pymoose::Table, double, input, __get_input, __set_input)
%attribute(pymoose::Table, double, output, __get_output, __set_output)
%attribute(pymoose::Table, int, stepMode, __get_stepMode, __set_stepMode)
%attribute(pymoose::Table, double, stepSize, __get_stepSize, __set_stepSize)
%attribute(pymoose::Table, double, threshold, __get_threshold, __set_threshold)
//%attribute(pymoose::Table, double, tableLookup, __get_tableLookup, __set_tableLookup)
//%attribute(pymoose::Table, double, outputSrc, __get_outputSrc, __set_outputSrc)
//%attribute(pymoose::Table, double, msgInput, __get_msgInput, __set_msgInput)
// %attribute(pymoose::Table, double, sum, __get_sum, __set_sum)
// %attribute(pymoose::Table, double, prd, __get_prd, __set_prd)

%include "SynChan.h"
%attribute(pymoose::SynChan, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(pymoose::SynChan, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::SynChan, double, tau1, __get_tau1, __set_tau1)
%attribute(pymoose::SynChan, double, tau2, __get_tau2, __set_tau2)
%attribute(pymoose::SynChan, bool, normalizeWeights, __get_normalizeWeights, __set_normalizeWeights)
%attribute(pymoose::SynChan, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::SynChan, double, Ik, __get_Ik)
%attribute(pymoose::SynChan, unsigned int, numSynapses, __get_numSynapses)
%pythoncode %{
class Weight:
      def __init__(self, synchan):
      	  self.synchan = synchan
      def __setitem__(index, value):
      	  self.synchan.setWeight(index, value)
      def __getitem__(index):
      	  return self.synchan.getWeight(index)	  	        
class Delay:
      def __init__(self, synchan):
      	  self.synchan = synchan
      def __setitem__(index, value):
      	  self.synchan.setDelay(index, value)
      def __getitem__(index):
      	  return self.synchan.getDelay(index)
%} //end pythoncode
%include "BinSynchan.h"
%attribute(pymoose::BinSynchan, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(pymoose::BinSynchan, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::BinSynchan, double, tau1, __get_tau1, __set_tau1)
%attribute(pymoose::BinSynchan, double, tau2, __get_tau2, __set_tau2)
%attribute(pymoose::BinSynchan, bool, normalizeWeights, __get_normalizeWeights, __set_normalizeWeights)
%attribute(pymoose::BinSynchan, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::BinSynchan, double, Ik, __get_Ik)
%attribute(pymoose::BinSynchan, unsigned int, numSynapses, __get_numSynapses)

%include "StochSynchan.h"
%attribute(pymoose::StochSynchan, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(pymoose::StochSynchan, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::StochSynchan, double, tau1, __get_tau1, __set_tau1)
%attribute(pymoose::StochSynchan, double, tau2, __get_tau2, __set_tau2)
%attribute(pymoose::StochSynchan, bool, normalizeWeights, __get_normalizeWeights, __set_normalizeWeights)
%attribute(pymoose::StochSynchan, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::StochSynchan, double, Ik, __get_Ik, __set_Ik)
%attribute(pymoose::StochSynchan, unsigned int, numSynapses, __get_numSynapses)
%attribute(pymoose::StochSynchan, double, activation, __get_activation, __set_activation)
%attribute(pymoose::StochSynchan, double, modulator, __get_modulator, __set_modulator)

//%include "PyMooseIterable.h"
//%template(BinSynchanDILookup) InnerPyMooseIterable < BinSynchan, unsigned int, double > ;
//%template(StochSynchanDILookup) InnerPyMooseIterable < StochSynchan, unsigned int, double > ;

%include "SpikeGen.h"
%attribute(pymoose::SpikeGen, double, threshold, __get_threshold, __set_threshold)
%attribute(pymoose::SpikeGen, double, refractT, __get_refractT, __set_refractT)
%attribute(pymoose::SpikeGen, double, absRefractT, __get_absRefractT, __set_absRefractT)
%attribute(pymoose::SpikeGen, double, amplitude, __get_amplitude, __set_amplitude)
%attribute(pymoose::SpikeGen, double, state, __get_state, __set_state)
%attribute(pymoose::SpikeGen, double, event, __get_event, __set_event)
%attribute(pymoose::SpikeGen, double, Vm, __get_Vm, __set_Vm)

%include "RandomSpike.h"
%attribute(pymoose::RandomSpike, double, minAmp, __get_minAmp, __set_minAmp)
%attribute(pymoose::RandomSpike, double, maxAmp, __get_maxAmp, __set_maxAmp)
%attribute(pymoose::RandomSpike, double, rate, __get_rate, __set_rate)
%attribute(pymoose::RandomSpike, double, resetValue, __get_resetValue, __set_resetValue)
%attribute(pymoose::RandomSpike, double, state, __get_state, __set_state)
%attribute(pymoose::RandomSpike, double, absRefract, __get_absRefract, __set_absRefract)
%attribute(pymoose::RandomSpike, double, lastEvent, __get_lastEvent)
%attribute(pymoose::RandomSpike, int, reset, __get_reset, __set_reset)

%include "PulseGen.h"
%attribute(pymoose::PulseGen, double, firstLevel, __get_firstLevel, __set_firstLevel)
%attribute(pymoose::PulseGen, double, firstWidth, __get_firstWidth, __set_firstWidth)
%attribute(pymoose::PulseGen, double, firstDelay, __get_firstDelay, __set_firstDelay)
%attribute(pymoose::PulseGen, double, secondLevel, __get_secondLevel, __set_secondLevel)
%attribute(pymoose::PulseGen, double, secondWidth, __get_secondWidth, __set_secondWidth)
%attribute(pymoose::PulseGen, double, secondDelay, __get_secondDelay, __set_secondDelay)
%attribute(pymoose::PulseGen, double, baseLevel, __get_baseLevel, __set_baseLevel)
%attribute(pymoose::PulseGen, double, output, __get_output)
%attribute(pymoose::PulseGen, double, trigTime, __get_trigTime, __set_trigTime)
%attribute(pymoose::PulseGen, int, trigMode, __get_trigMode, __set_trigMode)
%attribute(pymoose::PulseGen, int, prevInput, __get_prevInput)

%include "Nernst.h"
%attribute(pymoose::Nernst, double, E, __get_E)
%attribute(pymoose::Nernst, double, Temperature, __get_Temperature, __set_Temperature)
%attribute(pymoose::Nernst, int, valence, __get_valence, __set_valence)
%attribute(pymoose::Nernst, double, Cin, __get_Cin, __set_Cin)
%attribute(pymoose::Nernst, double, Cout, __get_Cout, __set_Cout)
%attribute(pymoose::Nernst, double, scale, __get_scale, __set_scale)

%include "CaConc.h"
%attribute(pymoose::CaConc, double, Ca, __get_Ca, __set_Ca)
%attribute(pymoose::CaConc, double, CaBasal, __get_CaBasal, __set_CaBasal)
%attribute(pymoose::CaConc, double, Ca_base, __get_Ca_base, __set_Ca_base)
%attribute(pymoose::CaConc, double, tau, __get_tau, __set_tau)
%attribute(pymoose::CaConc, double, B, __get_B, __set_B)
%attribute(pymoose::CaConc, double, thick, __get_thick, __set_thick)
%attribute(pymoose::CaConc, double, ceiling, __get_ceiling, __set_ceiling)
%attribute(pymoose::CaConc, double, floor, __get_floor, __set_floor)

%include "HHGate.h"
%attribute(pymoose::HHGate, Interpol*, A, __get_A)
%attribute(pymoose::HHGate, Interpol*, B, __get_B)

%include "Leakage.h"
%attribute(pymoose::Leakage, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::Leakage, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::Leakage, double, Ik, __get_Ik)
%attribute(pymoose::Leakage, double, activation, __get_activation, __set_activation)

%include "HHChannel.h"
%attribute(pymoose::HHChannel, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(pymoose::HHChannel, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::HHChannel, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::HHChannel, double, Xpower, __get_Xpower, __set_Xpower)
%attribute(pymoose::HHChannel, double, Ypower, __get_Ypower, __set_Ypower)
%attribute(pymoose::HHChannel, double, Zpower, __get_Zpower, __set_Zpower)
%attribute(pymoose::HHChannel, double, X, __get_X, __set_X)
%attribute(pymoose::HHChannel, double, Y, __get_Y, __set_Y)
%attribute(pymoose::HHChannel, double, Z, __get_Z, __set_Z)
%attribute(pymoose::HHChannel, double, instant, __get_instant, __set_instant)
%attribute(pymoose::HHChannel, int, useConcentration, __get_useConcentration, __set_useConcentration)


%include "Mg_block.h"
%attribute(pymoose::Mg_block, double, KMg_A, __get_KMg_A, __set_KMg_A)
%attribute(pymoose::Mg_block, double, K_MgB, __get_KMg_B, __set_KMg_B)
%attribute(pymoose::Mg_block, double, CMg, __get_CMg, __set_CMg)
%attribute(pymoose::Mg_block, double, Ik, __get_Ik, __set_Ik)
%attribute(pymoose::Mg_block, double, Gk, __get_Gk,  __set_Gk)
%attribute(pymoose::Mg_block, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::Mg_block, double, Zk, __get_Zk, __set_Zk)

%include "Compartment.h"
%attribute(pymoose::Compartment, double, Vm, __get_Vm, __set_Vm)
%attribute(pymoose::Compartment, double, Cm, __get_Cm, __set_Cm)
%attribute(pymoose::Compartment, double, Em, __get_Em, __set_Em)
%attribute(pymoose::Compartment, double, Im, __get_Im, __set_Im)
%attribute(pymoose::Compartment, double, inject, __get_inject, __set_inject)
%attribute(pymoose::Compartment, double, initVm, __get_initVm, __set_initVm)
%attribute(pymoose::Compartment, double, Rm, __get_Rm, __set_Rm)
%attribute(pymoose::Compartment, double, Ra, __get_Ra, __set_Ra)
%attribute(pymoose::Compartment, double, diameter, __get_diameter, __set_diameter)
%attribute(pymoose::Compartment, double, length, __get_length, __set_length)
%attribute(pymoose::Compartment, double, x, __get_x, __set_x)
%attribute(pymoose::Compartment, double, y, __get_y, __set_y)
%attribute(pymoose::Compartment, double, z, __get_z, __set_z)
%attribute(pymoose::Compartment, double, x0, __get_x0, __set_x0)
%attribute(pymoose::Compartment, double, y0, __get_y0, __set_y0)
%attribute(pymoose::Compartment, double, z0, __get_z0, __set_z0)

%include "NeuroScan.h"
%attribute(pymoose::NeuroScan, int, VDiv, __get_VDiv, __set_VDiv)
%attribute(pymoose::NeuroScan, double, VMin, __get_VMin, __set_VMin)
%attribute(pymoose::NeuroScan, double, VMax, __get_VMax, __set_VMax)
%attribute(pymoose::NeuroScan, int, CaDiv, __get_CaDiv, __set_CaDiv)
%attribute(pymoose::NeuroScan, double, CaMin, __get_CaMin, __set_CaMin)
%attribute(pymoose::NeuroScan, double, CaMax, __get_CaMax, __set_CaMax)



%include "HSolve.h"
%attribute(pymoose::HSolve, string, seedPath, __get_seed_path, __set_seed_path)
%attribute(pymoose::HSolve, int, NDiv, __get_NDiv, __set_NDiv)
%attribute(pymoose::HSolve, double, VLo, __get_VLo, __set_VLo)
%attribute(pymoose::HSolve, double, VHi, __get_VHi, __set_VHi)

%include "Kintegrator.h"
%attribute(pymoose::Kintegrator, bool, isInitiatilized, __get_isInitiatilized, __set_isInitiatilized)
//%attribute(pymoose::Kintegrator, string, integrate_method, __get_method, __set_method)
//%attribute_ref(Kintegrator, string, method)
%include "MathFunc.h"
%attribute(pymoose::MathFunc, string, mathML, __get_mathML, __set_mathML)
%attribute(pymoose::MathFunc, string, function, __get_function, __set_function)
%attribute(pymoose::MathFunc, double, result, __get_result, __set_result)

%include "Stoich.h"
%attribute(pymoose::Stoich, unsigned int, nMols, __get_nMols)
%attribute(pymoose::Stoich, unsigned int, nVarMols, __get_nVarMols)
%attribute(pymoose::Stoich, unsigned int, nSumTot, __get_nSumTot)
%attribute(pymoose::Stoich, unsigned int, nBuffered, __get_nBuffered)
%attribute(pymoose::Stoich, unsigned int, nReacs, __get_nReacs)
%attribute(pymoose::Stoich, unsigned int, nEnz, __get_nEnz)
%attribute(pymoose::Stoich, unsigned int, nMMenz, __get_nMMenz)
%attribute(pymoose::Stoich, unsigned int, nExternalRates, __get_nExternalRates)
%attribute(pymoose::Stoich, bool, useOneWayReacs, __get_useOneWayReacs, __set_useOneWayReacs)
//%attribute(pymoose::Stoich, string, path, __get_path, __set_path) -- path here is something different from element path
%attribute(pymoose::Stoich, unsigned int, rateVectorSize, __get_rateVectorSize)

%include "KineticHub.h"
%attribute(pymoose::KineticHub, unsigned int, nMol, __get_nMol, __set_nMol)
%attribute(pymoose::KineticHub, unsigned int, nReac, __get_nReac, __set_nReac)
%attribute(pymoose::KineticHub, unsigned int, nEnz, __get_nEnz, __set_nEnz)
%attribute(pymoose::KineticHub, double, molSum, __get_molSum, __set_molSum)

%include "Enzyme.h"
%attribute(pymoose::Enzyme, double, k1, __get_k1, __set_k1)
%attribute(pymoose::Enzyme, double, k2, __get_k2, __set_k2)
%attribute(pymoose::Enzyme, double, k3, __get_k3, __set_k3)
%attribute(pymoose::Enzyme, double, Km, __get_Km, __set_Km)
%attribute(pymoose::Enzyme, double, kcat, __get_kcat, __set_kcat)
%attribute(pymoose::Enzyme, bool, mode, __get_mode, __set_mode)
//%attribute(pymoose::Enzyme, double,double, prd, __get_prd, __set_prd)

%include "Reaction.h"
%attribute(pymoose::Reaction, double, kf, __get_kf, __set_kf)
%attribute(pymoose::Reaction, double, kb, __get_kb, __set_kb)
%attribute(pymoose::Reaction, double, scaleKf, __get_scaleKf, __set_scaleKf)
%attribute(pymoose::Reaction, double, scaleKb, __get_scaleKb, __set_scaleKb)

%include "Molecule.h"
%attribute(pymoose::Molecule, double, nInit, __get_nInit, __set_nInit)
%attribute(pymoose::Molecule, double, volumeScale, __get_volumeScale, __set_volumeScale)
%attribute(pymoose::Molecule, double, n, __get_n, __set_n)
%attribute(pymoose::Molecule, int, mode, __get_mode, __set_mode)
%attribute(pymoose::Molecule, int, slave_enable, __get_slave_enable, __set_slave_enable)
%attribute(pymoose::Molecule, double, conc, __get_conc, __set_conc)
%attribute(pymoose::Molecule, double, concInit, __get_concInit, __set_concInit)
%attribute(pymoose::Molecule, double, nSrc, __get_nSrc, __set_nSrc)
//%attribute(pymoose::Molecule, double,double, prd, __get_prd, __set_prd)
%attribute(pymoose::Molecule, double, sumTotal, __get_sumTotal, __set_sumTotal)

/* %include "Enzyme.h"
	#include "KineticHub.h"		
	#include "Kintegrator.h"
	#include "Molecule.h"
	#include "Reaction.h"
	#include "Stoich.h"
	#include "../kinetics/SparseMatrix.h"
*/
/*
%include "TickTest.h"
%include "Sched0.h"
%include "Sched1.h"
%include "Sched2.h"
*/

//**********************************
// Random number related utilities *	
//**********************************
%include "../randnum/randnum.h"
/* These are the raw generic C++ classes - without any dependency on MOOSE */
%include "../randnum/Probability.h"
%include "../randnum/Binomial.h"
%include "../randnum/Gamma.h"
%include "../randnum/Normal.h"
%include "../randnum/Poisson.h"
%include "../randnum/Exponential.h"
/* The following are moose classes */

%include "RandGenerator.h"
%attribute(pymoose::RandGenerator, double, sample, __get_sample)
%attribute(pymoose::RandGenerator, double, mean, __get_mean)
%attribute(pymoose::RandGenerator, double, variance, __get_variance)

%include "UniformRng.h"
%attribute(pymoose::UniformRng, double, mean, __get_mean)
%attribute(pymoose::UniformRng, double, variance, __get_variance)
%attribute(pymoose::UniformRng, double, min, __get_min, __set_min)
%attribute(pymoose::UniformRng, double, max, __get_max, __set_max)

%include "GammaRng.h"
%attribute(pymoose::GammaRng, double, alpha, __get_alpha, __set_alpha)
%attribute(pymoose::GammaRng, double, theta, __get_theta, __set_theta)

%include "ExponentialRng.h"
%attribute(pymoose::ExponentialRng, double, mean, __get_mean, __set_mean)
%attribute(pymoose::ExponentialRng, int, method, __get_method, __set_method)

%include "BinomialRng.h"
%attribute(pymoose::BinomialRng, int, n, __get_n, __set_n)
%attribute(pymoose::BinomialRng, double, p, __get_p, __set_p)

%include "PoissonRng.h"
%attribute(pymoose::PoissonRng, double, mean, __get_mean, __set_mean)

%include "NormalRng.h"
%attribute(pymoose::NormalRng, double, mean, __get_mean, __set_mean)
%attribute(pymoose::NormalRng, double, variance, __get_variance, __set_variance)
%attribute(pymoose::NormalRng, int, method, __get_method, __set_method)

%include "KineticManager.h"
%attribute(pymoose::KineticManager, bool, autoMode, __get_autoMode, __set_autoMode)
%attribute(pymoose::KineticManager, bool, stochastic, __get_stochastic, __set_stochastic)
%attribute(pymoose::KineticManager, bool, spatial, __get_spatial, __set_spatial)
%attribute(pymoose::KineticManager, string, method, __get_method, __set_method)
%attribute(pymoose::KineticManager, bool, variableDt, __get_variableDt)
%attribute(pymoose::KineticManager, bool, singleParticle, __get_singleParticle)
%attribute(pymoose::KineticManager, bool, multiscale, __get_multiscale)
%attribute(pymoose::KineticManager, bool, implicit, __get_implicit)
%attribute(pymoose::KineticManager, string, description, __get_description)
%attribute(pymoose::KineticManager, double, recommendedDt, __get_recommendedDt)
%attribute(pymoose::KineticManager, double, eulerError, __get_eulerError, __set_eulerError)

%include "KinCompt.h"
%attribute(pymoose::KinCompt, double, volume, __get_volume, __set_volume)
%attribute(pymoose::KinCompt, double, area, __get_area, __set_area)
%attribute(pymoose::KinCompt, double, perimeter, __get_perimeter, __set_perimeter)
%attribute(pymoose::KinCompt, double, size, __get_size, __set_size)
%attribute(pymoose::KinCompt, unsigned int, numDimensions, __get_numDimensions, __set_numDimensions)

%include "Panel.h"
%attribute(pymoose::Panel, unsigned int, nPts, __get_nPts)
%attribute(pymoose::Panel, unsigned int, nDims, __get_nDims)
%attribute(pymoose::Panel, unsigned int, nNeighbors, __get_nNeighbors)
%attribute(pymoose::Panel, unsigned int, shapeId, __get_shapeId)
%attribute(pymoose::Panel, const vector<double>&, coords, __get_coords)

%include "DiskPanel.h"
%include "CylPanel.h"
%include "HemispherePanel.h"
%include "SpherePanel.h"
%include "TriPanel.h"
%include "RectPanel.h"
%include "Surface.h"
%attribute(pymoose::Surface, double, volume, __get_volume)
%include "Geometry.h"
%attribute(pymoose::Geometry, double, epsilon, __get_epsilon, __set_epsilon)
%attribute(pymoose::Geometry, double, neighdist, __get_neighdist, __set_neighdist)

 %include "Adaptor.h"
 %attribute(pymoose::Adaptor, double, inputOffset, __get_inputOffset, __set_inputOffset)
 %attribute(pymoose::Adaptor, double, outputOffset, __get_outputOffset, __set_outputOffset)
 %attribute(pymoose::Adaptor, double, scale, __get_scale, __set_scale)
 %attribute(pymoose::Adaptor, double, output, __get_output)

 %include "SigNeur.h"
 %attribute(pymoose::SigNeur, Id&, cellProto, __get_cellProto, __set_cellProto)
 %attribute(pymoose::SigNeur, Id&, spineProto, __get_spineProto, __set_spineProto)
 %attribute(pymoose::SigNeur, Id&, dendProto, __get_dendProto, __set_dendProto)
 %attribute(pymoose::SigNeur, Id&, somaProto, __get_somaProto, __set_somaProto)
 %attribute(pymoose::SigNeur, Id&, cell, __get_cell)
 %attribute(pymoose::SigNeur, Id&, spine, __get_spine)
 %attribute(pymoose::SigNeur, Id&, dend, __get_dend)
 %attribute(pymoose::SigNeur, Id&, soma, __get_soma)
 %attribute(pymoose::SigNeur, string, cellMethod, __get_cellMethod, __set_cellMethod)
 %attribute(pymoose::SigNeur, string, spineMethod, __get_spineMethod, __set_spineMethod)
 %attribute(pymoose::SigNeur, string, dendMethod, __get_dendMethod, __set_dendMethod)
 %attribute(pymoose::SigNeur, string, somaMethod, __get_somaMethod, __set_somaMethod)
 %attribute(pymoose::SigNeur, double, sigDt, __get_sigDt, __set_sigDt)
 %attribute(pymoose::SigNeur, double, cellDt, __get_cellDt, __set_cellDt)
 %attribute(pymoose::SigNeur, double, Dscale, __get_Dscale, __set_Dscale)
 %attribute(pymoose::SigNeur, double, lambda_, __get_lambda, __set_lambda)
 %attribute(pymoose::SigNeur, int, parallelMode, __get_parallelMode, __set_parallelMode)
 %attribute(pymoose::SigNeur, double, updateStep, __get_updateStep, __set_updateStep)
 %attribute(pymoose::SigNeur, double, calciumScale, __get_calciumScale, __set_calciumScale)
 %attribute(pymoose::SigNeur, string, dendInclude, __get_dendInclude, __set_dendInclude)
 %attribute(pymoose::SigNeur, string, dendExclude, __get_dendExclude, __set_dendExclude)

%include "AscFile.h"
%attribute(pymoose::AscFile, string, fileName, __get_fileName, __set_fileName)
%attribute(pymoose::AscFile, int, appendFlag, __get_appendFlag, __set_appendFlag)

%include "DifShell.h"
%attribute(pymoose::DifShell, double, C, __get_C)
%attribute(pymoose::DifShell, double, Ceq, __get_Ceq, __set_Ceq)
%attribute(pymoose::DifShell, double, D, __get_D, __set_D)
%attribute(pymoose::DifShell, double, valence, __get_valence, __set_valence)
%attribute(pymoose::DifShell, double, leak, __get_leak, __set_leak)
%attribute(pymoose::DifShell, unsigned int, shapeMode, __get_shapeMode, __set_shapeMode)
%attribute(pymoose::DifShell, double, length, __get_length, __set_length)
%attribute(pymoose::DifShell, double, diameter, __get_diameter, __set_diameter)
%attribute(pymoose::DifShell, double, thickness, __get_thickness, __set_thickness)
%attribute(pymoose::DifShell, double, volume, __get_volume, __set_volume)
%attribute(pymoose::DifShell, double, outerArea, __get_outerArea, __set_outerArea)
%attribute(pymoose::DifShell, double, innerArea, __get_innerArea, __set_innerArea)

%include "GssaStoich.h"
%attribute(pymoose::GssaStoich, string, method, __get_method, __set_method)
%attribute(pymoose::GssaStoich, string, path, __get_path, __set_path)

%include "TauPump.h"
%attribute(pymoose::TauPump, double, pumpRate, __get_pumpRate, __set_pumpRate)
%attribute(pymoose::TauPump, double, eqConc, __get_eqConc, __set_eqConc)
%attribute(pymoose::TauPump, double, TA, __get_TA, __set_TA)
%attribute(pymoose::TauPump, double, TB, __get_TB, __set_TB)
%attribute(pymoose::TauPump, double, TC, __get_TC, __set_TC)
%attribute(pymoose::TauPump, double, TV, __get_TV, __set_TV)

%include "GLcell.h"
%attribute(pymoose::GLcell, string, vizpath, __get_vizpath, __set_vizpath)
%attribute(pymoose::GLcell, string, host, __get_clientHost, __set_clientHost)
%attribute(pymoose::GLcell, string, port, __get_clientPort, __set_clientPort)
%attribute(pymoose::GLcell, string, attribute, __get_attributeName, __set_attributeName)
%attribute(pymoose::GLcell, double, threshold, __get_changeThreshold, __set_changeThreshold)
%attribute(pymoose::GLcell, double, vscale, __get_VScale, __set_VScale)
%attribute(pymoose::GLcell, string, sync, __get_syncMode, __set_syncMode)
%attribute(pymoose::GLcell, string, bgcolor, __get_bgColor, __set_bgColor)
%attribute(pymoose::GLcell, double, highvalue, __get_highValue, __set_highValue)
%attribute(pymoose::GLcell, double, lowvalue, __get_lowValue, __set_lowValue)

%include "TimeTable.h"
%attribute(pymoose::TimeTable, double, maxTime, __get_maxTime, __set_maxTime)
%attribute(pymoose::TimeTable, const vector < double >&, tableVector, __get_tableVector, __set_tableVector)
%attribute(pymoose::TimeTable, unsigned int, tableSize, __get_tableSize)

%include "RC.h"
%attribute(pymoose::RC, double, V0, __get_V0, __set_V0)
%attribute(pymoose::RC, double, R, __get_R, __set_R)
%attribute(pymoose::RC, double, C, __get_C, __set_C)
%attribute(pymoose::RC, double, state, __get_state)
%attribute(pymoose::RC, double, inject, __get_inject, __set_inject)

%include "PIDController.h"
%attribute(pymoose::PIDController, double, gain, __get_gain, __set_gain)
%attribute(pymoose::PIDController, double, saturation, __get_saturation, __set_saturation)
%attribute(pymoose::PIDController, double, command, __get_command, __set_command)
%attribute(pymoose::PIDController, double, sensed, __get_sensed)
%attribute(pymoose::PIDController, double, tauI, __get_tauI, __set_tauI)
%attribute(pymoose::PIDController, double, tauD, __get_tauD, __set_tauD)
%attribute(pymoose::PIDController, double, output, __get_output)

%include "DiffAmp.h"
%attribute(pymoose::DiffAmp, double, gain, __get_gain, __set_gain)
%attribute(pymoose::DiffAmp, double, saturation, __get_saturation, __set_saturation)
%attribute(pymoose::DiffAmp, double, plus, __get_plus)
%attribute(pymoose::DiffAmp, double, minus, __get_minus)
%attribute(pymoose::DiffAmp, double, output, __get_output)

%include "IntFire.h"
%attribute(pymoose::IntFire, double, Vt, __get_Vt, __set_Vt)
%attribute(pymoose::IntFire, double, Vr, __get_Vr, __set_Vr)
%attribute(pymoose::IntFire, double, Rm, __get_Rm, __set_Rm)
%attribute(pymoose::IntFire, double, Cm, __get_Cm, __set_Cm)
%attribute(pymoose::IntFire, double, Vm, __get_Vm, __set_Vm)
%attribute(pymoose::IntFire, double, tau, __get_tau)
%attribute(pymoose::IntFire, double, Em, __get_Em, __set_Em)
%attribute(pymoose::IntFire, double, refractT, __get_refractT, __set_refractT)
%attribute(pymoose::IntFire, double, initVm, __get_initVm, __set_initVm)
%attribute(pymoose::IntFire, double, inject, __get_inject, __set_inject)

%include "IzhikevichNrn.h"
%attribute(pymoose::IzhikevichNrn, double, Vmax, __get_Vmax, __set_Vmax)
%attribute(pymoose::IzhikevichNrn, double, c, __get_c, __set_c)
%attribute(pymoose::IzhikevichNrn, double, d, __get_d, __set_d)
%attribute(pymoose::IzhikevichNrn, double, a, __get_a, __set_a)
%attribute(pymoose::IzhikevichNrn, double, b, __get_b, __set_b)
%attribute(pymoose::IzhikevichNrn, double, Vm, __get_Vm, __set_Vm)
%attribute(pymoose::IzhikevichNrn, double, u, __get_u)
%attribute(pymoose::IzhikevichNrn, double, Im, __get_Im)
%attribute(pymoose::IzhikevichNrn, double, initVm, __get_initVm, __set_initVm)
%attribute(pymoose::IzhikevichNrn, double, initU, __get_initU, __set_initU)
%attribute(pymoose::IzhikevichNrn, double, alpha, __get_alpha, __set_alpha)
%attribute(pymoose::IzhikevichNrn, double, beta, __get_beta, __set_beta)
%attribute(pymoose::IzhikevichNrn, double, gamma, __get_gamma, __set_gamma)

%include "GHK.h"
%attribute(pymoose::GHK, double, Ik, __get_Ik)
%attribute(pymoose::GHK, double, Gk, __get_Gk)
%attribute(pymoose::GHK, double, Ek, __get_Ek)
%attribute(pymoose::GHK, double, T, __get_T, __set_T)
%attribute(pymoose::GHK, double, p, __get_p, __set_p)
%attribute(pymoose::GHK, double, Vm, __get_Vm, __set_Vm)
%attribute(pymoose::GHK, double, Cin, __get_Cin, __set_Cin)
%attribute(pymoose::GHK, double, Cout, __get_Cout, __set_Cout)
%attribute(pymoose::GHK, double, valency, __get_valency, __set_valency)



%include "HHChannel2D.h"
%attribute(pymoose::HHChannel2D, string, Xindex, __get_Xindex, __set_Xindex)
%attribute(pymoose::HHChannel2D, string, Yindex, __get_Yindex, __set_Yindex)
%attribute(pymoose::HHChannel2D, string, Zindex, __get_Zindex, __set_Zindex)
%include "HHGate2D.h"
