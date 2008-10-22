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
// 	#include "Adaptor.h"
// 	#include "SigNeur.h"
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
%}
//%feature("autodoc", "1");
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
%include "PyMooseContext.h"
%include "PyMooseBase.h"
%attribute(pymoose::PyMooseBase, Id*, id, __get_id)
%attribute(pymoose::PyMooseBase, Id*, parent, __get_parent)

// The following attributes, when traversed without assigning to a
// variable, give segmentation fault. May be some problem with the
// swig interface to vectors. But sticking to function form helps.

//%attribute(pymoose::PyMooseBase, vector <Id>, children, __get_children)
// %attribute(pymoose::PyMooseBase, vector <std::string>&, inMessages, __get_incoming_messages)
// %attribute(pymoose::PyMooseBase, vector <std::string>&, outMessages, __get_outgoing_messages)
%attribute(pymoose::PyMooseBase, const std::string, name, __get_name)
//%attribute(pymoose::PyMooseBase, string& , path, _path)
// The above gives segmentation fault, path is dynamically generated,
// so when using pointers, the memory may already have been deallocated
// better try writing to a string stream and returb stream.str()
//%ignore PyMooseBase::getPath;
//%attribute(pymoose::PyMooseBase, string, path, getPath)
%include "Neutral.h"
%attribute(pymoose::Neutral, int, childSrc, __get_childSrc, __set_childSrc)
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
%attribute(pymoose::ClockTick, double, dt, __get_dt, __set_dt)
%attribute(pymoose::ClockTick, int, stage, __get_stage, __set_stage)
%attribute(pymoose::ClockTick, int, ordinal, __get_ordinal, __set_ordinal)
%attribute(pymoose::ClockTick, double, nextTime, __get_nextTime, __set_nextTime)
//%attribute(pymoose::ClockTick, string&, path, __get_path, __set_path) 
%attribute(pymoose::ClockTick, double, updateDtSrc, __get_updateDtSrc, __set_updateDtSrc)

%include "ClockJob.h"
%attribute(pymoose::ClockJob, double, runTime, __get_runTime, __set_runTime)
%attribute(pymoose::ClockJob, double, currentTime, __get_currentTime, __set_currentTime)
%attribute(pymoose::ClockJob, int, nsteps, __get_nsteps, __set_nsteps)
%attribute(pymoose::ClockJob, int, currentStep, __get_currentStep, __set_currentStep)
%attribute(pymoose::ClockJob, double, start, __get_start, __set_start)
%attribute(pymoose::ClockJob, int, step, __get_step, __set_step)
/* Numpy interface for InterpolationTable */
#ifdef NUMPY
%extend pymoose::InterpolationTable{
  %pythoncode {
  __array_struct__ = property(__array_struct__,
                                                 doc='Array protocol')
  }
} // end of extend
	
#endif
// WORK IN PROGRESS
/**/
%include "Interpol.h"
%attribute(pymoose::InterpolationTable, double, xmin, __get_xmin, __set_xmin)
%attribute(pymoose::InterpolationTable, double, xmax, __get_xmax, __set_xmax)
%attribute(pymoose::InterpolationTable, int, xdivs, __get_xdivs, __set_xdivs)
%attribute(pymoose::InterpolationTable, int, mode, __get_mode, __set_mode)
%attribute(pymoose::InterpolationTable, double, dx, __get_dx, __set_dx)
%attribute(pymoose::InterpolationTable, double, sy, __get_sy, __set_sy)
%attribute(pymoose::InterpolationTable, int, calcMode, __get_calcMode, __set_calcMode)

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
	pymoose::InterpolationTable.__setitem__ = InterpolationTable.__set_table
	pymoose::InterpolationTable.__getitem__ = InterpolationTable.__get_table
%}
*/
//%attribute(pymoose::InterpolationTable, string&, dumpFile, __get_print, __set_print) 
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
%attribute(pymoose::SynChan, double, Ik, __get_Ik, __set_Ik)
%attribute(pymoose::SynChan, unsigned int, numSynapses, __get_numSynapses)
%attribute(pymoose::SynChan, double, activation, __get_activation, __set_activation)
%attribute(pymoose::SynChan, double, modulator, __get_modulator, __set_modulator)

%include "BinSynchan.h"
%attribute(pymoose::BinSynchan, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(pymoose::BinSynchan, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::BinSynchan, double, tau1, __get_tau1, __set_tau1)
%attribute(pymoose::BinSynchan, double, tau2, __get_tau2, __set_tau2)
%attribute(pymoose::BinSynchan, bool, normalizeWeights, __get_normalizeWeights, __set_normalizeWeights)
%attribute(pymoose::BinSynchan, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::BinSynchan, double, Ik, __get_Ik, __set_Ik)
%attribute(pymoose::BinSynchan, unsigned int, numSynapses, __get_numSynapses)
%attribute(pymoose::BinSynchan, double, activation, __get_activation, __set_activation)
%attribute(pymoose::BinSynchan, double, modulator, __get_modulator, __set_modulator)

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
%attribute(pymoose::Nernst, double, E, __get_E, __set_E)
%attribute(pymoose::Nernst, double, Temperature, __get_Temperature, __set_Temperature)
%attribute(pymoose::Nernst, int, valence, __get_valence, __set_valence)
%attribute(pymoose::Nernst, double, Cin, __get_Cin, __set_Cin)
%attribute(pymoose::Nernst, double, Cout, __get_Cout, __set_Cout)
%attribute(pymoose::Nernst, double, scale, __get_scale, __set_scale)
%attribute(pymoose::Nernst, double, ESrc, __get_ESrc, __set_ESrc)
%attribute(pymoose::Nernst, double, CinMsg, __get_CinMsg, __set_CinMsg)
%attribute(pymoose::Nernst, double, CoutMsg, __get_CoutMsg, __set_CoutMsg)
%include "CaConc.h"
%attribute(pymoose::CaConc, double, Ca, __get_Ca, __set_Ca)
%attribute(pymoose::CaConc, double, CaBasal, __get_CaBasal, __set_CaBasal)
%attribute(pymoose::CaConc, double, Ca_base, __get_Ca_base, __set_Ca_base)
%attribute(pymoose::CaConc, double, tau, __get_tau, __set_tau)
%attribute(pymoose::CaConc, double, B, __get_B, __set_B)
%include "HHGate.h"
%attribute(pymoose::HHGate, InterpolationTable*, A, __get_A)
%attribute(pymoose::HHGate, InterpolationTable*, B, __get_B)
%include "HHChannel.h"
%attribute(pymoose::HHChannel, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(pymoose::HHChannel, double, Ek, __get_Ek, __set_Ek)
%attribute(pymoose::HHChannel, double, Xpower, __get_Xpower, __set_Xpower)
%attribute(pymoose::HHChannel, double, Ypower, __get_Ypower, __set_Ypower)
%attribute(pymoose::HHChannel, double, Zpower, __get_Zpower, __set_Zpower)
%attribute(pymoose::HHChannel, double, X, __get_X, __set_X)
%attribute(pymoose::HHChannel, double, Y, __get_Y, __set_Y)
%attribute(pymoose::HHChannel, double, Z, __get_Z, __set_Z)
%attribute(pymoose::HHChannel, int, instant, __get_instant, __set_instant)
%attribute(pymoose::HHChannel, double, Gk, __get_Gk, __set_Gk)
%attribute(pymoose::HHChannel, double, Ik, __get_Ik, __set_Ik)
%attribute(pymoose::HHChannel, int, useConcentration, __get_useConcentration, __set_useConcentration)
%attribute(pymoose::HHChannel, double, concen, __get_concen, __set_concen)

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
%attribute(pymoose::Stoich, unsigned int, nMols, __get_nMols, __set_nMols)
%attribute(pymoose::Stoich, unsigned int, nVarMols, __get_nVarMols, __set_nVarMols)
%attribute(pymoose::Stoich, unsigned int, nSumTot, __get_nSumTot, __set_nSumTot)
%attribute(pymoose::Stoich, unsigned int, nBuffered, __get_nBuffered, __set_nBuffered)
%attribute(pymoose::Stoich, unsigned int, nReacs, __get_nReacs, __set_nReacs)
%attribute(pymoose::Stoich, unsigned int, nEnz, __get_nEnz, __set_nEnz)
%attribute(pymoose::Stoich, unsigned int, nMMenz, __get_nMMenz, __set_nMMenz)
%attribute(pymoose::Stoich, unsigned int, nExternalRates, __get_nExternalRates, __set_nExternalRates)
%attribute(pymoose::Stoich, bool, useOneWayReacs, __get_useOneWayReacs, __set_useOneWayReacs)
//%attribute(pymoose::Stoich, string, path, __get_path, __set_path)
%attribute(pymoose::Stoich, unsigned int, rateVectorSize, __get_rateVectorSize, __set_rateVectorSize)
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
%attribute(pymoose::Enzyme, double, scaleKm, __get_scaleKm, __set_scaleKm)
%attribute(pymoose::Enzyme, double, scaleKcat, __get_scaleKcat, __set_scaleKcat)
%attribute(pymoose::Enzyme, double, intramol, __get_intramol, __set_intramol)
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
%attribute(pymoose::RandGenerator, double, sample, __get_sample, __set_sample)
%attribute(pymoose::RandGenerator, double, mean, __get_mean, __set_mean)
%attribute(pymoose::RandGenerator, double, variance, __get_variance, __set_variance)
%attribute(pymoose::RandGenerator, double, output, __get_output, __set_output)
%include "UniformRng.h"
%attribute(pymoose::UniformRng, double, mean, __get_mean, __set_mean)
%attribute(pymoose::UniformRng, double, variance, __get_variance, __set_variance)
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
// %include "Adaptor.h"
// %attribute(pymoose::Adaptor, double, inputOffset, __get_inputOffset, __set_inputOffset)
// %attribute(pymoose::Adaptor, double, outputOffset, __get_outputOffset, __set_outputOffset)
// %attribute(pymoose::Adaptor, double, scale, __get_scale, __set_scale)
// %attribute(pymoose::Adaptor, double, output, __get_output)
// %include "SigNeur.h"
// %attribute(pymoose::SigNeur, none, cell, __get_cell, __set_cell)
// %attribute(pymoose::SigNeur, none, spine, __get_spine, __set_spine)
// %attribute(pymoose::SigNeur, none, dend, __get_dend, __set_dend)
// %attribute(pymoose::SigNeur, none, soma, __get_soma, __set_soma)
// %attribute(pymoose::SigNeur, string, cellMethod, __get_cellMethod, __set_cellMethod)
// %attribute(pymoose::SigNeur, string, spineMethod, __get_spineMethod, __set_spineMethod)
// %attribute(pymoose::SigNeur, string, dendMethod, __get_dendMethod, __set_dendMethod)
// %attribute(pymoose::SigNeur, string, somaMethod, __get_somaMethod, __set_somaMethod)
// %attribute(pymoose::SigNeur, double, Dscale, __get_Dscale, __set_Dscale)
// %attribute(pymoose::SigNeur, int, parallelMode, __get_parallelMode, __set_parallelMode)
// %attribute(pymoose::SigNeur, double, updateStep, __get_updateStep, __set_updateStep)
// %attribute(pymoose::SigNeur, double, calciumScale, __get_calciumScale, __set_calciumScale)
