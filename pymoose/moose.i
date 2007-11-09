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
	#include "SpikeGen.h"
	#include "Nernst.h"
	#include "CaConc.h"
	#include "HHGate.h"
	#include "HHChannel.h"
	#include "Compartment.h"
	#include "HSolve.h"
	#include "Enzyme.h"
	#include "KineticHub.h"		
	#include "Kintegrator.h"
	#include "Molecule.h"
	#include "Reaction.h"
	#include "Stoich.h"
	#include "../kinetics/SparseMatrix.h"
	#include "../utility/utility.h"
	/* Random number related utilities */
	#include "../utility/randnum/randnum.h"
	/* These are the raw generic C++ classes - without any dependency on MOOSE */
	#include "../utility/randnum/Probability.h"
	#include "../utility/randnum/Binomial.h"
	#include "../utility/randnum/Gamma.h"
	#include "../utility/randnum/Normal.h"
	#include "../utility/randnum/Poisson.h"
	#include "../utility/randnum/Exponential.h"
	/* The following are moose classes */
	#include "RandGenerator.h"
	#include "BinomialRng.h"
	#include "GammaRng.h"
	#include "NormalRng.h"
	#include "PoissonRng.h"
	#include "ExponentialRng.h"
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
%ignore Id::operator();
%ignore operator<<;
%ignore operator>>;
%include "../basecode/Id.h"

%include "PyMooseContext.h"
%include "PyMooseBase.h"
%attribute(PyMooseBase, Id*, id, __get_id)
%attribute(PyMooseBase, Id*, parent, __get_parent)
%attribute(PyMooseBase, vector <Id>&, children, __get_children)
%attribute(PyMooseBase, vector <std::string>&, inMessages, __get_incoming_messages)
%attribute(PyMooseBase, vector <std::string>&, outMessages, __get_outgoing_messages)
//%attribute(PyMooseBase, string& , path, _path)
// The above gives segmentation fault, path is dynamically generated,
// so when using pointers, the memory may already have been deallocated
// better try writing to a string stream and returb stream.str()
//%ignore PyMooseBase::getPath;
//%attribute(PyMooseBase, string, path, getPath)
%include "Neutral.h"
%attribute(Neutral, int, childSrc, __get_childSrc, __set_childSrc)
%attribute(Neutral, int, child, __get_child, __set_child)

%include "Class.h"
%attribute(Class, std::string, name, __get_name, __set_name)
%attribute(Class, std::string, author, __get_author)
%attribute(Class, std::string, description, __get_description)
%attribute(Class, unsigned int, tick, __get_tick, __set_tick)
%attribute(Class, unsigned int, stage, __get_stage, __set_stage)

%include "Cell.h"
%include "Tick.h"
%attribute(ClockTick, double, dt, __get_dt, __set_dt)
%attribute(ClockTick, int, stage, __get_stage, __set_stage)
%attribute(ClockTick, int, ordinal, __get_ordinal, __set_ordinal)
%attribute(ClockTick, double, nextTime, __get_nextTime, __set_nextTime)
//%attribute(ClockTick, string&, path, __get_path, __set_path) 
%attribute(ClockTick, double, updateDtSrc, __get_updateDtSrc, __set_updateDtSrc)

%include "ClockJob.h"
%attribute(ClockJob, double, runTime, __get_runTime, __set_runTime)
%attribute(ClockJob, double, currentTime, __get_currentTime, __set_currentTime)
%attribute(ClockJob, int, nsteps, __get_nsteps, __set_nsteps)
%attribute(ClockJob, int, currentStep, __get_currentStep, __set_currentStep)
%attribute(ClockJob, double, start, __get_start, __set_start)
%attribute(ClockJob, int, step, __get_step, __set_step)

%include "Interpol.h"
%attribute(InterpolationTable, double, xmin, __get_xmin, __set_xmin)
%attribute(InterpolationTable, double, xmax, __get_xmax, __set_xmax)
%attribute(InterpolationTable, int, xdivs, __get_xdivs, __set_xdivs)
%attribute(InterpolationTable, int, mode, __get_mode, __set_mode)
%attribute(InterpolationTable, int, calc_mode, __get_calc_mode, __set_calc_mode)
%attribute(InterpolationTable, double, dx, __get_dx, __set_dx)
%attribute(InterpolationTable, double, sy, __get_sy, __set_sy)
%attribute(InterpolationTable, double, lookup, __get_lookup, __set_lookup)
%include "TableIterator.h"
%extend TableIterator
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
	InterpolationTable.__setitem__ = InterpolationTable.__set_table
	InterpolationTable.__getitem__ = InterpolationTable.__get_table
%}
*/
//%attribute(InterpolationTable, string&, dumpFile, __get_print, __set_print) 
%include "Table.h"
%attribute(Table, double, input, __get_input, __set_input)
%attribute(Table, double, output, __get_output, __set_output)
%attribute(Table, int, step_mode, __get_step_mode, __set_step_mode)
%attribute(Table, int, stepmode, __get_stepmode, __set_stepmode)
%attribute(Table, double, stepsize, __get_stepsize, __set_stepsize)
%attribute(Table, double, threshold, __get_threshold, __set_threshold)
//%attribute(Table, double, tableLookup, __get_tableLookup, __set_tableLookup)
%attribute(Table, double, outputSrc, __get_outputSrc, __set_outputSrc)
%attribute(Table, double, msgInput, __get_msgInput, __set_msgInput)
%attribute(Table, double, sum, __get_sum, __set_sum)
%attribute(Table, double, prd, __get_prd, __set_prd)
%include "SynChan.h"
%attribute(SynChan, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(SynChan, double, Ek, __get_Ek, __set_Ek)
%attribute(SynChan, double, tau1, __get_tau1, __set_tau1)
%attribute(SynChan, double, tau2, __get_tau2, __set_tau2)
%attribute(SynChan, bool, normalizeWeights, __get_normalizeWeights, __set_normalizeWeights)
%attribute(SynChan, double, Gk, __get_Gk, __set_Gk)
%attribute(SynChan, double, Ik, __get_Ik, __set_Ik)
%attribute(SynChan, unsigned int, numSynapses, __get_numSynapses, __set_numSynapses)
%attribute(SynChan, double, weight, __get_weight, __set_weight)
%attribute(SynChan, double, delay, __get_delay, __set_delay)
%attribute(SynChan, double, IkSrc, __get_IkSrc, __set_IkSrc)
%attribute(SynChan, double, synapse, __get_synapse, __set_synapse)
%attribute(SynChan, double, activation, __get_activation, __set_activation)
%attribute(SynChan, double, modulator, __get_modulator, __set_modulator)
%include "SpikeGen.h"
%attribute(SpikeGen, double, threshold, __get_threshold, __set_threshold)
%attribute(SpikeGen, double, refractT, __get_refractT, __set_refractT)
%attribute(SpikeGen, double, abs_refract, __get_abs_refract, __set_abs_refract)
%attribute(SpikeGen, double, amplitude, __get_amplitude, __set_amplitude)
%attribute(SpikeGen, double, state, __get_state, __set_state)
%attribute(SpikeGen, double, event, __get_event, __set_event)
%attribute(SpikeGen, double, Vm, __get_Vm, __set_Vm)
%include "Nernst.h"
%attribute(Nernst, double, E, __get_E, __set_E)
%attribute(Nernst, double, Temperature, __get_Temperature, __set_Temperature)
%attribute(Nernst, int, valence, __get_valence, __set_valence)
%attribute(Nernst, double, Cin, __get_Cin, __set_Cin)
%attribute(Nernst, double, Cout, __get_Cout, __set_Cout)
%attribute(Nernst, double, scale, __get_scale, __set_scale)
%attribute(Nernst, double, ESrc, __get_ESrc, __set_ESrc)
%attribute(Nernst, double, CinMsg, __get_CinMsg, __set_CinMsg)
%attribute(Nernst, double, CoutMsg, __get_CoutMsg, __set_CoutMsg)
%include "CaConc.h"
%attribute(CaConc, double, Ca, __get_Ca, __set_Ca)
%attribute(CaConc, double, CaBasal, __get_CaBasal, __set_CaBasal)
%attribute(CaConc, double, Ca_base, __get_Ca_base, __set_Ca_base)
%attribute(CaConc, double, tau, __get_tau, __set_tau)
%attribute(CaConc, double, B, __get_B, __set_B)
%attribute(CaConc, double, concSrc, __get_concSrc, __set_concSrc)
%attribute(CaConc, double, current, __get_current, __set_current)
%attribute(CaConc, double, increase, __get_increase, __set_increase)
%attribute(CaConc, double, decrease, __get_decrease, __set_decrease)
%attribute(CaConc, double, basalMsg, __get_basalMsg, __set_basalMsg)
%include "HHGate.h"
%include "HHChannel.h"
%attribute(HHChannel, double, Gbar, __get_Gbar, __set_Gbar)
%attribute(HHChannel, double, Ek, __get_Ek, __set_Ek)
%attribute(HHChannel, double, Xpower, __get_Xpower, __set_Xpower)
%attribute(HHChannel, double, Ypower, __get_Ypower, __set_Ypower)
%attribute(HHChannel, double, Zpower, __get_Zpower, __set_Zpower)
%attribute(HHChannel, int, instant, __get_instant, __set_instant)
%attribute(HHChannel, double, Gk, __get_Gk, __set_Gk)
%attribute(HHChannel, double, Ik, __get_Ik, __set_Ik)
%attribute(HHChannel, int, useConcentration, __get_useConcentration, __set_useConcentration)
%attribute(HHChannel, double, IkSrc, __get_IkSrc, __set_IkSrc)
%attribute(HHChannel, double, concen, __get_concen, __set_concen)
%include "Compartment.h"
%attribute(Compartment, double, Vm, __get_Vm, __set_Vm)
%attribute(Compartment, double, Cm, __get_Cm, __set_Cm)
%attribute(Compartment, double, Em, __get_Em, __set_Em)
%attribute(Compartment, double, Im, __get_Im, __set_Im)
%attribute(Compartment, double, inject, __get_inject, __set_inject)
%attribute(Compartment, double, initVm, __get_initVm, __set_initVm)
%attribute(Compartment, double, Rm, __get_Rm, __set_Rm)
%attribute(Compartment, double, Ra, __get_Ra, __set_Ra)
%attribute(Compartment, double, diameter, __get_diameter, __set_diameter)
%attribute(Compartment, double, length, __get_length, __set_length)
%attribute(Compartment, double, x, __get_x, __set_x)
%attribute(Compartment, double, y, __get_y, __set_y)
%attribute(Compartment, double, z, __get_z, __set_z)
%attribute(Compartment, double, VmSrc, __get_VmSrc, __set_VmSrc)
%attribute(Compartment, double, injectMsg, __get_injectMsg, __set_injectMsg)


%include "HSolve.h"
%attribute(HSolve, string, seedPath, __get_seed_path, __set_seed_path)
%attribute(HSolve, int, NDiv, __get_NDiv, __set_NDiv)
%attribute(HSolve, double, VLo, __get_VLo, __set_VLo)
%attribute(HSolve, double, VHi, __get_VHi, __set_VHi)

%include "Kintegrator.h"
%attribute(Kintegrator, bool, isInitiatilized, __get_isInitiatilized, __set_isInitiatilized)
//%attribute(Kintegrator, string, integrate_method, __get_method, __set_method)
//%attribute_ref(Kintegrator, string, method)
%include "Stoich.h"
%attribute(Stoich, unsigned int, nMols, __get_nMols, __set_nMols)
%attribute(Stoich, unsigned int, nVarMols, __get_nVarMols, __set_nVarMols)
%attribute(Stoich, unsigned int, nSumTot, __get_nSumTot, __set_nSumTot)
%attribute(Stoich, unsigned int, nBuffered, __get_nBuffered, __set_nBuffered)
%attribute(Stoich, unsigned int, nReacs, __get_nReacs, __set_nReacs)
%attribute(Stoich, unsigned int, nEnz, __get_nEnz, __set_nEnz)
%attribute(Stoich, unsigned int, nMMenz, __get_nMMenz, __set_nMMenz)
%attribute(Stoich, unsigned int, nExternalRates, __get_nExternalRates, __set_nExternalRates)
%attribute(Stoich, bool, useOneWayReacs, __get_useOneWayReacs, __set_useOneWayReacs)
//%attribute(Stoich, string, path, __get_path, __set_path)
%attribute(Stoich, unsigned int, rateVectorSize, __get_rateVectorSize, __set_rateVectorSize)
%include "KineticHub.h"
%attribute(KineticHub, unsigned int, nMol, __get_nMol, __set_nMol)
%attribute(KineticHub, unsigned int, nReac, __get_nReac, __set_nReac)
%attribute(KineticHub, unsigned int, nEnz, __get_nEnz, __set_nEnz)
%attribute(KineticHub, double, molSum, __get_molSum, __set_molSum)
%include "Enzyme.h"
%attribute(Enzyme, double, k1, __get_k1, __set_k1)
%attribute(Enzyme, double, k2, __get_k2, __set_k2)
%attribute(Enzyme, double, k3, __get_k3, __set_k3)
%attribute(Enzyme, double, Km, __get_Km, __set_Km)
%attribute(Enzyme, double, kcat, __get_kcat, __set_kcat)
%attribute(Enzyme, bool, mode, __get_mode, __set_mode)
//%attribute(Enzyme, double,double, prd, __get_prd, __set_prd)
%attribute(Enzyme, double, scaleKm, __get_scaleKm, __set_scaleKm)
%attribute(Enzyme, double, scaleKcat, __get_scaleKcat, __set_scaleKcat)
%attribute(Enzyme, double, intramol, __get_intramol, __set_intramol)
%include "Reaction.h"
%attribute(Reaction, double, kf, __get_kf, __set_kf)
%attribute(Reaction, double, kb, __get_kb, __set_kb)
%attribute(Reaction, double, scaleKf, __get_scaleKf, __set_scaleKf)
%attribute(Reaction, double, scaleKb, __get_scaleKb, __set_scaleKb)
%include "Molecule.h"
%attribute(Molecule, double, nInit, __get_nInit, __set_nInit)
%attribute(Molecule, double, volumeScale, __get_volumeScale, __set_volumeScale)
%attribute(Molecule, double, n, __get_n, __set_n)
%attribute(Molecule, int, mode, __get_mode, __set_mode)
%attribute(Molecule, int, slave_enable, __get_slave_enable, __set_slave_enable)
%attribute(Molecule, double, conc, __get_conc, __set_conc)
%attribute(Molecule, double, concInit, __get_concInit, __set_concInit)
%attribute(Molecule, double, nSrc, __get_nSrc, __set_nSrc)
//%attribute(Molecule, double,double, prd, __get_prd, __set_prd)
%attribute(Molecule, double, sumTotal, __get_sumTotal, __set_sumTotal)

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
%include "../utility/randnum/randnum.h"
/* These are the raw generic C++ classes - without any dependency on MOOSE */
%include "../utility/randnum/Binomial.h"
%include "../utility/randnum/Gamma.h"
%include "../utility/randnum/Normal.h"
%include "../utility/randnum/Poisson.h"
%include "../utility/randnum/Exponential.h"
/* The following are moose classes */

%include "RandGenerator.h"
%attribute(RandGenerator, double, sample, __get_sample, __set_sample)
%attribute(RandGenerator, double, mean, __get_mean, __set_mean)
%attribute(RandGenerator, double, variance, __get_variance, __set_variance)
%attribute(RandGenerator, double, output, __get_output, __set_output)
%include "GammaRng.h"
%attribute(GammaRng, double, alpha, __get_alpha, __set_alpha)
%attribute(GammaRng, double, theta, __get_theta, __set_theta)
%include "ExponentialRng.h"
%attribute(ExponentialRng, double, mean, __get_mean, __set_mean)
%attribute(ExponentialRng, int, method, __get_method, __set_method)
%include "BinomialRng.h"
%attribute(BinomialRng, int, n, __get_n, __set_n)
%attribute(BinomialRng, double, p, __get_p, __set_p)
%include "PoissonRng.h"
%attribute(PoissonRng, double, mean, __get_mean, __set_mean)
%include "NormalRng.h"
%attribute(NormalRng, double, mean, __get_mean, __set_mean)
%attribute(NormalRng, double, variance, __get_variance, __set_variance)
%attribute(NormalRng, int, method, __get_method, __set_method)
