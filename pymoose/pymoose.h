#ifndef _PYMOOSE_H
#define _PYMOOSE_H
#include "PyMooseIterable.h"
#include "PyMooseContext.h"
#include "PyMooseUtil.h"
#include "PyMooseBase.h"
#include "Neutral.h"

#include "Class.h"

#include "HSolve.h"
#include "Cell.h"
#include "Compartment.h"
#include "HHGate.h"
#include "HHChannel.h"
#include "SpikeGen.h"
#include "SynChan.h"
#include "NMDAChan.h"
#include "BinSynchan.h"

#include "Interpol.h"
#include "Table.h"
#include "TableIterator.h"
#include "Tick.h"
#include "ClockJob.h"

#include "KineticHub.h"
#include "Kintegrator.h"
#include "MathFunc.h"
//#include "Mg_block.h"
#include "Neutral.h"

#include "RandGenerator.h"
#include "BinomialRng.h"
#include "ExponentialRng.h"
#include "GammaRng.h"
#include "NormalRng.h"
#include "PoissonRng.h"

#include "Enzyme.h"
#include "Reaction.h"
#include "Stoich.h"
#include "Molecule.h"
#include "Nernst.h"
#include "CaConc.h"

#include "Tick.h"
#ifdef USE_GL
#include "GLcell.h"
#include "GLview.h"
#endif
namespace pymoose{
extern PyMooseContext* context;
#if 0
// A set of convenience functions 
Id pwe(); // similar to pwd in unix
void ce(const string& path); // similar to cd
void ce(const Id& id); // similar to cd
void ce(const PyMooseBase& obj); // similar to cd
Id id(const string& path); // path2id
void loadg(const string& filepath); // load a genesis script
void rung(const string& statement); // run a genesis statement
vector<Id> children(Id id); // list of children of the object with Id id
vector<Id> children(const string& path); // list of children of the object with Id id
const string& getfield(const string& path, const string& field); 
const string& getfield(Id obj, const string& field);
void setfield(const string& path, const string& field, string value);
void setfield(const string& path, const string& field, double value);
void setfield(const string& path, const string& field, int value);
vector<string> messagelist(Id obj, const string& field, bool incoming);
void srandom(long seed); // seed the random number generator
void step(double runtime);
void step(long steps);
void step();
void stop();
void setclock(int clockNo, double dt, int stage=0);
void useclock(int clockNo, const string path, const string func="process");
bool exists(const string path);
#endif
}
#endif
