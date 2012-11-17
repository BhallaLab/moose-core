/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// This file has header dependencies for the Stoich class, as
// these are frequently used by other classes.

#include "header.h"
#include "RateTerm.h"
class MathFunc;
#include "../kinetics/FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"

class Stoich;
#include "../kinetics/PoolBase.h"
#include "../kinetics/lookupSizeFromMesh.h"
#include "../mesh/Stencil.h"
#include "Port.h"
#include "SolverJunction.h"
#include "Stoich.h"
#include "StoichCore.h"
