/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"

#include <ctime>
#include <cmath>
#include <queue>
#include <thread>

#include "../scheduling/Clock.h"
#include "../msg/DiagonalMsg.h"
#include "../basecode/SparseMatrix.h"
#include "../msg/SparseMsg.h"
#include "../mpi/PostMaster.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../shell/Shell.h"

unsigned int getNumCores()
{
    auto numCores = std::thread::hardware_concurrency();
    if (0 == numCores)
        numCores = 1;
    return numCores;
}

//////////////////////////////////////////////////////////////////

void checkChildren(Id parent, const string &info)
{
    vector<Id> ret;
    Neutral::children(parent.eref(), ret);
    cout << info << " checkChildren of " << parent.element()->getName() << ": "
         << ret.size() << " children\n";
    for (vector<Id>::iterator i = ret.begin(); i != ret.end(); ++i) {
        cout << i->element()->getName() << endl;
    }
}

Id init(int argc, char **argv, bool &doUnitTests)
{
    unsigned int numCores = getNumCores();
    int numNodes = 1;
    int myNode = 0;
    bool isInfinite = 0;
    int opt;
    Cinfo::rebuildOpIndex();

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myNode);
#endif

    Id shellId;
    Element *shelle = new GlobalDataElement(shellId, Shell::initCinfo(), "root", 1);

    Id clockId = Id::nextId();
    assert(clockId.value() == 1);
    Id classMasterId = Id::nextId();
    Id postMasterId = Id::nextId();

    Shell *s = reinterpret_cast<Shell *>(shellId.eref().data());
    s->setShellElement(shelle);
    s->setHardware(numCores, numNodes, myNode);
    s->loadBalance();

    /// Sets up the Elements that represent each class of Msg.
    unsigned int numMsg = Msg::initMsgManagers();

    new GlobalDataElement(clockId, Clock::initCinfo(), "clock", 1);
    new GlobalDataElement(classMasterId, Neutral::initCinfo(), "classes", 1);
    new GlobalDataElement(postMasterId, PostMaster::initCinfo(), "postmaster", 1);

    assert(shellId == Id());
    assert(clockId == Id(1));
    assert(classMasterId == Id(2));
    assert(postMasterId == Id(3));

    // s->connectMasterMsg();

    Shell::adopt(shellId, clockId, numMsg++);
    Shell::adopt(shellId, classMasterId, numMsg++);
    Shell::adopt(shellId, postMasterId, numMsg++);

    assert(numMsg == 10); // Must be the same on all nodes.

    Cinfo::makeCinfoElements(classMasterId);

    return shellId;
}
