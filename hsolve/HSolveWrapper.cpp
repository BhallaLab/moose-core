using namespace std;

#include "header.h"
#include <stdio.h>
#include <list>
#include <map>
#include <vector>
#include "../basecode/header.h"
#include "../basecode/Element.h"
#include "../basecode/Ftype.h"
#include "ChCompartment.h"
#include "HSolve.h"
#include "HSolveWrapper.h"


Finfo* HSolveWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< string >(
		"path", &HSolveWrapper::getPath, 
		&HSolveWrapper::setPath, "string" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"reinitIn", &HSolveWrapper::reinitFunc,
		&HSolveWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &HSolveWrapper::processFunc,
		&HSolveWrapper::getProcessConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &HSolveWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo HSolveWrapper::cinfo_(
	"HSolve",
	"",
	"HSolve: This class implements the HinesSolver",
	"Neutral",
	HSolveWrapper::fieldArray_,
	sizeof(HSolveWrapper::fieldArray_)/sizeof(Finfo *),
	&HSolveWrapper::create
);

///////////////////////////////////////////////////
// EvalField function definitions
///////////////////////////////////////////////////

string HSolveWrapper::localGetPath() const
{
			return path_;
}
void HSolveWrapper::localSetPath( string value ) {
			path_ = value;
			configure();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HSolveWrapper::processFuncLocal( ProcInfo info )
{
			proc(info->dt_);
                        //cout << dTime << "\t";
    			//for(int i = 0; i < compartmentCount; ++i)
    			//{
        		//	cout << pdNewVoltage_list[i] << "\t";
    			//}
    			//cout << endl;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnHSolveLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( HSolveWrapper, processConn_ );
	return reinterpret_cast< HSolveWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
void HSolveWrapper::configure()
{   
    vector< Element* > elist;
    vector< Element* >::iterator i;
    Element::startFind(path_, elist);
    compartmentCount = elist.size();
    allocate();
    int n = 0;
    for( i = elist.begin(); 
	 (i != elist.end()) && ( n < compartmentCount); ++i)
    {
	compartmentList[n].nPriorityNo = -1;
	compartmentList[n].nParent = -1;
	tmpElementList[n] = *i;
	mapElementIndex[*i] = n;
	++n;
    }
    n = 0;
    for( i = elist.begin();
	 (i != elist.end()) && ( n < compartmentCount); ++i)
    { 
	setParentChildRelation(compartmentList, n, *i); 
	++n;
    }
    nTotalNonZeroElmtInMatrix = 0;
    int nParent, nCount;
    for(n=0; n<compartmentCount; n++)
    {
	nCount = compartmentList[n].nNoOfChild;
	nParent = compartmentList[n].nParent;
	if(nParent < 0 )
	{
	    nRootCompartment = n;
	    nCount++;   
	}
	else
	{
	    nCount++;
	    nCount += compartmentList[nParent].nNoOfChild;
	}
	nTotalNonZeroElmtInMatrix += nCount;         
    }
    Matrix = new double [nTotalNonZeroElmtInMatrix];
    MatrixBackup = new double [nTotalNonZeroElmtInMatrix];    
    for( n = 0; n < nTotalNonZeroElmtInMatrix; ++n)
    {
        Matrix[n] = MatrixBackup[n] = 0.0;
    }
    assignPriority();
    Element* pelm = NULL;
    list<int>::iterator lstIterator = lstEliminationPriorityList.begin();
    n = 0;
    mapElementIndex.clear();
   //cout << "Time\t";
    while(lstIterator != lstEliminationPriorityList.end())
    {
	pelm = tmpElementList[*lstIterator++];
	assignCompartmentParameters(pCompartmentList[n], pelm);
	pCompartmentList[n].nParent = -1;
	pCompartmentList[n].nPriorityNo = -1;
	pdOldVoltage_list[n] = 
	    pdNewVoltage_list[n] = 
	    pCompartmentList[n].dVInitial;
	mapElementIndex[pelm] = n;
	elementList[n] = pelm;
	std::string name;
	Ftype1< string >::get((Element*)pelm, (string)"name", name); 
	//cout << name << "\t";
	n++;
    }
    //cout << endl;
    //cout << 0 << "\t";
    //for( n = 0; n < compartmentCount; ++n)
    //{
//	cout << pdNewVoltage_list[n] << "\t";
  //  }
  //  cout << endl;
    lstEliminationPriorityList.clear();
    for(n = 0; n <compartmentCount; ++n)
    {
	lstEliminationPriorityList.push_back(n);
	pelm = elementList[n];
	setParentChildRelation(pCompartmentList, n, pelm);
    }
    for(n = 0; n <compartmentCount; ++n)
    {
	nCount = pCompartmentList[n].nNoOfChild;
	nParent = pCompartmentList[n].nParent;
	if(nParent < 0 )
	{
	    nRootCompartment = n;
	    nCount++;   
	}
	else
	{
	    nCount++;
	    nCount += pCompartmentList[nParent].nNoOfChild;
	}
	TotElmInRow[n]  = nCount;          
    }
    delete []compartmentList;
    compartmentList = NULL;
    isMemoryAllocated = true;
    isDataStructureSet = false;
}
bool HSolveWrapper::assignCompartmentParameters(ChCompartment& compt, Element* e)
{
    compt.nPriorityNo = -1;
    Ftype1< double >::get( e, (string)"initVm", compt.dVInitial );
    Ftype1< double >::get( e, (string)"Ra", compt.dRa );
    Ftype1< double >::get( e, (string)"Rm", compt.dRm );
    Ftype1< double >::get( e, (string)"Cm", compt.dCm );
    Ftype1< double >::get( e, (string)"Inject", compt.dIinj );
    Ftype1< double >::get( e, (string)"Em", compt.dEm );
    compt.nParent = -1;
    return true;
}
