#ifndef _HSOLVE_CPP
#define _HSOLVE_CPP

#include <vector>
#include <map>
#include <list>
using namespace std;
#include "../basecode/header.h"
#include "../basecode/Element.h"
#include "../basecode/Ftype.h"
#include "ChCompartment.h"
#include "HSolve.h"

HSolve::HSolve()
{      
        // cerr << "HSolve() " << endl;
    
    pdNewVoltage_list = NULL;
    pdOldVoltage_list = NULL;
    pCompartmentList = NULL;
    compartmentList = NULL;
    tmpElementList = NULL;
    elementList = NULL;
    Matrix = NULL;
    MatrixBackup = NULL;
    SumGk = NULL;
    RHS_Fixed = NULL;
    LHS_Dia_p1 = NULL;
    RHS_Var_p1 = NULL;
    dtByCm = NULL;
    TotElmInRow = NULL;
    StartingPos = NULL;
    DiagElmtPos = NULL;
    pPrimaryCacheElement = NULL;
    nPrimaryCacheValue = 0;
    nRootCompartment = 0;       
    compartmentCount = 0;
    ddt = 0;
    dTime = 0;
    nTotalNonZeroElmtInMatrix = 0;
    isDataStructureSet = false;
    isMemoryAllocated = false;
    isTimeStepSet = false;
    
//    cerr << "End HSolve()" << endl;
    
}
/**
   Deallocates all allocated memory for member variables
*/
HSolve::~HSolve()
{
    clearDataStructures();
}
/**
   Allocate memory for member variables
*/
void HSolve::allocate(){
//    cerr << "void HSolve::allocate()" << endl;
    
    if(compartmentCount <= 0 ){
        fprintf(stderr, "\n Error: Number of compartments should be positive in allocate()\n");
        return;
    }
    // Allocate memory to member varibales using size of list

    pCompartmentList    = new ChCompartment[compartmentCount];
    compartmentList     = new ChCompartment[compartmentCount];
    SCompartmentList = new SCompartment[compartmentCount];pdNewVoltage_list     = new double [compartmentCount] ;
    pdOldVoltage_list   = new double [compartmentCount] ;
    SumGk               = new double [compartmentCount] ;
    RHS_Fixed           = new double [compartmentCount] ;
    RHS_Var_p1          = new double [compartmentCount] ;
    dtByCm              = new double [compartmentCount] ;
    LHS_Dia_p1          = new double [compartmentCount] ;
    TotElmInRow         = new int [compartmentCount];       
    StartingPos         = new int [compartmentCount];       
    DiagElmtPos         = new int [compartmentCount];       
    tmpElementList      = new Element*[compartmentCount];
    elementList         = new Element*[compartmentCount];
    for(int i = 0; i < compartmentCount; ++i)
    {
        tmpElementList[i] = NULL;
    }
//     CoeffMatrix = new double* [compartmentCount];
//     for(int i = 0; i<compartmentCount; i++)
//     {
//      CoeffMatrix[i] = new double [compartmentCount];
//         cerr << "CoeffMatrix["<<i<<"] = " << CoeffMatrix[i] << endl;        
//     }

    //      Initialize CoeffMatrix with 0.
//     for(int i = 0; i<compartmentCount; i++)
//     {
//      for(int j = 0; j<compartmentCount; j++)
//      {
//          CoeffMatrix[i][j] = 0.0;
//      }
//     }
//    cerr << "void HSolve::allocate() - done" << endl;    
}


/**
   Function: setParentChildRelation
   Type    :  private
   Purpose :  This function sets parent child relation between compartments.
   Parameters:  
   nIndex   : Index of compartment.
                  pelement : Moose Element ponter of corresponding compartment.
   Logic    :  Using Moose element pointer get the child, parent and sibling
                                  relation and set them in compartment data structure.
   Steps -
                  1.   Get the total no of child for given element/compartment.
                  2.   Allocate that memory to pnChildList and set nNoOfChild.
                  3.   Iterate through the child list and get child element pointer
                           Look for element pointer in mapElementIndex to get index of that element.
                           store this index in pnChildList[x].
                  4.   For each child set nIndex as parent index.
*/
void HSolve::setParentChildRelation(ChCompartment* compartmentList, 
                                    int index, 
                                    Element* element)
{

    Field srcField = element->field("raxialIn");
    vector< Field > srclist;
    vector< Field >::iterator i;
    int childCount, childIndex;

    srcField.src( srclist );
    compartmentList[index].nNoOfChild = childCount = srclist.size();
    if(childCount == 0)
    {
        compartmentList[index].pnChildList = NULL;
    }
    else
    {
        compartmentList[index].pnChildList = new int[childCount];
    }
    int n = 0;
    for( i = srclist.begin(); i != srclist.end(); i++ ) 
    {
        map<Element* , int>::iterator mapIterator =
            mapElementIndex.find( i->getElement() );
        if ( mapIterator != mapElementIndex.end() ) 
        {
            childIndex = mapIterator->second;
            compartmentList[index].pnChildList[n] = childIndex;
            compartmentList[childIndex].nParent = index;
        }
        else 
        {
            cerr << "\nERROR: HSolveWrapper::setParentChildRelation()"
                 << "- Element no found in map!\n";
        }
        ++n;
    }
}
/**
   Assign Hines numbers to the compartments and organize them in a
   priority queue.  Modifies lstEliminationPriorityList
*/

void HSolve::assignPriority()
{
    // cerr << "void HSolve::assignPriority() " << endl;
    
    list<int> nodeList;
    lstEliminationPriorityList.clear();
    nodeList.push_back(nRootCompartment);

    int priority = compartmentCount - 1;
    int nodeNo, childNo;
    while(!nodeList.empty())
    {
        nodeNo = nodeList.front();
        nodeList.pop_front();
        // set priority of the compartment in front of the queue (if unassigned)
        // and put it in lstEliminationPriorityList
        if(compartmentList[nodeNo].nPriorityNo <0)
        {
            compartmentList[nodeNo].nPriorityNo = priority--;
            lstEliminationPriorityList.push_front(nodeNo);
        }
        // enqueue children connected at one end only (higher priority)
        for(int i = 0; i < compartmentList[nodeNo].nNoOfChild; ++i)
        {
            childNo = compartmentList[nodeNo].pnChildList[i];
            if(0 == compartmentList[childNo].nNoOfChild)
            {
                nodeList.push_back(childNo);
            }             
        }
        // enqueue children connected at both ends (lower priority)
        for( int i = 0; i < compartmentList[nodeNo].nNoOfChild; ++i)
        {
            childNo = compartmentList[nodeNo].pnChildList[i];
            if(0!=compartmentList[childNo].nNoOfChild)
            {
                nodeList.push_back(childNo);
            }
        }
    }
//    cerr << "void HSolve::assignPriority() - done" << endl;
    
}

/**
   Function Name  :  createSparseMatrix()
   Type           :  private .
   Purpose        :  This function creates initial matrix 
                     for Solving PDE using Crank Nicolson method.
   Parameters   :  None.
   Logic        :  Matrix creation includes converting 2D matrix 
                                     into Linear array also and then all matrix operations 
                                     are performned on linear array rather than 2D array 
                                     which is more efficient in space and time.
                                     Equation to solve is
                                     CV = Q
                                     C dv/dt = dq/dt = I 
                                     For mathematical details refer to HSolve.pdf
     Steps :
                                  1. Get channel information from Moose. - currently disabled
                                  2. Iterate through all (Parent, sibling, child)
                                     connected components  and calculate required term/sum.
                                  3. Copy all nonzero terms from 2D matrix to Linear Matrix. - no 2D matrix now                
                                  4. For Linear equation of the form AX = B
                                     create B which is pdOldVoltage_list in this function.


*/
/*
** The technique used for creating the sparse matrix:
** Let us assume - we have this structure:
**
**       s(2)        d1(1)            
**  ----------------------            the numbers in parenthesis indicating the sequence in priority list
**            \
**             \  d2(0)
**              \
**
** According to crank-nicolson method
**
** | 1 + t/2RC + gt/2C           -(1/2)(t/2RC)           -(1/2)(t/2RC)       |   |V0'|
** | -(1/2)(t/2RC)               1 + t/2RC + gt/2C       -(1/2)(t/2RC)       | x |V1'|  =
** | -(1/2)(t/2RC)               -(1/2)(t/2RC)           1 + t/2RC + gt/2C   |   |V2'|
**
**	| 1 - t/2RC - gt/2C          (1/2)(t/2RC)           (1/2)(t/2RC)       |   |V0|
**	| (1/2)(t/2RC)               1 - t/2RC - gt/2C      (1/2)(t/2RC)       | x |V1|
**	| (1/2)(t/2RC)               (1/2)(t/2RC)           1 - t/2RC - gt/2C  |   |V2|
**
** where t is the timestep, g is conductance, R is membrane resistance, C is membrane capacitance
** V0' V1' V2' new voltage (at time T+t) and V0 V1 V2 old voltage (at time T)
** R and g in each position must be the equivalents as calculated between connected compartment
** ( we have removed the subscripts for readability - otherwise R in row i column j is actually Rij
** - the equivalent resistance between compartment i and j)
**
** by multiplying each row on both sides by C/t, we have
** | C/t + 1/2R + g/2		-(1/2)(1/2R)		-(1/2)(1/2R)	 |   |V0'|
** | -(1/2)(1/2R)		C/t + 1/2R + g/2        -(1/2)(1/2R)	 | x |V1'| =
** | -(1/2)(1/2R)		-(1/2)(1/2R)		C/t + 1/2R + g/2 |   |V2'|
** 
**	| C/t - 1/2R - g/2		(1/2)(1/2R)		(1/2)(1/2R)	  |   |V0|
**	| (1/2)(1/2R)			C/t - 1/2R - g/2        (1/2)(1/2R)	  | x |V1|   
**	| (1/2)(1/2R)			(1/2)(1/2R)		C/t - 1/2R - g/2  |   |V2|
** 
*/
int HSolve::createSparseMatrix()
{
//    cerr << "int HSolve::createSparseMatrix()" << endl;
    int         nChild, nTotalChild, nSibling   ;
    int         nParent, i, j                                   ;
    double      temp, dConductancej = 0                 ;
    double      dCurrent= 0, dSumVj_Vm= 0, dRaParentSum =0      ;
    int         count = 0, count1 =0;
    list<int>::iterator lstIterator = lstEliminationPriorityList.begin();
    double *dTempMatrixRow = new double[compartmentCount];
    double *pMatrixBackup = new double [nTotalNonZeroElmtInMatrix]      ;       
    double *RaSum = new double [nTotalNonZeroElmtInMatrix - compartmentCount]   ;       
    int *pStartingPos   = new int        [compartmentCount]     ;       
    int *pDiagElmtPos   = new int        [compartmentCount]     ;       
    int nLinearMatrixCounter = 0, nRaCounter =0;
    for(i=0; i<compartmentCount; i++)
    {
        dTempMatrixRow[i] = 0;
    }
        //Start with first element of priority list for elemination and
        //continue till end of Prioritylist.
    for(i= 0; i<compartmentCount; i++)
    {
            //GetChannelInfo(i,pCompartmentList[i].pelement, &dSumGk, &dGk2Ek_Vm);
            //GetChannelInfo(i, &dSumGk, &dGk2Ek_Vm);
        nTotalChild     = pCompartmentList[i].nNoOfChild;
        nParent                 = pCompartmentList[i].nParent   ;
        dConductancej   = 0;
        dCurrent                = 0;
        dSumVj_Vm               = 0;
                
        if(nParent >=0)
        {
            SCompartmentList[i].nNoOfConnected = nTotalChild + pCompartmentList[nParent].nNoOfChild;
            SCompartmentList[i].pnConnectedList=  new int[SCompartmentList[i].nNoOfConnected]   ;
        }
        else
        {
            SCompartmentList[i].nNoOfConnected = nTotalChild;
            if ( nTotalChild >0)
                SCompartmentList[i].pnConnectedList=  new int[nTotalChild]      ;
            else
                SCompartmentList[i].pnConnectedList= NULL;
        }
                                
        SCompartmentList[i].nParent             = nParent;
        count = 0;
            //Connected  Childrens 
        for(j=0 ; j<nTotalChild; j++)
        {
            nChild = pCompartmentList[i].pnChildList[j];
            temp        = 1.0 /(pCompartmentList[i].dRa +  pCompartmentList[nChild].dRa)        ;
            RaSum[nRaCounter++]  = temp;
            dTempMatrixRow[nChild] = -temp; 
            dConductancej += temp;
            dSumVj_Vm += ((pdNewVoltage_list[nChild] - pdNewVoltage_list[i]) * temp);
            SCompartmentList[i].pnConnectedList[count++] = nChild;
        }

        if (nParent >=0)
        {
            temp = 1.0 / (pCompartmentList[i].dRa +  pCompartmentList[nParent].dRa)     ;
            dRaParentSum = temp;
            dTempMatrixRow[nParent]  =  -temp;
            dConductancej +=  temp;
            dSumVj_Vm += ((pdNewVoltage_list[nParent] - pdNewVoltage_list[i]) * temp)  ;
            int siblingCount = pCompartmentList[nParent].nNoOfChild;
            SCompartmentList[i].nNoOfSiblingParent  = siblingCount;
            SCompartmentList[i].pnSiblingParentList = new int[siblingCount];
//                      SCompartmentList[i].pdRaSPSum                   = new double[ pCompartmentList[nParent].nNoOfChild];
            count1 =0;
            for(j=0; j< siblingCount; j++)
            {
                nSibling = pCompartmentList[nParent].pnChildList[j];
                if (nSibling != i)
                {
                    temp        = 1.0 / (pCompartmentList[i].dRa +  pCompartmentList[nSibling].dRa)     ;
                    RaSum[nRaCounter++] = temp;
                    dTempMatrixRow[nSibling] = -temp; 
                    dConductancej += temp;
                    dSumVj_Vm += ((pdNewVoltage_list[nSibling] - pdNewVoltage_list[i]) * temp);                         
                    SCompartmentList[i].pnSiblingParentList[count1++] = nSibling;
                    SCompartmentList[i].pnConnectedList[count++]   = nSibling;
                }
            }
            RaSum[nRaCounter++] = dRaParentSum;
            SCompartmentList[i].pnSiblingParentList[count1] = nParent;
            SCompartmentList[i].pnConnectedList[count]   = nParent;
        }
        else
        {
            SCompartmentList[i].pnSiblingParentList = NULL;
            SCompartmentList[i].nNoOfSiblingParent      = 0;
        }
            // now  dConductancej conatains summantion of 1/(Rai + Raj) where j are the branch connected to i.
                
        LHS_Dia_p1[i] = (pCompartmentList[i].dCm/ddt) + (dConductancej + (0.5 / pCompartmentList[i].dRm));   
                
            //Set Diagonal element
        dTempMatrixRow[i] = LHS_Dia_p1[i];//+(pChannelDataList[i].dGk);
        pdOldVoltage_list[i] = RHS_Fixed[i] + (RHS_Var_p1[i] * pdNewVoltage_list[i]) + dSumVj_Vm; //+ (pChannelDataList[i].dEk);
        pStartingPos[i] = nLinearMatrixCounter;
        for(j=0; j<compartmentCount; j++)
        {
            if(dTempMatrixRow[j] != 0)
            {
                Matrix[nLinearMatrixCounter] = dTempMatrixRow[j];
                pMatrixBackup[nLinearMatrixCounter] = dTempMatrixRow[j];
                if(i == j)
                    pDiagElmtPos[i]     =       nLinearMatrixCounter;
                nLinearMatrixCounter++;
                dTempMatrixRow[j]       = 0;                                    
            }
        }               
    }
        
    if(nLinearMatrixCounter != nTotalNonZeroElmtInMatrix)
    {
        cerr << "\n ERROR : Problem in conversion of 2d Matrix to Linear MAtrix" << endl;
        exit(1);
    }
    
        
    if(nRaCounter != (nTotalNonZeroElmtInMatrix - compartmentCount))
    {
        cerr << "\n ERROR : Problem in RaSum variable array" << endl;
        exit(1);
    }
    
    delete      []dTempMatrixRow;
    dTempMatrixRow              = NULL;
    MatrixBackup                = pMatrixBackup;
    pMatrixBackup               = NULL;
    StartingPos                 = pStartingPos  ;
    pStartingPos                = NULL;
    DiagElmtPos                 = pDiagElmtPos  ;
    pDiagElmtPos                = NULL;
    pSCompartmentList = SCompartmentList;
    SCompartmentList    = NULL;
    pdRaSum                             = RaSum;
    RaSum                               = NULL;
    //Update Solver state variable
    isDataStructureSet = true;
//    cerr << "int HSolve::createSparseMatrix() - done" << endl;
    
    return 0;
}

/**
     Function Name:  setSparseMatrix()
     Type         :  private .
     Purpose              :  In HSolve class there are two crital function
     Logic                        1. CreateSparseMatrix
     2. SetSparseMatrix
     To solve equation of the form AX = B
     creation of A is done in Create Sparse matrix. After each
     integration step A gets disturbed so to do next integration 
     step creation of complete matrix is not required.
     Only those terms which are affected should be changed and
     rest all should be copied from previous calculated value.                                    
     So in this function values and copied from MatrixBackup to
     Matrix. and also new values are calculated.
     for details about each variable refer to HSolve.pdf
     Parameters   :  None.

*/

void HSolve::setSparseMatrix()
{
//    pChannelDataList = pChanSolver->SolveAllChannels(pdNewVoltage_list);
    int nConnected,nTotalConnected;
    int j;
    double  dSumVj_Vm;
    int nMatrixIndex =0, nRaCounter = 0;
    for(int i= 0; i<compartmentCount; i++)
    {
        nTotalConnected = pSCompartmentList[i].nNoOfConnected;
        dSumVj_Vm = 0;

            //Connected  Childrens
        for(j=0; j<nTotalConnected; j++)
        {
            nConnected = pSCompartmentList[i].pnConnectedList[j];
            dSumVj_Vm += ((pdNewVoltage_list[nConnected] - pdNewVoltage_list[i]) * pdRaSum[nRaCounter++] )  ;
        }
                                                
            //Set Diagonal element
        Matrix[nMatrixIndex++] = LHS_Dia_p1[i];// + (pChannelDataList[i].dGk);
        pdOldVoltage_list[i] = RHS_Fixed[i] + (RHS_Var_p1[i] * pdNewVoltage_list[i]) + dSumVj_Vm;//+ (pChannelDataList[i].dEk);                 
       
        if(i < (compartmentCount-1))
        {
            for(;nMatrixIndex < DiagElmtPos[i+1]; nMatrixIndex++)
            {
                Matrix[nMatrixIndex]    =       MatrixBackup[nMatrixIndex];
            }
        }
    }
}



  /**
     Function Name  :  solve()
     Type           :  private (Helper function for Backward_Euler_Method).
     Purpose        :  This function solves Coeff matrix which is already 
                                           ordered in priority.
     Parameters    :  None.
     Logic                 :  Start with least priority i.e first row in matrix 
                                           and then eleminate other offtridiagonal elements. 
                               using information about child nodes.
     Hines Algorithm --
     Start with root and number all branches (in decrementing order) using BFS. 
     (Breadth first search)
     In this method start with lower no and eleiminate row in increasing order.
     Once you have reached root then traverse back exactly in same manner.
     In this way row elimination will not conver zero element into non zero element.              
                  |5
                  |
                  |
                 /|\
                / | \
              4/ 2|  \3
                 /                                            
                /
               /1
              |
              |
              |0           

  */
int HSolve::solve()
{
    for(int nNode =0; nNode<compartmentCount; nNode++)
    {
        for(int i =0; i<pSCompartmentList[nNode].nNoOfSiblingParent; i++)
        {
            int nRowToBeEliminated = pSCompartmentList[nNode].pnSiblingParentList[i];
            int nElmIndex = StartingPos[nRowToBeEliminated];
            int nOpStartIndex = nElmIndex;
            int nCondition = nElmIndex + TotElmInRow[nRowToBeEliminated];
            
            for(int j=nElmIndex; j < nCondition; j++)
            {
                if((Matrix[j] != 0) && (j != DiagElmtPos[nRowToBeEliminated]))
                {
                    nOpStartIndex = j;
                    break;
                }
            }
            int nPivotIndex = DiagElmtPos[nNode];
            //printf("nPivotRow = %d, RowElm = %d", nPivotRow,nRowToBeEliminated);
//Rajnish//            pdOldVoltage_list[nRowToBeEliminated] = ((pdOldVoltage_list[nRowToBeEliminated] * Matrix[nPivotIndex])/Matrix[nOpStartIndex]) - pdOldVoltage_list[nNode];
/*//Subhasis//*/            pdOldVoltage_list[nRowToBeEliminated] = pdOldVoltage_list[nRowToBeEliminated]-(pdOldVoltage_list[nNode] *Matrix[nOpStartIndex])/ Matrix[nPivotIndex];
            int iend = (StartingPos[nNode] + TotElmInRow[nNode]);
            int jend = (StartingPos[nRowToBeEliminated] + TotElmInRow[nRowToBeEliminated]);
            for(int j=nOpStartIndex+1, k=nPivotIndex+1; ((k<iend) && (j<jend)); k++, j++)
            {
//Rajnish//                  Matrix[j] = ((Matrix[j] * Matrix[nPivotIndex])/Matrix[nOpStartIndex]) -  Matrix[k] ;
/*//Subhasis//*/	          Matrix[j] = Matrix[j] - ((Matrix[k] * Matrix[nOpStartIndex]) / Matrix[nPivotIndex]);
            }
            Matrix[nOpStartIndex]=0.0;  
        }
    }
    pdNewVoltage_list[compartmentCount-1]= pdOldVoltage_list[compartmentCount-1]/Matrix[DiagElmtPos[compartmentCount-1]];

    if(compartmentCount > 1){
        for(int i = compartmentCount-2; i>=0; i--) 
        {
            /* What does this do?
               StartingPos[i+1]  is the starting position of the next row
               StartingPos[i+1] - 1 is the end position of the current row
               Matrix[StartingPos[i+1]-1] is the last element in this row
               new voltage = (old voltage - last element in row * new voltage of parent )/ diagonal element
               this is done from last but one row sweeping to first row
            */
            pdNewVoltage_list[i]= (pdOldVoltage_list[i] - (Matrix[StartingPos[i+1]-1]* pdNewVoltage_list[pSCompartmentList[i].nParent]))/Matrix[DiagElmtPos[i]];
        }       
    }
    return 0;
}


  /*=====================================================================
    Function Name  :  crankNicolsonMethod()
    Type         :  public.
    Purpose        :  This function calculates voltages for given compartment
    using Crank-Nicolson method.
    used for debugging only.
    Parameters     :  None.
    Logic        :  Step 1: Solving Tridiagonal matrix Calculate new Voltages.
    Step 2: store new values as old values.
    Step 3: Initialize Tridiagonal matrix
    step 4: Repeat step 1 to 3 for required no of TimeIteration.
    ========================================================================*/
int HSolve::crankNicolsonMethod()
{
    // cerr << "int HSolve::crankNicolsonMethod()" << endl;
    
    int i;
    FILE *fp = fopen("graph.txt","a+");

    double dTime=0, dstep = 0;  

    printf("%f\t%f\n",dTime, pdNewVoltage_list[0]);
    dstep               = 30.0e-06;
    int nTimeIteration = 5000;
    for(i =0; i<nTimeIteration; i++)
    {       
        proc(dstep);
        printf("%f\t%f\n",dTime, pdNewVoltage_list[0]);
        dTime += ddt;       
    }
    fclose(fp);
    printf("\n Run over \n");
    // cerr << "int HSolve::crankNicolsonMethod()" << endl;
    return 0;
}

/**
    Function Name  :  setTimeStep()
    Type         :  public.
    Purpose        :  This function updates datastructure related to timestep
    in two conditions
    1. Initial condition (ddt ==0)
    In this case solver is getting initialized so datastructure needs to be updated.
    2. given Time step is different than earlier time steps.
    Parameters     :  dTimeStep.
    Logic        :  Update datastructure only when timestep is changed or during initialization.
    */

void HSolve::setTimeStep(double dTimeStep)
{
    // cerr << "void HSolve::setTimeStep(double dTimeStep)" << endl;
    if((dTimeStep!=ddt) && (dTimeStep > 0))
    {
        ddt = dTimeStep;
        for(int i=0; i<compartmentCount; i++)
        {
            RHS_Fixed[i] = (pCompartmentList[i].dIinj + (pCompartmentList[i].dEm/pCompartmentList[i].dRm));
            RHS_Var_p1[i] = (pCompartmentList[i].dCm / ddt) - (0.5/pCompartmentList[i].dRm);
        }
    } 
    // cerr << "void HSolve::setTimeStep(double dTimeStep) - done .. ddt =" << ddt <<  endl;
}


    /**
     Function Name        :  proc()
     Type                 :  public
     Purpose                      :  This function does just one integration step with given Timestep.
     Parameters           :  dt : Timestep for current integration step. 
     Logic                        :  Self Explanatory.
    */
void HSolve::proc(double dt)                              
{
    // cerr << " proc(double dt)" << endl;
    
    dTime       =       dTime + dt;
//    setTimeStep(dt);
    setDataStructure(dt);
    solve();    
    // cerr << " proc(double dt) - done" << endl;
}




/**
    Function Name  :  setDataStructure()
    Type         :  public.
    Purpose        :  This function 
    either
    Creates all required data structure if solver is in initialization state
    or in deleted state.
    OR
    changes only those positions which need to be changed if solver
    is already initialized/Allocated datastructures.                                                
*/
void HSolve::setDataStructure(double dt)
{
    // cerr << "setDataStructure()" << endl;
    
    if(isMemoryAllocated && isDataStructureSet) // this means all datastructure already existing
    {
        setSparseMatrix();               // and we need to just set the values.         
    }
    else
    {
        setTimeStep(dt);
        createSparseMatrix();    // In this case create and calculate all values
    }

    // cerr << "setDataStructure() - done" << endl;
}


/**
     Function Name        :  GetChannelInfo()
     Type                 :  private
     Purpose                      :  To get channel information for given compartment from Moose.
     Parameters           :  nIndex       :       Index of compartment whose channel information is required.
     pelement     :       corresponding Moose Element pointer.
     pdSumGk      :       This is returned as out parameter. and contains
     sum of all channel conductances.
     pdSumGk2Ek_Vm: This is returned as out parameter, and contains
     sum of Gk * (2Ek - Vm).
     for details refer to HSolve.pdf.
     Logic                        :  USing Moose Element pointer get all channel pointers.
     using channel poniters get channel conductance and channel
     reversal potential (Ek).
     calculate sum of all conductances to get pdSumGk.
     calculate sum of Gk * (2Ek - Vm) to get pdSumGk2Ek_Vm.
     return these two variable as out parameter.
*/
/*
int HSolve::GetChannelInfo(int nIndex, double *pdSumGk, double *pdSumGk2Ek_Vm)
{
    *pdSumGk            =       0;
    *pdSumGk2Ek_Vm      =       0;
    int nChannelIndex = pCompartmentList[nIndex].nChannelStartIndex;
    for(int i=0; i<pCompartmentList[nIndex].nNoOfConnectedChannels; i++)
    {
        *pdSumGk += pChannelDataList[nChannelIndex].dGk;
        *pdSumGk2Ek_Vm  += (pChannelDataList[nChannelIndex].dGk * ((2.0 * pChannelDataList[nChannelIndex].dEk) - pdNewVoltage_list[nIndex]));
        nChannelIndex++;
    }
    return 0;
}
*/
  /**
     Cleans up all datastructure, used in destructor and reinit()
   */
void HSolve::clearDataStructures()
{
    // cerr << "void HSolve::clearDataStructures()" << endl;
    
    if(pCompartmentList)
    {
        for( int i = 0; i < compartmentCount; ++i)
        {
            delete []pCompartmentList[i].pnChildList;
        }
        delete []pCompartmentList;
    }

    if(compartmentList)
    {
        for( int i = 0; i < compartmentCount; ++i)
        {
            delete []compartmentList[i].pnChildList;
        }
        delete []compartmentList;
    }
    if(CoeffMatrix)
    {
        for(int i = 0; i < compartmentCount; ++i)
        {
            delete []CoeffMatrix[i];
        }
        delete []CoeffMatrix;
    }
    if(pSCompartmentList)
    {
        for(int i = 0; i<compartmentCount; i++)
        {
            delete []pSCompartmentList[i].pnConnectedList;
            delete []pSCompartmentList[i].pnSiblingParentList;
        }
        delete []pSCompartmentList;
        pSCompartmentList = NULL;
    }
    if(SCompartmentList){
        delete []SCompartmentList;
    }
    
    if (tmpElementList != NULL)
        delete []tmpElementList;
    if(elementList != NULL)
        delete []elementList;
    if(pdNewVoltage_list)
        delete []pdNewVoltage_list;
    if(pdOldVoltage_list)
        delete []pdOldVoltage_list;
    if(Matrix)
        delete []Matrix;
    if(MatrixBackup)
        delete []MatrixBackup;
    if(SumGk)
        delete []SumGk;
    if(RHS_Fixed)
        delete []RHS_Fixed;
    if(LHS_Dia_p1)
        delete []LHS_Dia_p1;
    if(RHS_Var_p1)
        delete []RHS_Var_p1;
    if(dtByCm)
        delete []dtByCm;
    if(TotElmInRow)
        delete []TotElmInRow;
    if(StartingPos)
        delete []StartingPos;
    if(DiagElmtPos)
        delete []DiagElmtPos;
    lstEliminationPriorityList.clear();
    mapElementIndex.clear();
    isMemoryAllocated = false; 
    isDataStructureSet = false;

    compartmentCount = 0;
    ddt = 0;
    pCompartmentList = NULL;
    compartmentList = NULL;
    tmpElementList = NULL;
    nRootCompartment = 0;
    nTotalNonZeroElmtInMatrix= 0;
    dTime = 0;
    pdNewVoltage_list = NULL;
    pdOldVoltage_list = NULL;
    CoeffMatrix = NULL;
    Matrix = NULL;
    MatrixBackup = NULL;
    SumGk = NULL;
    RHS_Fixed = NULL;
    RHS_Var_p1 = NULL;
    LHS_Dia_p1 = NULL;
    dtByCm = NULL;
    TotElmInRow = NULL;
    StartingPos = NULL;
    DiagElmtPos = NULL;
    // cerr << "void HSolve::clearDataStructures() - done" << endl;
}

// for debugging only
void HSolve::printVoltages()
{
    for(int i = 0; i < compartmentCount; ++i)
    {
        cerr << "new = "<<pdNewVoltage_list[i] << "    old = " << pdOldVoltage_list[i] << endl;
    }
    cerr << "Matrix entries:\n";
    
    for(int i = 0; i < nTotalNonZeroElmtInMatrix; ++i)
    {
        cerr  << Matrix[i] << "\t";
    }
    cerr << endl;
    
}
#endif
