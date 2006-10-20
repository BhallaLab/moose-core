using namespace std;

#ifndef _HSolve_h
#define _HSolve_h
class HSolve
{
	friend class HSolveWrapper;
	public:
		HSolve();
		~HSolve();		

	private:
	int nPrimaryCacheValue;
	Element* pPrimaryCacheElement;
	int compartmentCount;   
	double ddt;             
	double dTime;
	ChCompartment* pCompartmentList;
	ChCompartment* compartmentList;
	SCompartment* pSCompartmentList;
	SCompartment* SCompartmentList; 
	int nRootCompartment;
	int nTotalNonZeroElmtInMatrix;	
	double* pdNewVoltage_list;	
	double* pdOldVoltage_list;	
	double** CoeffMatrix;	
	double* MatrixBackup;
	double* SumGk;		
	double* RHS_Fixed;	
	double* RHS_Var_p1;	
	double* LHS_Dia_p1;	
	double* dtByCm;		
	double *pdRaSum;
	double* Matrix;		
	int* TotElmInRow;	
	int* StartingPos;	
	int* DiagElmtPos;	
	list<int> lstEliminationPriorityList;	
	map<Element*, int> mapElementIndex;	
	Element** elementList; 
	Element** tmpElementList;
	bool isMemoryAllocated;                          
	bool isDataStructureSet; 
	bool isTimeStepSet;
	void allocate(); 
	void assignPriority();	
	int createSparseMatrix();
	void setSparseMatrix();	
	int doForwardElimination(int Row1, int Row2);
	int doBackwardElimination(int nPivotRow, int nRowToBeEliminated);
	int solve();
	void proc(double dt); 
	void clearDataStructures();
	void setDataStructure(double time_step);
	int updateRowforChangedRa(int nIndex);
	int crankNicolsonMethod();
	void setParentChildRelation(ChCompartment* compartmentList, int index, Element* element);
	void setTimeStep(double dTimeStep);
	void printVoltages();
};
#endif // _HSolve_h
