#ifndef _CHCOMPARTMENT_H
#define _CHCOMPARTMENT_H
typedef struct compartment_struct
{
	double dRa;	
	double dRm, dCm, dIinj; 
	double dEm;
	double dVInitial;
	int *pnChildList;
	int nNoOfChild;
	int nParent, nPriorityNo;
//	int nNoOfConnectedChannels;
//	int nChannelStartIndex;
	double dPrevVoltage;
	double dPrevVoltage_1;
	double dPrevVoltage_2;
//	element* pelement;
} ChCompartment;
struct SCompartment
{
	int     *pnSiblingParentList;
	int 	*pnConnectedList;
	int 	nParent;
	short 	nNoOfSiblingParent;
	short	nNoOfConnected;
};

struct SecondaryCache
{
	int 		nCacheValue1	;
	int 		nCacheValue2	;
	int 		nCacheValue3	;
	int 		nCacheValue4	;
	Element*	pCacheElement1	;
	Element*	pCacheElement2	;
	Element*	pCacheElement3	;
	Element*	pCacheElement4	;
};

#endif
