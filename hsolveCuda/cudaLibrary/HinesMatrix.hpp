/*
 * HinesMatrix.hpp
 *
 *  Created on: 05/06/2009
 *      Author: rcamargo
 */

//#include <iostream>
//#include <fstream>
#include <cstdio>
#include "ActiveChannels.hpp"
#include "SynapticChannels.hpp"
#include "Definitions.hpp"
//#include <fcntl.h>
using namespace std;

#ifndef HINESMATRIX_HPP_
#define HINESMATRIX_HPP_

#define PYRAMIDAL_CELL 0
#define INHIBITORY_CELL 1
#define BASKET_CELL 2

class HinesMatrix {

public:

	int neuron;
	int type;

	ftype *memory;      // Pointers to ftype neuron data
	ucomp *ucompMemory; // Pointers to ucomp neuron data

	ftype *rhsM;        // Right-hand side (B) -> A * x = B
	ftype *vmList;      // List of membrane potentials on each compartment
	ftype *vmTmp;       //

	ftype *leftList;       // contains the Hines Matrix in sparse form
	ucomp *leftListLine;   // contains the line of each matrix element
	ucomp *leftListColumn; // contains the column of each matrix element
	ucomp *leftStartPos;   // contains the start position for each compartment
	int leftListSize;      // contains the size of the left list

	/**
	 * Triangularized list
	 */
	ftype *triangList;     // contains the hines matrix after the triangularization

	/**
	 *  Used for triangSingle
	 **/
	ftype *mulList;     // contains the values to perform the update of the rhsM
	ucomp *mulListDest; // contains the line in the rhsM that will be updated
	ucomp *mulListComp; // contains the line used to update the rhsM
	int mulListSize;

	ftype *curr;

	ftype *Cm;
	ftype *Rm;
	ftype *Ra;

	ftype *active; // Current from active channels

	int **junctions;
	ftype *radius;

	ftype RM, CM, RA;

	ftype vRest;

	ftype dx;

	int nComp;

	int currStep;
	ftype dt;

	ftype posx, posy, posz;

	FILE *outFile;

	ActiveChannels *activeChannels; // Active channels

	SynapticChannels *synapticChannels; // Synaptic channels

	/**
	 * Generated spikes
	 */
	// Contains the time of the last spike generated on each neuron
	ftype lastSpike;
	// Contains the time of the spikes generated in the current execution block
	ftype *spikeTimes;
	int spikeTimeListSize;
	int nGeneratedSpikes;

	ftype threshold; // in mV
	ftype minSpikeInterval; // in mV

	int triangAll;

	HinesMatrix();
	~HinesMatrix();

	void redefineGenSpikeTimeList( ftype *targetSpikeTimeListAddress );

	int getnComp() { return nComp; }
	void setCurrent(int comp, ftype value) {
		curr[comp] = value;
	}

	/**
	 *  [ 1 | 2 | 3 ]
	 */
	void defineNeuronCable();

	void defineNeuronTreeN(int nComp, int active);

	void defineNeuronSingle();

	void defineNeuronCableSquid();

	void initializeFieldsSingle();
	/*
	 * Create a matrix for the neuron
	 */
	void createTestMatrix();

	/**
	 * Performs the upper triangularization of the matrix
	 */
	void upperTriangularizeSingle();

	/***************************************************************************
	 * This part is executed in every integration step
	 ***************************************************************************/

	void upperTriangularizeAll();

	//void findActiveCurrents();

	void updateRhs();

	void backSubstitute();

	void solveMatrix();

	/***************************************************************************
	 * This part is executed in every integration step
	 ***************************************************************************/

	void writeVmToFile(FILE *outFile);

	void printMatrix(ftype *list);

	void freeMem();
};


#endif /* HINESMATRIX_HPP_ */
