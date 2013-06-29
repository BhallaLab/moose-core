#include "HinesMatrix.hpp"
#include <cassert>
#include <cmath>

HinesMatrix::HinesMatrix() {
	CM = 1; // uF/cm2
	RA = 0.1; // kOhm*cm
	RM = 10; // kOhm*cm2

	dt = 0.1; // ms 0.1
	vRest = 0; // mV
	dx = 50e-4; // cm
	currStep = -1;

	outFile = 0;
	activeChannels = 0;
	triangAll = 0;

	lastSpike = -1000;
	spikeTimeListSize = GENSPIKETIMELIST_SIZE;
	spikeTimes = new ftype[spikeTimeListSize];
	nGeneratedSpikes = 0;
	threshold = 50;
	minSpikeInterval = 5;

	posx=0; posy=0; posz=0;
}

void HinesMatrix::freeMem() {

	delete[] memory;
	delete[] ucompMemory;
	for (int i=0; i<nComp; i++)
		delete[] junctions[i];
	delete[] junctions;
	delete[] radius;

	if (outFile != 0) fclose (outFile);

}

HinesMatrix::~HinesMatrix() {

	freeMem();
}

void HinesMatrix::redefineGenSpikeTimeList( ftype *targetSpikeTimeListAddress ) {
	delete []spikeTimes;
	spikeTimes = targetSpikeTimeListAddress;
}

/**
 *  [ 1 | 2 | 3 ]
 */
void HinesMatrix::defineNeuronCable() {
	nComp = 5;
	junctions = new int *[nComp];
	for (int i=0; i<nComp; i++) {
		junctions[i] = new int[nComp];
		for (int j=0; j<nComp; j++)
			junctions[i][j] = 0;
	}
	junctions[0][1] = 1;
	for (int i=1; i<nComp-1; i++) {
		junctions[i][i-1] = 1;
		junctions[i][i+1] = 1;
	}
	junctions[nComp-1][nComp-2] = 1;

	radius = new ftype[nComp];
	for (int i=0; i<nComp-1; i++)
		radius[i] = 0.5e-4; // cm
	radius[nComp-1] = 25e-4; // Soma

	initializeFieldsSingle();
}

void HinesMatrix::defineNeuronSingle() {
	nComp = 1;
	junctions = new int *[nComp];
	for (int i=0; i<nComp; i++) {
		junctions[i] = new int[nComp];
	}

	junctions[0][0]=0;
	radius = new ftype[1];
	radius[0] = 25e-4; // Soma

	initializeFieldsSingle();
}


void HinesMatrix::defineNeuronCableSquid() {
	nComp = 1;
	junctions = new int *[nComp];
	for (int i=0; i<nComp; i++) {
		junctions[i] = new int[nComp];
		for (int j=0; j<nComp; j++)
			junctions[i][j] = 0;
	}

	//juctions[0][1] = 1;
	//juctions[1][0] = 1;

	//juctions[1][2] = 1;
	//juctions[2][1] = 1;

	radius = new ftype[nComp];
	radius[0] = 250e-4;
	//rad[1] = 250e-4;
	//rad[2] = 250e-4;

	dx = 500e-4; // cm
	CM = 1; // uF/cm2
	RA = 0.030; // kOhm*cm
	RM = 1.0/0.3; // kOhm*cm2

	initializeFieldsSingle();

	/**
	 * Create active channels
	 */
	int nActivecomp = 1;
	ucomp *activeCompList = new ucomp[nActivecomp];
	activeCompList[0] = nComp-1;
	activeChannels = new ActiveChannels (dt, vmList, nComp);
	activeChannels->setActiveChannels (nActivecomp, activeCompList);
}

/**
 * 1-3, 2-6, 3-12, 4-24, 5-48
 */
void HinesMatrix::defineNeuronTreeN(int nComp, int active) {

	this->nComp = nComp;

	// All neurons will approximately the same size
	this->dx *= (4.0/nComp);

	junctions = new int *[nComp];
	for (int i=0; i<nComp; i++) {
		junctions[i] = new int[nComp];
		for (int j=0; j<nComp; j++)
			junctions[i][j] = 0;
	}

	/**
	 *  0   1
	 *    2
	 *    3
	 */
	if (nComp % 4 == 0) {
		for (int k=0; k<nComp; k+=4) {
			junctions[k+0][k+2] = 1;
			junctions[k+1][k+2] = 1;
			junctions[k+2][k+0] = 1;
			junctions[k+2][k+1] = 1;
			junctions[k+2][k+3] = 1;
			junctions[k+3][k+2] = 1;
		}

		// Soma is already connected
		for (int x=1, k = nComp-1-(4*x); x<4096*4096 && x*4<nComp; x *= 2) {

			int next = k + 1;
			for (int i = 0; i < 2*x && k >= 0; k-=4, i++) {
				junctions[k][next] = 1;
				junctions[next][k] = 1;

				if (next % 2 == 0) next++;
				else next += 3;
			}
		}
	}
	else if (nComp < 4) {
		junctions[0][1] = 1;
		for (int i=1; i<nComp-1; i++) {
			junctions[i][i-1] = 1;
			junctions[i][i+1] = 1;
		}
		junctions[nComp-1][nComp-2] = 1;
	}

	radius = new ftype[nComp];
	for (int i=0; i<nComp-1; i++)
		radius[i] = 5e-4; // cm
	radius[nComp-1] = 20e-4; // Soma

	// This is to equilibrate the fire rate with the neuron with nComp=4
	if (nComp == 8) radius[nComp-1] = 40e-4;
	if (nComp == 12) radius[nComp-1] = 40e-4;
	if (nComp == 16) radius[nComp-1] = 40e-4;

	initializeFieldsSingle();

	int nActivecomp = 1;

	if (nActivecomp == 1) triangAll = 0;
	else triangAll = 1;

	assert (nActivecomp < nComp);
	ucomp *activeCompList = new ucomp[nActivecomp];
	for (int i=0; i<nActivecomp; i++)
		activeCompList[i] = (ucomp)( (nComp-1)-i );

	activeChannels = new ActiveChannels (dt, vmList, nComp);

	int nChannels    = 2 * nActivecomp;
	ucomp *nGates    = new ucomp[nChannels];
	ucomp *compList  = new ucomp[nChannels];
	ftype *channelEk = new ftype[nChannels];
	ftype *gBar      = new ftype[nChannels];

	ftype *eLeak     = new ftype[nActivecomp];

	for (int activeComp=0; activeComp<nActivecomp; activeComp++) {

		// Na channel
		nGates[2*activeComp]    = 2;
		compList[2*activeComp]  = activeCompList[activeComp];
		channelEk[2*activeComp] = 115.0009526;  // obtained from Squid.g
		gBar[2*activeComp]      = 120 * (2*PI*radius[ activeCompList[activeComp] ]*dx);

		// K channel
		nGates[2*activeComp+1]    = 1;
		compList[2*activeComp+1]  = activeCompList[activeComp];
		channelEk[2*activeComp+1] = -11.99979277; // obtained from Squid.g
		gBar[2*activeComp+1]      =  36 * (2*PI*radius[ activeCompList[activeComp] ]*dx);

		// Eleak for Na and K from compartment zero
		eLeak[activeComp]         =  10.613;      // obtained from Squid.g
	}

	activeChannels->createChannelList (nChannels, nGates, compList, channelEk, gBar, eLeak, nActivecomp, activeCompList);
	delete[] activeCompList;

	for (int activeComp=0; activeComp<nActivecomp; activeComp++) {

		// Gate m: channel 0, gate 0, power 3
		// alpha LINOID: 0.1; A=-0.1, B=-10, V0=25
		// beta EXPONENTIAL: A = 4, B=-18, V0=0
		activeChannels->setGate(2*activeComp, 0, 0.0529, 3, LINOID, -0.1, -10, 25, EXPONENTIAL, 4, -18, 0);

		// Gate h: channel 0, gate 1, power 1
		// alpha EXPONENTIAL: A = 0.07, B=-20, V0=0
		// beta SIGMOID: A = 1, B=-10, V0=30
		activeChannels->setGate(2*activeComp, 1, 0.5960, 1, EXPONENTIAL, 0.07, -20, 0, SIGMOID, 1, -10, 30);

		// Gate n: channel 1, gate 0, power 4
		// alpha LINOID: 0.1; A=-0.01, B=-10, V0=10
		// beta EXPONENTIAL: A = 0.125, B=-80, V0=0
		activeChannels->setGate(2*activeComp+1, 0, 0.3177, 4, LINOID, -0.01, -10, 10, EXPONENTIAL, 0.125, -80, 0);
	}



	/**
	 * Create synaptic channels
	 */
	synapticChannels = new SynapticChannels (this->active, vmList, nComp); // SYN
}


void HinesMatrix::initializeFieldsSingle() {

	mulListSize = 0;
	if (triangAll == 0) {
		for (int l=1; l<nComp; l++)
			for (int pos=0; pos<l; pos++)
				if ( junctions[l][pos] == 1)
					mulListSize++;
	}

	leftListSize = 0;
	for (int c=0; c<nComp; c++)
		for (int l=0; l<nComp; l++)
			if ( junctions[c][l] != 0 || l==c)
				leftListSize++;

	ucompMemory    = new ucomp[2*mulListSize + 2*leftListSize + nComp];
	mulListComp    = ucompMemory;
	mulListDest    = mulListComp  + mulListSize;
	leftListLine   = mulListDest  + mulListSize;
	leftListColumn = leftListLine   + leftListSize;
	leftStartPos   = leftListColumn + leftListSize;

	memory = new ftype[8*nComp + 2*leftListSize + mulListSize];
	rhsM = memory;
	vmList = rhsM + nComp;
	vmTmp = vmList + nComp;
	curr = vmTmp + nComp;
	active = curr + nComp;
	triangList = active + nComp;

	Cm = triangList + leftListSize;
	Ra = Cm + nComp;
	Rm = Ra + nComp;
	leftList   = Rm + nComp;
	mulList    = leftList + leftListSize;

	for (int i=0; i<nComp; i++) {
		Cm[i] = CM * (2*PI*radius[i]*dx);
		Ra[i] = RA * dx / (PI * radius[i] * radius[i]);
		Rm[i] = RM / (2*PI*radius[i]*dx);
		active[i] = 0;
		curr[i] = 0;
		vmList[i] = vRest;

		rhsM[i] = 0;
		vmTmp[i] = 0;
	}

}

/*
 * Create a matrix for the neuron
 */
void HinesMatrix::createTestMatrix() {

	/**
	 * Creates the left Matrix
	 */
	for (int i=0; i<leftListSize; i++)
		leftList[i]=0;

	int pos = 0;
	for (int i=0; i<nComp; i++) {

		leftStartPos[i] = pos;
		int centerPos = 0;
		ftype centerValue = -2 * Cm[i] / dt - 1/Rm[i];

		for (int k=0; k<nComp; k++) {

			if (junctions[i][k] == 1 || i == k) {

				leftListLine[pos]   = i;
				leftListColumn[pos] = k;

				if (junctions[i][k] == 1) {
					if (k < i) leftList[pos] = 1 / Ra[k];
					else leftList[pos] = 1 / Ra[i];
					centerValue -= leftList[pos];
				}
				else if (i == k)
					centerPos   = pos;

				pos++;
			}
		}

		leftList[centerPos] = centerValue;
	}

	// Initializes the right-hand side of the linear system
	//printMatrix(leftList);
	if (triangAll == 0)
		upperTriangularizeSingle();
//	printf (" ----------------------------------------------------------------- \n");
//	printMatrix(triangList);

	currStep = 0;
}

/**
 * Performs the upper triangularization of the matrix
 */
void HinesMatrix::upperTriangularizeSingle() {

	//printf("UpperTriangularizeSingle\n");

	for (int k = 0; k < leftListSize; k++) {
		triangList[k] = leftList[k];
	}

	int mulListPos = 0;
	for (int k = 0; k < leftListSize; k++) {

		int c = leftListColumn[k];
		int l = leftListLine[k];

		if( c < l ) {
			int pos = leftStartPos[c];
			for (; c == leftListLine[pos]; pos++)
				if (leftListColumn[pos] == c)
					break;

			mulList[mulListPos] = -triangList[k] / triangList[pos];
			mulListDest[mulListPos] = l;
			mulListComp[mulListPos] = c;

			//printf ("%f, %d %d\n", mulList[mulListPos], mulListDest[mulListPos], mulListComp[mulListPos]);

			pos = leftStartPos[c];
			int tempK = leftStartPos[l];
			for (; c == leftListLine[pos] && pos < leftListSize; pos++) {
				for (; leftListColumn[tempK] < leftListColumn[pos] && tempK < leftListSize ; tempK++);

				triangList[tempK] += triangList[pos] * mulList[mulListPos];

				//printf ("n=%f, %d %d\n", triangList[tempK], leftListLine[tempK], leftListColumn[tempK]);
			}

			mulListPos++;
		}
	}

}

/***************************************************************************
 * This part is executed in every integration step
 ***************************************************************************/

//void HinesMatrix::findActiveCurrents() {
//
//	activeChannels->evaluateCurrents( );
//
//	int comp;
//	for (int i=0; i<activeChannels->getCompListSize(); i++) {
//
//		comp = activeChannels->getCompList()[i];
//		active[ comp ] -= activeChannels->gNaChannel[i] * activeChannels->ENa ;
//		active[ comp ] -=  activeChannels->gKChannel[i] * activeChannels->EK  ;
//		active[ comp ] -=  ( 1 / Rm[comp] ) * ( activeChannels->ELeak );
//
//	}
//
//}

void HinesMatrix::upperTriangularizeAll() {

	if (activeChannels != 0 && currStep >= 0)
		activeChannels->evaluateCurrents(Rm, active);

	if (synapticChannels != 0)
		synapticChannels->evaluateCurrentsNew(currStep * dt);
		//synapticChannels->evaluateCurrents(currStep * dt, type, neuron);

	for (int i=0; i<nComp; i++)
		rhsM[i] = (-2) * vmList[i] * Cm[i] / dt - curr[i] + active[i];

	for (int k = 0; k < leftListSize; k++) {
		triangList[k] = leftList[k];
	}

	for (int i = 0; i < activeChannels->getCompListSize(); i++) {

		int comp = activeChannels->getCompList()[i];
		int pos = leftStartPos[ comp ];

		for (; leftListColumn[pos] < comp && pos < leftListSize ; pos++);

		triangList[pos] -= activeChannels->getActiveConductances(i);
	}

	for (int k = 0; k < leftListSize; k++) {

		int c = leftListColumn[k];
		int l = leftListLine[k];

		//triangList[k] = leftList[k];
		if( c < l ) {
			int pos = leftStartPos[c];
			for (; c == leftListLine[pos]; pos++)
				if (leftListColumn[pos] == c)
					break;

			double mul = -triangList[k] / triangList[pos];

			pos = leftStartPos[c];
			int tempK = leftStartPos[l];
			for (; c == leftListLine[pos] && pos < leftListSize; pos++) {
				for (; leftListColumn[tempK] < leftListColumn[pos] && tempK < leftListSize ; tempK++);

				triangList[tempK] += triangList[pos] * mul;
			}

			rhsM[l] += rhsM[c] * mul;
		}
	}
}

void HinesMatrix::updateRhs() {

	if (activeChannels != 0 && currStep >= 0)
		activeChannels->evaluateCurrents(Rm, active);

	if (synapticChannels != 0)
		synapticChannels->evaluateCurrentsNew(currStep * dt);

	for (int i=0; i<nComp; i++)
		rhsM[i] = (-2) * vmList[i] * Cm[i] / dt - curr[i] + active[i];

	for (int mulListPos = 0; mulListPos < mulListSize; mulListPos++) {
		int dest = mulListDest[mulListPos];
		int pos = mulListComp[mulListPos];
		rhsM[dest] += rhsM[pos] * mulList[mulListPos];
	}
}

void HinesMatrix::backSubstitute() {

	if (triangAll == 0 && activeChannels != 0)
		vmTmp[nComp-1] = rhsM[nComp-1] / ( triangList[leftListSize-1] - activeChannels->getActiveConductances(0));
	else
		vmTmp[nComp-1] = rhsM[nComp-1]/triangList[leftListSize-1];

	ftype tmp = 0;
	for (int leftListPos = leftListSize-2; leftListPos >=0 ; leftListPos--) {
		int line   = leftListLine[leftListPos];
		int column = leftListColumn[leftListPos];
		if (line == column) {
			vmTmp[line] = (rhsM[line] - tmp) / triangList[leftListPos];
			tmp = 0;
		}
		else
			tmp += vmTmp[column] * triangList[leftListPos];		
	}

	for (int l = 0 ; l < nComp; l++) {		
		vmList[l] = 2 * vmTmp[l] - vmList[l];
		active[l] = 0; // clear the active currents
	}
}

void HinesMatrix::solveMatrix() {

	//if (currStep == 0) printMatrix(leftList);
	if (triangAll == 1)
		upperTriangularizeAll();
	else
		updateRhs();

	//if (currStep == 0) printMatrix(triangList);
	backSubstitute();

	currStep++;

	//printf("%d %f\n", nComp-1, vmList[nComp-1]);

	if (vmList[nComp-1] >= threshold && ((currStep * dt) - lastSpike) > minSpikeInterval) {
		spikeTimes[nGeneratedSpikes] = currStep * dt;
		lastSpike = currStep * dt;
		nGeneratedSpikes++;
	}
}

/***************************************************************************
 * This part is executed in every integration step
 ***************************************************************************/

void HinesMatrix::writeVmToFile(FILE *outFile) {

	fprintf(outFile, "%10.2f\t%10.2f\t%10.2f\n", dt * currStep, vmList[nComp-1], vmList[0]);
}

void HinesMatrix::printMatrix(ftype *list) {

	printf ("-------------------------------------------------\n");
	int pos = 0;
	ftype zero = 0;
	for (int i=0; i<nComp; i++) {
		for (int j=0; j<nComp; j++) {

			if (i == leftListLine[pos] && j == leftListColumn[pos])
				printf( "%10.4e\t", list[pos++]);
			else
				printf( "%10.4e\t", zero);
		}
		printf( "%10.2f\t", vmList[i]); // mV
		printf( "%10.2e\t\n", rhsM[i]);
	}
}

