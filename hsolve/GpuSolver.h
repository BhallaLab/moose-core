#ifndef EXAMPLE6_H
#define EXAMPLE6_H

class GpuInterface
{
        public:
                int n[20];
                int y;
                int asize;
		double *A_d, *B_d;
		int *ASize_d, *BSize_d;
		double *xmin_d, *xmax_d;
		double *invDx_d;

                GpuInterface();
		void sayHi();
		void setupTables(double*, double*, double, double, double*, double*, double*);
		void lookupTables(double&, double*, double*) const;
                int calculateSum();
                void setY(int);
};
#endif
