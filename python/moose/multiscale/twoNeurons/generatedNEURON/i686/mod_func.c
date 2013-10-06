#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," DoubExpSyn.mod");
    fprintf(stderr," LeakConductance.mod");
    fprintf(stderr," SingleSyn1.mod");
    fprintf(stderr, "\n");
  }
  _DoubExpSyn_reg();
  _LeakConductance_reg();
  _SingleSyn1_reg();
}
