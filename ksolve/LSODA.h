/***
 *       Filename:  LSODE.h
 *
 *    Description:  See LSODE.cpp file to more information.
 *
 *        Version:  0.0.1
 *        Created:  2018-08-09

 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL3
 */

#ifndef LSODE_H
#define LSODE_H

#include <memory>
#include <array>
using namespace std;


class LSODA
{
    typedef void  (*_lsoda_f) (double, double *, double *, void *);

public:

    LSODA();
    ~LSODA();

    int idamax( int n, double* dx, int incx);

    void dscal(int n, double da, double* dx, int incx);

    double ddot(int n, double* dx, int ncx, double* dy, int incy);

    void daxpy(int n, double da, double* dx, int incx, double* dy, int incy);

    void dgesl(double** a, int n, int* ipvt, double* b, int job);

    void dgefa(double** a, int n, int* ipvt, int* info);

    void prja(int neq, double *y, _lsoda_f f, void *_data);

    int n_lsoda(double y[], int n, double *x, double xout, double eps
            , const double yscal[], _lsoda_f devis, void *data
            );

    void lsoda( _lsoda_f f, int neq, double *y, double *t, double tout
                , int itask, int *istate, int iopt, int jt
                , int iwork1, int iwork2, int iwork5, int iwork6, int iwork7, int iwork8, int iwork9
                , double rwork1, double rwork5, double rwork6, double rwork7
                , void *_data
              );

    void n_lsoda_terminate(void);

    void correction( int neq, double *y, _lsoda_f f, int *corflag
            , double pnorm, double *del, double *delp, double *told
            , int *ncf, double *rh, int *m, void *_data
            );

    void stoda(int neq, double *y, _lsoda_f f, void *_data);

    void lsoda_update( _lsoda_f f, double* y, double* t, const double tout
            , int* istate, void* _data
            );

    void     terminate(int *istate);
    void     terminate2(double *y, double *t);
    void     successreturn(double *y, double *t, int itask, int ihit, double tcrit, int *istate);
    void     _freevectors(void);
    void     ewset(double *ycur);
    void     resetcoeff(void);
    void     solsy(double *y);
    void     endstoda(void);
    void     orderswitch(double *rhup, double dsm, double *pdh, double *rh, int *orderflag);
    void     intdy(double t, int k, double *dky, int *iflag);
    void     corfailure(double *told, double *rh, int *ncf, int *corflag);
    void     methodswitch(double dsm, double pnorm, double *pdh, double *rh);
    void     cfode(int meth);
    void     scaleh(double *rh, double *pdh);
    double   fnorm(int n, double **a, double *w);
    double   vmnorm(int n, double *v, double *w);

private:
    int      g_nyh = 0, g_lenyh = 0;

    int      ml, mu, imxer;
    int      mord[3] = {0, 12, 5};
    double   sqrteta, *yp1, *yp2;
    double   sm1[13] = {0., 0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025};

    double   ccmax, el0, h, hmin, hmxi, hu, rc, tn;
    int      illin = 0, init = 0, mxstep, mxhnil, nhnil, ntrep = 0, nslast, nyh, ierpj, iersl,
             jcur, jstart, kflag, l, meth, miter, maxord, maxcor, msbp, mxncf, n, nq, nst,
             nfe, nje, nqu;
    double   tsw, pdnorm;
    int      ixpr = 0, jtyp, mused, mxordn, mxords;

    double   conit, crate, el[14], elco[13][14], hold, rmax, tesco[13][4];
    int      ialth, ipup, lmax, nslp;
    double   pdest, pdlast, ratio, cm1[13], cm2[6];
    int      icount, irflag;

    double **yh, **wm, *ewt, *savf, *acor;
    int     *ipvt;

private:
    std::array<double, 4> atol_; //= {0.0, 1e-6, 1e-10, 1e-6};
    std::array<double, 4> rtol_; //= {0.0, 1e-04, 1e-8, 1e-4};

    int itol_ = 2;
    int istate_ = 1;
};


#endif /* end of include guard: LSODE_H */
