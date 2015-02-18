#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"

#define _SUCCESS_ 0
#define _FAILURE_ -1
#define _TRUE_ 1
#define _FALSE_ 0
#define _ZETA3_ 1.2020569031595942854
#define _PI_ 3.1415926535897932385

double getRealTime( );

#define timing(a) clock_t diff, start=clock(); a; diff = clock() - start; int msec = diff * 1000 / CLOCKS_PER_SEC; printf("msecs: %d\n",msec);
#define wctiming(a) double start=getRealTime(); a; printf("secs: %g\n",1*(getRealTime()-start));


struct cl_integ_parameters {
  double q;
  double qpr;
  double T;
  int l;
};

struct Zlm_integ_parameters {
  double q;
  double qpr;
  double Tnu0;
  double ldbl;
  double x;
  //Parameters for Laguerre:
  int Nlag;
  double *xlag;
  double *wlag;
};

int init_Zlm_param(int Nlag,
                   struct Zlm_integ_parameters * p);
int gk_quad(int function(void * params_for_function, double x, double *fx),
	    void * params_for_function,
	    double a,
	    double b,
	    int isindefinite,
            double *I,
            double *err);
int gk_adapt(int f(void *param, double x, double *fx),
             double xleft,
             double xright,
             double *I,
             double *err,
             void *param,
             double rtol,
             double abstol,
             int isindefinite);
int compute_Laguerre(double *x, double *w, int N, double alpha, double *b, double *c,int totalweight);
int evaluate_rule(int f(void *param, double x, double *fx),
                  void *param,
                  double *x,
                  double *w,
                  int N,
                  double xoffset,
                  double *I);

int myfun(void *param, double x, double *fx);
double Imyfun(double x);
int myfun2(void *param, double x, double *fx);
int Imyfun2(void *param, double *Ix);
int myfuninf(void *param, double x, double *fx);
int legpol(int l, double x, double *P);
double Plx(int l, double x);
int cl_integ(void *param, double x, double *fx);
int compute_Zl0(double Tnu0, double g, double *Z, int lmax, double *qvec, int size_qvec);
int cl_integ2(void *param, double x, double *fx);
int compute_Zl0_omp(double Tnu0, double g, double *Z, int lmax, double *qvec, int size_qvec);

int K3_integ(void *param, double r, double *fx);
int K1_integ(void *param, double r, double *fx);
int Zlm_integ(void *param, double x, double *fx);
int compute_Zlm(double Tnu0, double g, double a_m_phi, double *Z, int lmax, double *qvec, int size_qvec);
int compute_Zlm_omp(double Tnu0, double g, double a_m_phi, double *Z, int lmax, double *qvec, int size_qvec);

int main(){
  double I,err;
  double a=4;
  double rtol = 1e-6;
  double abstol = 1e-15;
  double Iexact,Ilag;
  double xl, xr;
  int Nlag;
  double *xlag, *wlag, *blag, *clag, alpha, fx;
  int index_x;

  /**

  gk_adapt(myfuninf, 0.,1., &I, &err, (void *) (&a),rtol,abstol,_TRUE_);
  Iexact = 1.8030853547393914281;
  printf("\nNumerically: %.16e, Exact: %.16e, Error: %g\n",I,Iexact,1.-I/Iexact);

  Nlag = 30;
  xlag = malloc(sizeof(double)*Nlag);
  wlag = malloc(sizeof(double)*Nlag);
  blag = malloc(sizeof(double)*Nlag);
  clag = malloc(sizeof(double)*Nlag);
  alpha = 0.0;
  compute_Laguerre(xlag, wlag, Nlag, alpha, blag, clag, _TRUE_);

  evaluate_rule(myfuninf,(void *) (&a),xlag,wlag,Nlag,0.,&Ilag);
  printf("Ilag: %.16e using %d points. Error: %g\n",Ilag, Nlag,1.-Ilag/Iexact);

  double x0 = 2.5;
  evaluate_rule(myfuninf,(void *) (&a),xlag,wlag,Nlag,2.5,&Ilag);

  Iexact = 1.0578823034664061127;
  printf("Exact integral on [%g,inf] = %.16e\n",x0,Iexact);

  gk_adapt(myfuninf, 0.,1./(1.+x0), &I, &err, (void *) (&a),rtol,abstol,_TRUE_);
  printf("\nNumerically: %.16e, Error: %g\n",I,1.-I/Iexact);

  printf("Ilag on [%g,inf]: %.16e using %d points. Error: %g\n",x0, Ilag, Nlag,1.-Ilag/Iexact);


  xl = 1.;
  xr = 30.;

  gk_adapt(myfun, xl,xr, &I, &err, (void *) (&a),rtol,abstol,_FALSE_);
  Iexact = Imyfun(xr)-Imyfun(xl);

  printf("\n");
  printf("I = %.16e, err = %g\n",I,err);
  printf("Iexact = %.16e, Actual error: %g\n",Iexact,Iexact-I);

  double param[3];
  double q, qpr, T;
  q = 0.5;
  qpr = 1.5;
  T = 1.;
  param[0] = q;
  param[1] = qpr;
  param[2] = T;

  gk_adapt(myfun2, -1, 1, &I, &err, (void *) param, rtol, abstol, _FALSE_);
  Imyfun2((void *) param, &Iexact);

  printf("\n");
  printf("I = %.16e, err = %g\n",I,err);
  printf("Iexact = %.16e, Actual error: %g\n",Iexact,Iexact-I);

  double qmin = 1e-4;
  double qmax = 12;

  FILE * Inumfile = fopen("Inum.dat","w");
  FILE * Iexactfile = fopen("Iexact.dat","w");
  int i,j;

  for (i=0; i<101; i++){
    q = qmin+i*0.01*(qmax-qmin);
    for (j=0; j<101; j++){
      qpr = qmin+j*0.01*(qmax-qmin);

      param[0] = q;
      param[1] = qpr;
      param[2] = T;

      gk_adapt(myfun2, -1, 1, &I, &err, (void *) param, rtol, abstol, _FALSE_);
      Imyfun2((void *) param, &Iexact);

      fprintf(Inumfile,"%.16e ",I);
      fprintf(Iexactfile,"%.16e ",Iexact);
    }
    fprintf(Inumfile,"\n");
    fprintf(Iexactfile,"\n");
  }

  fclose(Inumfile);
  fclose(Iexactfile);

  FILE * legfile = fopen("legendre.dat","w");
  int lmax = 1000;
  double *P = malloc(sizeof(double)*(lmax+1));

  for (i=0; i<101; i++){
    legpol(lmax,-1+i*0.02,P);
    for (j=0; j<=lmax; j++)
      fprintf(legfile,"%.16e ",P[j]);
    fprintf(legfile,"\n");
  }
  fclose(legfile);
  free(P);

*/

  /** Test Zl0 */
  printf("Testing Zl0 computation...\n");
  int qvec_size=10;
  int lmax2 = 25;
  int ii;
  double *qvec = malloc(sizeof(double)*qvec_size);
  double *Z = malloc(sizeof(double)*qvec_size*qvec_size*(lmax2+1));
  FILE * Zfile;
  char filename[256];
  char mcase;
  double dq;

  dq = 20./(qvec_size-1);

  for (ii=0; ii<qvec_size; ii++)
    qvec[ii] = 1e-1+ii*dq;

  double Tnu0 = 0.71611;
  //mcase='0';compute_Zl0(Tnu0, 0.1, Z, lmax2, qvec, qvec_size);
  //mcase='m';wctiming(compute_Zlm(Tnu0, 0.1, 1.0, Z, lmax2, qvec, qvec_size));
  mcase='m';wctiming(compute_Zlm_omp(Tnu0, 0.1, 1.0, Z, lmax2, qvec, qvec_size));
  //mcase='0';compute_Zl0_omp(Tnu0, 0.1, Z, lmax2, qvec, qvec_size);

  int index_q, index_qpr, index_l;
  for (index_l=0; index_l<=lmax2; index_l++){
    sprintf(filename,"Z%c_%03d.dat",mcase,index_l);
    Zfile = fopen(filename,"w");
    for(index_q=0; index_q<qvec_size; index_q++){
      for(index_qpr=0; index_qpr<qvec_size; index_qpr++){
        fprintf(Zfile,"%.16e ", Z[index_q*(lmax2+1)*qvec_size+index_l*qvec_size+index_qpr]);
      }
      fprintf(Zfile,"\n");
    }
    fclose(Zfile);
  }


  free(qvec);
  free(Z);



  return 0;
}

int init_Zlm_param(int Nlag,
                   struct Zlm_integ_parameters * p){
  double alpha = 0.0;
  p->Nlag = Nlag;
  p->xlag = malloc(sizeof(double)*Nlag);
  p->wlag = malloc(sizeof(double)*Nlag);
  double *b = malloc(sizeof(double)*Nlag);
  double *c = malloc(sizeof(double)*Nlag);
  compute_Laguerre(p->xlag, p->wlag, p->Nlag, alpha, b, c, _TRUE_);
  return _SUCCESS_;
}

/**
int myfun(void *param, double x, double *fx){
  *fx = x*x;
  return _SUCCESS_;
}

double Imyfun(double x){
  return x*x*x/3.;
}
*/


int myfun(void *param, double x, double *fx){
  *fx = sqrt(x)*log(x);
  return _SUCCESS_;
}

int myfuninf(void *param, double x, double *fx){
  *fx = x*x/(exp(x)+1.);
  return _SUCCESS_;
}

int cl_integ(void *param, double x, double *fx){
  double * ptr = param;

  double sq, q, qpr, T;
  int l;
  q = ptr[0];
  qpr = ptr[1];
  T = ptr[2];
  l = (int) ptr[3];

  sq = sqrt(q*q+qpr*qpr-2*q*qpr*x);

  *fx = exp(-(q-qpr+sq)/(2.*T))/sq*Plx(l,x);
  return _SUCCESS_;
}

int Zlm_integ(void *param, double x, double *fx){
  struct Zlm_integ_parameters * ptr = param;

  double sq1, sq2, q, qpr, T;
  double I, err;
  double rtol = 1e-6;
  double abstol = 1e-12;

  int l;

  int Ntrap=500, i;
  double dx, K3itg, xmin, xmax;

  q = ptr->q;
  qpr = ptr->qpr;
  T = ptr->Tnu0;
  l = (int) ptr->ldbl;
  ptr->x = x;

  double K1, K3;

  /* This is |q_vec+qpr_vec| */
  sq1 = sqrt(q*q+qpr*qpr+2.*q*qpr*x);

  /* This is |q_vec-qpr_vec| */
  sq2 = sqrt(q*q+qpr*qpr-2.*q*qpr*x);

  if (1==1){
    /* Compute K3 from Eq. C.9 using gk_adapt() */
    gk_adapt(K3_integ, 0.5*(q+qpr-sq1), 0.5*(q+qpr+sq1), &I, &err, (void *) param, rtol, abstol, _FALSE_);
  }
  else{
    /* Compute K3 from Eq. C.9 using trapezoidal rule: */
    I = 0.;
    xmin = 0.5*(q+qpr-sq1);
    xmax = 0.5*(q+qpr+sq1);
    dx = (xmax-xmin)/(Ntrap-1.);
    for (i=0; i<Ntrap; i++){
      K3_integ(param, xmin+i*dx, &K3itg);
      I += K3itg;
    }
    I *= dx;
  }

  K3 = q*qpr/sq1*I;

  /* Compute K1 from C.21 */
  /* gk_adapt can do the integral int dx _a^Inf if we supply xl = 0, xr = 1/(1+a) and _TRUE_ for is indefinite. */
  //gk_adapt(K1_integ, 0.0, 1./(1.+0.5*(q-qpr+sq2)), &I, &err, (void *) param, rtol, abstol, _TRUE_);
  evaluate_rule(K1_integ, param, ptr->xlag, ptr->wlag, ptr->Nlag, 0.5*(q-qpr+sq2),&I);


  K1 = q*qpr/sq2*I;

  *fx = (2.*K1-exp(-q)*K3)*Plx(l,x);
  return _SUCCESS_;
}


int cl_integ2(void *param, double x, double *fx){
  struct cl_integ_parameters * ptr = param;

  double sq, q, qpr, T;
  int l;
  q = ptr->q;
  qpr = ptr->qpr;
  T = ptr->T;
  l = ptr->l;

  sq = sqrt(q*q+qpr*qpr-2*q*qpr*x);

  *fx = exp(-(q-qpr+sq)/(2.*T))/sq*Plx(l,x);
  return _SUCCESS_;
}


int K3_integ(void *param, double r, double *fx){
  double * ptr = param;

  double q, qpr, T, x;
  double a3, b3, c3;
  double A3, B3, C3;

  ptr = param;

  q = ptr[0];
  qpr = ptr[1];
  T = ptr[2];
  x = ptr[4];

  /** Eq B.12. Note that i want to use Eq. C9 as K_3(q,q',x) so I do the substitutions
      qpr --> r        (r becomes the integration variable in Eq. C.9)
      l --> qpr        (in Eq B.12 l' is l I think)
      cos(alpha) --> x
  */
  a3 = -4.*r*r*(q*q+qpr*qpr+2*q*qpr*x);
  b3 = 8.*r*(-q*qpr+q*qpr*x+q*r+qpr*r)*(q+qpr*x);
  c3 = 4.*qpr*qpr*r*r*(1.-x*x)-4.*pow(-q*qpr*(1-x)+q*r+qpr*r,2);

  /** Eq C.4, same subs as above */
  A3 = r*r;
  B3 = r*(qpr*(1.-x)-2.*r);
  C3 = r*r+qpr*qpr*(1.+x*x-2.*x)-r*qpr*(1.-x);

  /* Eq. C.9 */
  *fx = C3+(2.*B3*b3+A3*c3)/(2.*fabs(a3))+3.*A3*b3*b3/(8.*a3*a3);
  return _SUCCESS_;
}

int K1_integ(void *param, double r, double *fx){
  double * ptr = param;

  double q, qpr, T, x;
  double a1, b1, c1;
  double A1, B1, C1;

  ptr = param;

  q = ptr[0];
  qpr = ptr[1];
  T = ptr[2];
  x = ptr[4];

  /** I substitute
      l' --> r (so r is my integration variable in C21)
      cos(theta) --> x
  */

  /** Eq. B.36 */
  a1 = -4.*r*r*(qpr*qpr+q*q-2*q*qpr*x);
  b1 = 8.*r*(q*qpr*(x-1.)+qpr*r-r*q)*(qpr*x-q);
  c1 = 4.*r*r*qpr*qpr*(1-x*x)-4.*pow(q*qpr*(x-1.)-r*q+r*qpr,2);

  /** Eq C.17 */
  A1 = r*r;
  B1 = -r*(qpr*(1.-x)+2.*r);
  C1 = r*r+qpr*qpr*(1.+x*x-2.*x)+qpr*r*(1.-x);

  /** Eq. C21 */
  *fx = exp(-r)*(C1+(2.*B1*b1+A1*c1)/(2.*fabs(a1))+3.*A1*b1*b1/(8.*a1*a1));

  return _SUCCESS_;
}


int compute_Zl0(double Tnu0, double g, double *Z, int lmax, double *qvec, int size_qvec){

  /** To optimise the convolution integrals later, the best layout for Z is
      Z[index_qvec*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] */


  double q, qpr, ldbl;
  int index_q, index_qpr, index_l;
  double param[4];
  double rtol = 1e-6;
  double abstol = 1e-12;
  double I, err;
  double lastterm;

  double N, b, factor;

  N = 0.75*_ZETA3_;
  b = 24.0;
  factor = N*b*pow(g,4)/(8.*pow(2.*_PI_,3));

  param[2] = Tnu0;

  for (index_q = 0; index_q<size_qvec; index_q++){

    q = qvec[index_q];
    param[0] = q;

    for (index_l = 0; index_l <= lmax; index_l++){

      if (index_l == 0)
        lastterm = exp(-q/Tnu0);
      else
        lastterm = 0.;

      ldbl = index_l;
      param[3] = ldbl;

      for (index_qpr = 0; index_qpr<size_qvec; index_qpr++){

        qpr = qvec[index_qpr];
        param[1] = qpr;

        gk_adapt(cl_integ, -1, 1, &I, &err, (void *) param, rtol, abstol, _FALSE_);
        Z[index_q*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] =
          factor*qpr/q*(Tnu0*I-lastterm);

      }

    }

  }

  return _SUCCESS_;
}

int compute_Zlm(double Tnu0, double g, double a_m_phi, double *Z, int lmax, double *qvec, int size_qvec){

  /** To optimise the convolution integrals later, the best layout for Z is
      Z[index_qvec*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] */


  double q, qpr, ldbl;
  int index_q, index_qpr, index_l;
  struct Zlm_integ_parameters param;
  double rtol = 1e-6;
  double abstol = 1e-12;
  double I, err;
  int Nlag = 32;

  double N, c, factor;

  N = 0.75*_ZETA3_;
  c = 2.0;
  factor = N*c*pow(g,4)/(2.*pow(2.*_PI_,3)*pow(a_m_phi,4));

  init_Zlm_param(Nlag, &param);
  param.Tnu0 = Tnu0;

  for (index_q = 0; index_q<size_qvec; index_q++){

    q = qvec[index_q];
    param.q = q;

    for (index_l = 0; index_l <= lmax; index_l++){

      ldbl = index_l;
      param.ldbl = ldbl;

      for (index_qpr = 0; index_qpr<size_qvec; index_qpr++){

        qpr = qvec[index_qpr];
        param.qpr = qpr;

        gk_adapt(Zlm_integ, -1, 1, &I, &err, (void *) (&param), rtol, abstol, _FALSE_);
        Z[index_q*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] = factor*I;

      }

    }

  }

  return _SUCCESS_;
}

int compute_Zlm_omp(double Tnu0, double g, double a_m_phi, double *Z, int lmax, double *qvec, int size_qvec){

  /** To optimise the convolution integrals later, the best layout for Z is
      Z[index_qvec*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] */


  int index_q, index_qpr, index_l;
  struct Zlm_integ_parameters param;
  double rtol = 1e-6;
  double abstol = 1e-12;
  double I, err;
  int Nlag = 32;

  double N, c, factor;

  N = 0.75*_ZETA3_;
  c = 2.0;
  factor = N*c*pow(g,4)/(2.*pow(2.*_PI_,3)*pow(a_m_phi,4));

  init_Zlm_param(Nlag, &param);
  param.Tnu0 = Tnu0;

#pragma omp parallel for                                                \
  shared(qvec, size_qvec, lmax, rtol, abstol, factor)                   \
  firstprivate(param)                                                   \
  private(index_q, index_qpr, index_l, I, err)                          \
  schedule(static,1)
  for (index_q = 0; index_q<size_qvec; index_q++){

    param.q = qvec[index_q];

    for (index_l = 0; index_l <= lmax; index_l++){

      param.ldbl = index_l;

      for (index_qpr = 0; index_qpr<size_qvec; index_qpr++){

        param.qpr = qvec[index_qpr];

        gk_adapt(Zlm_integ, -1, 1, &I, &err, (void *) (&param), rtol, abstol, _FALSE_);
        Z[index_q*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] = factor*I;

      }

    }

  }

  return _SUCCESS_;
}

int compute_Zl0_omp(double Tnu0, double g, double *Z, int lmax, double *qvec, int size_qvec){

  /** To optimise the convolution integrals later, the best layout for Z is
      Z[index_qvec*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] */

  struct cl_integ_parameters param;

  double q, qpr, ldbl;
  int index_q, index_qpr, index_l;
  double rtol = 1e-6;
  double abstol = 1e-12;
  double I, err;
  double lastterm;

  double N, b, factor;

  N = 0.75*_ZETA3_;
  b = 24.0;
  factor = N*b*pow(g,4)/(8.*pow(2.*_PI_,3));

  //param[2] = Tnu0;

  omp_set_num_threads(4);

#pragma omp parallel for                                                \
  shared(Tnu0, qvec, size_qvec, lmax, rtol, abstol)                     \
  private(index_q, index_qpr, index_l, param, lastterm, I, err)         \
  schedule(static,1)
  for (index_q = 0; index_q<size_qvec; index_q++){

    //    printf("Number of potential threads: %d\n",omp_get_num_threads());

    param.T = Tnu0;
    param.q = qvec[index_q];

    for (index_l = 0; index_l <= lmax; index_l++){

      if (index_l == 0)
        lastterm = exp(-param.q/Tnu0);
      else
        lastterm = 0.;

      param.l = index_l;

      for (index_qpr = 0; index_qpr<size_qvec; index_qpr++){

        param.qpr = qvec[index_qpr];

        gk_adapt(cl_integ2, -1, 1, &I, &err, (void *) (&param), rtol, abstol, _FALSE_);
        Z[index_q*(lmax+1)*size_qvec+index_l*size_qvec+index_qpr] =
          factor*param.qpr/param.q*(Tnu0*I-lastterm);

      }

    }

  }

  return _SUCCESS_;
}

int get_Z_scatter(double *w, double *Z, double *Psi, int q_size, int lmax, double *Zscatter){
  int index_q, index_l, index_qpr;
  double I;
  double *ptr;

  for (index_q=0; index_q<q_size; index_q++){

    ptr = Z+index_q*q_size*(lmax+1);

    for (index_l=0; index_l<=lmax; index_l++){

      I = 0;
      for (index_qpr=0; index_qpr<q_size; index_qpr++)
        I += w[index_qpr]*Psi[index_qpr]*ptr[index_l*q_size+index_qpr];

      Zscatter[index_q*(lmax+1)+index_l] = I;

    }

  }

  return _SUCCESS_;

}



double Imyfun(double x){
  return ((2*pow(x,1.5)*(-2. + 3.*log(x)))/9.);
}

int myfun2(void *param, double x, double *fx){
  double * ptr = param;

  double sq, q, qpr, T;
  q = ptr[0];
  qpr = ptr[1];
  T = ptr[2];
  sq = sqrt(q*q+qpr*qpr-2*q*qpr*x);

  *fx = exp(-(q-qpr+sq)/(2.*T))/sq;
  return _SUCCESS_;
}

int Imyfun2(void *param, double *Ix){
  double * ptr = param;

  double q, qpr, T;
  q = ptr[0];
  qpr = ptr[1];
  T = ptr[2];

  *Ix = -2.*T/(q*qpr)*exp(-q/T)*(1.-exp((q+qpr-fabs(q-qpr))/(2.*T)));

  return _SUCCESS_;
}

int evaluate_rule(int f(void *param, double x, double *fx),
                  void *param,
                  double *x,
                  double *w,
                  int N,
                  double xoffset,
                  double *I){
  int index_x;
  double fx;

  for (index_x=0, *I=0.; index_x<N; index_x++){
    f(param,x[index_x]+xoffset,&fx);
    *I += fx*w[index_x];
  }
  return _SUCCESS_;
}

int gk_adapt(int f(void *param, double x, double *fx),
             double xleft,
             double xright,
             double *I,
             double *err,
             void *param,
             double rtol,
             double abstol,
             int isindefinite){

  double Ileft, Iright, Eleft, Eright;
  //printf("#");

  gk_quad(f,
          param,
          xleft,
	  xright,
	  isindefinite,
          I,
          err);

  if ((fabs(*I)<abstol) || (fabs((*err)/(*I)) < rtol)){
    /* Converged! */
    return _SUCCESS_;
  }
  else{
    gk_adapt(f,xleft,0.5*(xleft+xright),&Ileft,&Eleft,param,1.4*rtol,0.5*abstol,isindefinite);
    gk_adapt(f,0.5*(xleft+xright),xright,&Iright,&Eright,param,1.4*rtol,0.5*abstol,isindefinite);
    *I = Ileft+Iright;
    *err = sqrt(Eleft*Eleft+Eright*Eright);
    return _SUCCESS_;
  }
}

int gk_quad(int function(void * params_for_function, double x, double *fx),
	    void * params_for_function,
	    double a,
	    double b,
	    int isindefinite,
            double *I,
            double *err){
  const double z_k[15]={-0.991455371120813,
			-0.949107912342759,
			-0.864864423359769,
			-0.741531185599394,
			-0.586087235467691,
			-0.405845151377397,
			-0.207784955007898,
			0.0,
			0.207784955007898,
			0.405845151377397,
			0.586087235467691,
			0.741531185599394,
			0.864864423359769,
			0.949107912342759,
			0.991455371120813};
  const double w_k[15]={0.022935322010529,
			0.063092092629979,
			0.104790010322250,
			0.140653259715525,
			0.169004726639267,
			0.190350578064785,
			0.204432940075298,
			0.209482141084728,
			0.204432940075298,
			0.190350578064785,
			0.169004726639267,
			0.140653259715525,
			0.104790010322250,
			0.063092092629979,
			0.022935322010529};
  const double w_g[7]={0.129484966168870,
		       0.279705391489277,
		       0.381830050505119,
		       0.417959183673469,
		       0.381830050505119,
		       0.279705391489277,
		       0.129484966168870};
  int i,j;
  double x,wg,wk,t,Ik,Ig,y,y2;

  /* 	Loop through abscissas, transform the interval and form the Kronrod
     15 point estimate of the integral.
     Every second time we update the Gauss 7 point quadrature estimate. */

  Ik=0.0;
  Ig=0.0;
  for (i=0;i<15;i++){
    /* Transform z into t in interval between a and b: */
    t = 0.5*(a*(1-z_k[i])+b*(1+z_k[i]));
    /* Modify weight such that it reflects the linear transformation above: */
    wk = 0.5*(b-a)*w_k[i];
    if (isindefinite==_TRUE_){
      /* Transform t into x in interval between 0 and inf: */
      x = 1.0/t-1.0;
      //  printf("%g ",x);
      /* Modify weight accordingly: */
      wk = wk/(t*t);
    }
    else{
      x = t;
    }
    function(params_for_function,x,&y);
    /* Update Kronrod integral: */
    Ik +=wk*y;
    /* If i is uneven, update Gauss integral: */
    if ((i%2)==1){
      j = (i-1)/2;
      /* Transform weight according to linear transformation: */
      wg = 0.5*(b-a)*w_g[j];
      if (isindefinite == _TRUE_){
        /* Transform weight according to non-linear transformation x = 1/t -1: */
        wg = wg/(t*t);
      }
      /* Update integral: */
      Ig +=wg*y;
    }
  }
  //  printf("\n");
  *err = pow(200.*fabs(Ik-Ig),1.5);
  *I = Ik;
  return _SUCCESS_;
}


int legpol(int l, double x, double *P){
  int index_l;
  double ldbl;
  if (l>=0){
    P[0] = 1.;
  }
  if (l>=1){
    P[1] = x;
  }
  for (index_l=2; index_l<=l; index_l++){
    ldbl = index_l;
    P[index_l] = (2.*ldbl-1.)/ldbl*x*P[index_l-1]-(ldbl-1.)/ldbl*P[index_l-2];
  }
  return _SUCCESS_;
}

double Plx(int l, double x){
  double Plm2, Plm1, Pl;
  int ll;
  double ldbl;
  if (l==0)
    return 1.;
  if (l==1)
    return x;
  if (l>1){
    Plm2 = 1.;
    Plm1 = x;
    for (ll=2; ll<=l; ll++){
      ldbl = ll;
      Pl = (2.*ldbl-1.)/ldbl*x*Plm1-(ldbl-1.)/ldbl*Plm2;
      Plm2 = Plm1;
      Plm1 = Pl;
    }
    return Pl;
  }
  return 0.;
}


int compute_Laguerre(double *x, double *w, int N, double alpha, double *b, double *c,int totalweight){
  int i,j,iter,maxiter=10;
  double x0=0.,r1,r2,ratio,d,logprod,logcc;
  double p0,p1,p2,dp0,dp1,dp2;
  double eps=1e-14;
  /* Initialise recursion coefficients: */
  for(i=0; i<N; i++){
    b[i] = alpha + 2.0*i +1.0;
    c[i] = i*(alpha+i);
  }
  logprod = 0.0;
  for(i=1; i<N; i++) logprod +=log(c[i]);
  logcc = lgamma(alpha+1)+logprod;

  /* Loop over roots: */
  for (i=0; i<N; i++){
    /* Estimate root: */
    if (i==0) {
      x0 =(1.0+alpha)*(3.0+0.92*alpha)/( 1.0+2.4*N+1.8*alpha);
    }
    else if (i==1){
      x0 += (15.0+6.25*alpha)/( 1.0+0.9*alpha+2.5*N);
    }
    else{
      r1 = (1.0+2.55*(i-1))/( 1.9*(i-1));
      r2 = 1.26*(i-1)*alpha/(1.0+3.5*(i-1));
      ratio = (r1+r2)/(1.0+0.3*alpha);
      x0 += ratio*(x0-x[i-2]);
    }
    /* Refine root using Newtons method: */
    for(iter=1; iter<=maxiter; iter++){
      /* We need to find p2=L_N(x0), dp2=L'_N(x0) and
	 p1 = L_(N-1)(x0): */
      p1 = 1.0;
      dp1 = 0.0;
      p2 = x0 - alpha - 1.0;
      dp2 = 1.0;
      for (j=1; j<N; j++ ){
	p0 = p1;
	dp0 = dp1;
	p1 = p2;
	dp1 = dp2;
	p2  = (x0-b[j])*p1 - c[j]*p0;
	dp2 = (x0-b[j])*dp1 + p1 - c[j]*dp0;
      }
      /* New guess at root: */
      d = p2/dp2;
      x0 -= d;
      if (fabs(d)<=eps*(fabs(x0)+1.0)) break;
    }
    /* Okay, write root and weight: */
    x[i] = x0;

    if (totalweight == _TRUE_)
      w[i] = exp(x0+logcc-log(dp2*p1))*pow(x0,-alpha);
    else
       w[i] = exp(logcc-log(dp2*p1));
  }

  return 0;

}



/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <Windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>	/* POSIX flags */
//#include <time.h>	/* clock_gettime(), time() */
#include <sys/time.h>	/* gethrtime(), gettimeofday() */

#if defined(__MACH__) && defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

#else
#error "Unable to define getRealTime( ) for an unknown OS."
#endif





/**
 * Returns the real time, in seconds, or -1.0 if an error occurred.
 *
 * Time is measured since an arbitrary and OS-dependent start time.
 * The returned real time is only useful for computing an elapsed time
 * between two calls to this function.
 */
double getRealTime( )
{
#if defined(_WIN32)
	FILETIME tm;
	ULONGLONG t;
#if defined(NTDDI_WIN8) && NTDDI_VERSION >= NTDDI_WIN8
	/* Windows 8, Windows Server 2012 and later. ---------------- */
	GetSystemTimePreciseAsFileTime( &tm );
#else
	/* Windows 2000 and later. ---------------------------------- */
	GetSystemTimeAsFileTime( &tm );
#endif
	t = ((ULONGLONG)tm.dwHighDateTime << 32) | (ULONGLONG)tm.dwLowDateTime;
	return (double)t / 10000000.0;

#elif (defined(__hpux) || defined(hpux)) || ((defined(__sun__) || defined(__sun) || defined(sun)) && (defined(__SVR4) || defined(__svr4__)))
	/* HP-UX, Solaris. ------------------------------------------ */
	return (double)gethrtime( ) / 1000000000.0;

#elif defined(__MACH__) && defined(__APPLE__)
	/* OSX. ----------------------------------------------------- */
	static double timeConvert = 0.0;
	if ( timeConvert == 0.0 )
	{
		mach_timebase_info_data_t timeBase;
		(void)mach_timebase_info( &timeBase );
		timeConvert = (double)timeBase.numer /
			(double)timeBase.denom /
			1000000000.0;
	}
	return (double)mach_absolute_time( ) * timeConvert;

#elif defined(_POSIX_VERSION)
	/* POSIX. --------------------------------------------------- */
#if defined(_POSIX_TIMERS) && (_POSIX_TIMERS > 0)
	{
		struct timespec ts;
#if defined(CLOCK_MONOTONIC_PRECISE)
		/* BSD. --------------------------------------------- */
		const clockid_t id = CLOCK_MONOTONIC_PRECISE;
#elif defined(CLOCK_MONOTONIC_RAW)
		/* Linux. ------------------------------------------- */
		const clockid_t id = CLOCK_MONOTONIC_RAW;
#elif defined(CLOCK_HIGHRES)
		/* Solaris. ----------------------------------------- */
		const clockid_t id = CLOCK_HIGHRES;
#elif defined(CLOCK_MONOTONIC)
		/* AIX, BSD, Linux, POSIX, Solaris. ----------------- */
		const clockid_t id = CLOCK_MONOTONIC;
#elif defined(CLOCK_REALTIME)
		/* AIX, BSD, HP-UX, Linux, POSIX. ------------------- */
		const clockid_t id = CLOCK_REALTIME;
#else
		const clockid_t id = (clockid_t)-1;	/* Unknown. */
#endif /* CLOCK_* */
		if ( id != (clockid_t)-1 && clock_gettime( id, &ts ) != -1 )
			return (double)ts.tv_sec +
				(double)ts.tv_nsec / 1000000000.0;
		/* Fall thru. */
	}
#endif /* _POSIX_TIMERS */

	/* AIX, BSD, Cygwin, HP-UX, Linux, OSX, POSIX, Solaris. ----- */
	struct timeval tm;
	gettimeofday( &tm, NULL );
	return (double)tm.tv_sec + (double)tm.tv_usec / 1000000.0;
#else
	return -1.0;		/* Failed. */
#endif
}
