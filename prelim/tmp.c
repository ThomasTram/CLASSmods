double doconvolve(double *w, double *Psi, double *ptr, int q_size);

int main(){

  return 0;
}

int get_Z_scatter(double *w, double *Z, double *Psi, int q_size, int lmax, double *Zscatter){
  int index_q, index_l, index_qpr;
  double I;
  double *ptr;
  double *out;

  for (index_q=0; index_q<q_size; index_q++){

    out = Zscatter+index_q*(lmax+1);
    ptr = Z+index_q*q_size*(lmax+1);

    for (index_l=0; index_l<=lmax; index_l++){

      ptr += q_size;

      out[index_l] = doconvolve(w,Psi,ptr,q_size);

    }

  }

  return 0;

}

double doconvolve(double * w, double * Psi, double * ptr, int q_size){
  double I=0;
  int index_qpr;
  for (index_qpr=0; index_qpr<q_size; index_qpr++)
    I += w[index_qpr]*Psi[index_qpr]*ptr[index_qpr];
  return I;
}
