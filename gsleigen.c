#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

int
main (void)
{
  double data[] = { -1.0, 1.0, -1.0, 1.0,
                    -8.0, 4.0, -2.0, 1.0,
                    27.0, 9.0, 3.0, 1.0,
                    64.0, 16.0, 4.0, 1.0 };

  double data2[] = { -2.0, 8.0, -2.0, 2.0,
                    -8.0, 4.0, -2.0, 2.0,
                    27.0, 9.0, 3.0, 2.0,
                    100.0, 16.0, 4.0, 2.0 };
 
  gsl_matrix_view m
    = gsl_matrix_view_array (data, 4, 4); /* interpret data as 4x4 matrix, probably has to be gsl_matrix_view_vector later*/

  gsl_vector_complex *eval = gsl_vector_complex_alloc (4); /* allocate vector for eigenvalues */
  gsl_matrix_complex *evec = gsl_matrix_complex_alloc (4, 4); /* allocate matrix of eigenvectors (matrix that diagonalizes data) */ 

  gsl_matrix_complex *First = gsl_matrix_complex_alloc (4, 4); /* P*D */
  gsl_matrix_complex *Reproduce = gsl_matrix_complex_alloc (4, 4); /* P*D*P^(-1) */

  gsl_matrix_complex *inv = gsl_matrix_complex_alloc (4, 4);  /* inverse of matrix evec */
  gsl_matrix_complex *diag = gsl_matrix_complex_alloc (4, 4); /* diagonalized matrix of data */
  gsl_matrix_complex *unit = gsl_matrix_complex_alloc (4, 4); /* to test matrix inversion */

  gsl_eigen_nonsymmv_workspace * w = gsl_eigen_nonsymmv_alloc (4);

  gsl_eigen_nonsymmv (&m.matrix, eval, evec, w); /* This overwrites m/data! */

  gsl_eigen_nonsymmv_free (w);

  gsl_eigen_nonsymmv_sort (eval, evec,
                           GSL_EIGEN_SORT_ABS_DESC); /* sorts eigenvalues and matrix evec according to the size of eigenvalue */

  gsl_matrix_complex * my_diag_alloc(gsl_vector_complex * X) /* function that makes a diagonal matrix from a vector*/
{
    gsl_matrix_complex * mat = gsl_matrix_complex_alloc(X->size, X->size);
    gsl_vector_complex_view diag = gsl_matrix_complex_diagonal(mat);
    gsl_complex zero; 
    GSL_SET_COMPLEX (&zero, 0.0, 0.0);
    gsl_matrix_complex_set_all(mat,zero); 
    gsl_vector_complex_memcpy(&diag.vector, X);
    return mat;
}

  diag = my_diag_alloc(eval); /* diagonalized matrix of m */

  {
    int i, j;

    for (i = 0; i < 4; i++)
      {
        gsl_complex eval_i    /* get eigenvalue number i */
           = gsl_vector_complex_get (eval, i);
        gsl_vector_complex_view evec_i
           = gsl_matrix_complex_column (evec, i); /* get eigenvector number i */

        printf ("eigenvalue = %g + %gi\n",
                GSL_REAL(eval_i), GSL_IMAG(eval_i));
        printf ("eigenvector = \n");
        for (j = 0; j < 4; ++j)
          {
            gsl_complex z =
              gsl_vector_complex_get(&evec_i.vector, j);
            printf("%g + %gi\n", GSL_REAL(z), GSL_IMAG(z)); /* print each line of the eigenvector */
          }
      }
  }

/*----------- Reconstruct the original matrix (data), A=P D P^(-1) -----------*/

/* Multiply evec with diag*/

  gsl_complex alpha, beta; 
  GSL_SET_COMPLEX (&alpha, 1.0, 0.0);
  GSL_SET_COMPLEX (&beta, 0.0, 0.0); 

  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, alpha, evec, diag, beta, First); /* P*D  */

/* Get the inverse of matrix evec: */

  int s;
  gsl_permutation * p = gsl_permutation_alloc (4);

  gsl_linalg_complex_LU_decomp(evec, p, &s);    /* Have to decomposit evec into triangular matrices to find its inverse. This overwrites evec! */
  gsl_linalg_complex_LU_invert(evec, p, inv); 

/* Multiply (evec*diag) with inverse of evec */

  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, alpha, First, inv, beta, Reproduce);

   int i, j;
    for (i = 0; i < 4; i++)
      {
        gsl_vector_complex_view Reproduce_i
           = gsl_matrix_complex_column (Reproduce, i); /* get eigenvector number i */

        printf ("reproduced matrix = \n");
        for (j = 0; j < 4; ++j)
          {
            gsl_complex z2 =
              gsl_vector_complex_get(&Reproduce_i.vector, j);
            printf("%g + %gi\n", GSL_REAL(z2), GSL_IMAG(z2)); /* print each line of the unit matrix */
          }
      } 
  
 /* gsl_permutation_free (p);
  gsl_vector_complex_free(eval);
  gsl_matrix_complex_free(evec);

  gsl_matrix_complex_free(unit);
  gsl_matrix_complex_free(inv);
  gsl_matrix_complex_free(diag); */

  return 0;
}
