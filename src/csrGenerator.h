void csrGenerator(int *csrRowPtrA, int *csrColIndA, double *csrValA, 
		  const int NrInterior, const int NzInterior,
		  const double *h_r, const double *h_z, 
		  const double *h_rr, const double *h_s,
		  double *h_f,
		  const double h, const double uInf);
