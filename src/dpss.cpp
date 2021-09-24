//
// Created by heyanbai on 2021/9/12.
//

#include "dpss.h"

dpss::dpss(uint16_t seq_length, double time_halfbandwidth, uint16_t num_seq) {
    //TODO: Better Assertation
    assert(time_halfbandwidth < seq_length/2);

    _seq_length = seq_length;
    _time_halfbandwidth = time_halfbandwidth;
    if(num_seq == 0){
        _num_seq = std::min( uint16_t(std::round(2 * time_halfbandwidth)), seq_length);
        _num_seq = std::max(_num_seq, uint16_t(1));
    }else{
        _num_seq = num_seq;
    }

    _tapers = Eigen::MatrixXd(_num_seq, _seq_length);
    _tapersum = Eigen::VectorXd(_num_seq);
    _eigvals = Eigen::VectorXd(_num_seq);

    _compute(_seq_length, _num_seq, _eigvals.data(), time_halfbandwidth, _tapers.data(), _tapersum.data());

    _tapers = _tapers / sqrt(_seq_length);
    for( int i = 0; i < _num_seq; ++i ){
        if (i % 2 == 0){
            if(_tapersum[i] < 0){
                _tapersum[i] *= -1;
                _tapers.row(i) *= -1;
            }
        }else{
            if( _tapers(i, 0) < 0){
                _tapersum[i] *= -1;
                _tapers.row(i) *= -1;
            }
        }
    }

    
}

bool dpss::_compute(int num_points, int nwin, double *lam, float npi, double *tapers, double *tapsum) {
/*
	 * get the multitaper slepian functions:
	num_points = number of points in data stream
	nwin = number of windows
	lam= vector of eigenvalues
	npi = order of
	slepian functions
	tapsum = sum of each taper, saved for use in adaptive weighting
	tapers =  matrix of slepian tapers, packed in a 1D double array
	 */

    int             i, j, k, kk;
    double         *z, ww, cs, ai, an, eps, rlu, rlb, aa;
    double          dfac, drat, gamma, bh, tapsq, TWOPI, DPI;
    double         *diag, *offdiag, *offsq;
    char           *k1[4];

    char            name[81];
    double         *scratch1, *scratch2, *scratch3, *scratch4, *scratch6;


    /* need to initialize iwflag = 0 */
    double          anpi;
    double         *ell;
    int             key, nbin, npad;
    int            *ip;
    double         *evecs;
    double         *zee;

    long            len;
    int             ierr;
    int             m11;
    DPI = M_PI;
    TWOPI = (double) 2 *DPI;

    anpi = npi;
    an = (double) (num_points);
    ww = (double) (anpi) / an;	/* this corresponds to P&W's W value  */
    cs = cos(TWOPI * ww);

    ell = new double[nwin];

    diag = new double[num_points];
    offdiag = new double[num_points];
    offsq = new double[num_points];

    scratch1 = new double[num_points];
    scratch2 = new double[num_points];
    scratch3 = new double[num_points];
    scratch4 = new double[num_points];
    scratch6 = new double[num_points];



    /* make the diagonal elements of the tridiag matrix  */

    for (i = 0; i < num_points; i++) {
        ai = (double) (i);
        diag[i] = -cs * (((an - 1.) / 2. - ai)) * (((an - 1.) / 2. - ai));
        offdiag[i] = -ai * (an - ai) / 2.;
        offsq[i] = offdiag[i] * offdiag[i];
    }

    eps = 1.0e-13;
    m11 = 1;


    ip = new int[nwin];

    /* call the eispac routines to invert the tridiagonal system */

    _jtridib(&num_points, &eps, diag, offdiag, offsq, &rlb, &rlu, &m11, &nwin, lam,
             ip, &ierr, scratch1, scratch2);
#if DIAG1
    fprintf(stderr, "ierr=%d rlb=%.8f rlu=%.8f\n", ierr, rlb, rlu);

	fprintf(stderr, "eigenvalues for the eigentapers\n");

	for (k = 0; k < nwin; k++)
		fprintf(stderr, "%.20f ", lam[k]);
	blank();
#endif


    len = num_points * nwin;
    evecs = new double[len];


    _jtinvit(&num_points, &num_points, diag, offdiag, offsq, &nwin, lam, ip, evecs, &ierr,
             scratch1, scratch2, scratch3, scratch4, scratch6);



    delete [] scratch1;
    delete [] scratch2;
    delete [] scratch3;
    delete [] scratch4;
    delete [] scratch6;



    /*
     * we calculate the eigenvalues of the dirichlet-kernel problem i.e.
     * the bandwidth retention factors from slepian 1978 asymptotic
     * formula, gotten from thomson 1982 eq 2.5 supplemented by the
     * asymptotic formula for k near 2n from slepian 1978 eq 61 more
     * precise values of these parameters, perhaps useful in adaptive
     * spectral estimation, can be calculated explicitly using the
     * rayleigh-quotient formulas in thomson (1982) and park et al (1987)
     *
     */
    dfac = (double) an *DPI * ww;
    drat = (double) 8. *dfac;


    dfac = (double) 4. *sqrt(DPI * dfac) * exp((double) (-2.0) * dfac);


    for (k = 0; k < nwin; k++) {
        lam[k] = (double) 1.0 - (double) dfac;
        dfac = dfac * drat / (double) (k + 1);



        /* fails as k -> 2n */
    }


    gamma = log((double) 8. * an * sin((double) 2. * DPI * ww)) + (double) 0.5772156649;



    for (k = 0; k < nwin; k++) {
        bh = -2. * DPI * (an * ww - (double) (k) /
                                    (double) 2. - (double) .25) / gamma;
        ell[k] = (double) 1. / ((double) 1. + exp(DPI * (double) bh));

    }

    for (i = 0; i < nwin; i++)
        lam[i] = fmax(ell[i], lam[i]);

    /************************************************************
        c   normalize the eigentapers to preserve power for a white process
        c   i.e. they have rms value unity
        c  tapsum is the average of the eigentaper, should be near zero for
        c  antisymmetric tapers
        ************************************************************/

    for (k = 0; k < nwin; k++) {
        kk = (k) * num_points;
        tapsum[k] = 0.;
        tapsq = 0.;
        for (i = 0; i < num_points; i++) {
            aa = evecs[i + kk];
            tapers[i + kk] = aa;
            tapsum[k] = tapsum[k] + aa;
            tapsq = tapsq + aa * aa;
        }
        aa = sqrt(tapsq / (double) num_points);
        tapsum[k] = tapsum[k] / aa;

        for (i = 0; i < num_points; i++) {
            tapers[i + kk] = tapers[i + kk] / aa;

        }
    }


    /* Free Memory */

    delete [] ell;
    delete [] diag;
    delete [] offdiag;
    delete [] offsq;
    delete [] ip;
    delete [] evecs;

    return 1;
}


int dpss::_jtinvit(int *nm, int *n, double *d, double *e, double *e2, int *m, double *w, int *ind, double *z, int *ierr,
                   double *rv1, double *rv2, double *rv3, double *rv4, double *rv6){
    /* Initialized data */

    static double machep = 1.25e-15;

    /* System generated locals */
    int z_dim1, z_offset, i__1, i__2, i__3;
    double d__1, d__2;

    /* Local variables */
    static double norm;
    static int i, j, p, q, r, s;
    static double u, v, order;
    static int group;
    static double x0, x1;
    static int ii, jj, ip;
    static double uk, xu;
    static int tag, its;
    static double eps2, eps3, eps4;

    static double rtem;


/*     this subroutine is a translation of the inverse iteration tech- */
/*     nique in the algol procedure tristurm by peters and wilkinson. */
/*     handbook for auto. comp., vol.ii-linear algebra, 418-439(1971). */

/*     this subroutine finds those eigenvectors of a tridiagonal */
/*     symmetric matrix corresponding to specified eigenvalues, */
/*     using inverse iteration. */

/*     on input: */

/*        nm must be set to the row dimension of two-dimensional */
/*          array parameters as declared in the calling program */
/*          dimension statement; */

/*        n is the order of the matrix; */

/*        d contains the diagonal elements of the input matrix; */

/*        e contains the subdiagonal elements of the input matrix */
/*          in its last n-1 positions.  e(1) is arbitrary; */

/*        e2 contains the squares of the corresponding elements of e, */
/*          with zeros corresponding to negligible elements of e. */
/*          e(i) is considered negligible if it is not larger than */
/*          the product of the relative machine precision and the sum */
/*          of the magnitudes of d(i) and d(i-1).  e2(1) must contain */
/*          0.0d0 if the eigenvalues are in ascending order, or 2.0d0 */
/*          if the eigenvalues are in descending order.  if  bisect, */
/*          tridib, or  imtqlv  has been used to find the eigenvalues, */
/*          their output e2 array is exactly what is expected here; */

/*        m is the number of specified eigenvalues; */

/*        w contains the m eigenvalues in ascending or descending order;
*/

/*        ind contains in its first m positions the submatrix indices */
/*          associated with the corresponding eigenvalues in w -- */
/*          1 for eigenvalues belonging to the first submatrix from */
/*          the top, 2 for those belonging to the second submatrix, etc.
*/

/*     on output: */

/*        all input arrays are unaltered; */

/*        z contains the associated set of orthonormal eigenvectors. */
/*          any vector which fails to converge is set to zero; */

/*        ierr is set to */
/*          zero       for normal return, */
/*          -r         if the eigenvector corresponding to the r-th */
/*                     eigenvalue fails to converge in 5 iterations; */

/*        rv1, rv2, rv3, rv4, and rv6 are temporary storage arrays. */

/*     questions and comments should be directed to b. s. garbow, */
/*     applied mathematics division, argonne national laboratory */

/*     ------------------------------------------------------------------
*/

/*     :::::::::: machep is a machine dependent parameter specifying */
/*                the relative precision of floating point arithmetic. */
/*                machep = 16.0d0**(-13) for long form arithmetic */
/*                on s360 :::::::::: */
/*  for f_floating dec fortran */
/*      data machep/1.1d-16/ */
/*  for g_floating dec fortran */
    /* Parameter adjustments */
    --rv6;
    --rv4;
    --rv3;
    --rv2;
    --rv1;
    --e2;
    --e;
    --d;
    z_dim1 = *nm;
    z_offset = z_dim1 + 1;
    z -= z_offset;
    --ind;
    --w;

    /* Function Body */

    *ierr = 0;
    if (*m == 0) {
        goto L1001;
    }
    tag = 0;
    order = 1. - e2[1];
    q = 0;
/*     :::::::::: establish and process next submatrix :::::::::: */
    L100:
    p = q + 1;

    i__1 = *n;
    for (q = p; q <= i__1; ++q) {
        if (q == *n) {
            goto L140;
        }
        if (e2[q + 1] == 0.) {
            goto L140;
        }
/* L120: */
    }
/*     :::::::::: find vectors by inverse iteration :::::::::: */
    L140:
    ++tag;
    s = 0;

    i__1 = *m;
    for (r = 1; r <= i__1; ++r) {
        if (ind[r] != tag) {
            goto L920;
        }
        its = 1;
        x1 = w[r];
        if (s != 0) {
            goto L510;
        }
/*     :::::::::: check for isolated root :::::::::: */
        xu = 1.;
        if (p != q) {
            goto L490;
        }
        rv6[p] = 1.;
        goto L870;
        L490:
        norm = (d__1 = d[p], abs(d__1));
        ip = p + 1;

        i__2 = q;
        for (i = ip; i <= i__2; ++i) {
/* L500: */
            norm = norm + (d__1 = d[i], abs(d__1)) + (d__2 = e[i], abs(d__2));
        }
/*     :::::::::: eps2 is the criterion for grouping, */
/*                eps3 replaces zero pivots and equal */
/*                roots are modified by eps3, */
/*                eps4 is taken very small to avoid overflow :::::::::
: */
        eps2 = norm * .001;
        eps3 = machep * norm;
        uk = (double) (q - p + 1);
        eps4 = uk * eps3;
        uk = eps4 / sqrt(uk);
        s = p;
        L505:
        group = 0;
        goto L520;
/*     :::::::::: look for close or coincident roots :::::::::: */
        L510:
        if ((d__1 = x1 - x0, abs(d__1)) >= eps2) {
            goto L505;
        }
        ++group;
        if (order * (x1 - x0) <= 0.) {
            x1 = x0 + order * eps3;
        }
/*     :::::::::: elimination with interchanges and */
/*                initialization of vector :::::::::: */
        L520:
        v = 0.;

        i__2 = q;
        for (i = p; i <= i__2; ++i) {
            rv6[i] = uk;
            if (i == p) {
                goto L560;
            }
            if ((d__1 = e[i], abs(d__1)) < abs(u)) {
                goto L540;
            }
/*     :::::::::: warning -- a divide check may occur here if */
/*                e2 array has not been specified correctly ::::::
:::: */
            xu = u / e[i];
            rv4[i] = xu;
            rv1[i - 1] = e[i];
            rv2[i - 1] = d[i] - x1;
            rv3[i - 1] = 0.;
            if (i != q) {
                rv3[i - 1] = e[i + 1];
            }
            u = v - xu * rv2[i - 1];
            v = -xu * rv3[i - 1];
            goto L580;
            L540:
            xu = e[i] / u;
            rv4[i] = xu;
            rv1[i - 1] = u;
            rv2[i - 1] = v;
            rv3[i - 1] = 0.;
            L560:
            u = d[i] - x1 - xu * v;
            if (i != q) {
                v = e[i + 1];
            }
            L580:
            ;
        }

        if (u == 0.) {
            u = eps3;
        }
        rv1[q] = u;
        rv2[q] = 0.;
        rv3[q] = 0.;
/*     :::::::::: back substitution */
/*                for i=q step -1 until p do -- :::::::::: */
        L600:
        i__2 = q;
        for (ii = p; ii <= i__2; ++ii) {
            i = p + q - ii;
            rtem = rv6[i] - u * rv2[i] - v * rv3[i];
            rv6[i] = (rtem) / rv1[i];
            v = u;
            u = rv6[i];
/* L620: */
        }
/*     :::::::::: orthogonalize with respect to previous */
/*                members of group :::::::::: */
        if (group == 0) {
            goto L700;
        }
        j = r;

        i__2 = group;
        for (jj = 1; jj <= i__2; ++jj) {
            L630:
            --j;
            if (ind[j] != tag) {
                goto L630;
            }
            xu = 0.;

            i__3 = q;
            for (i = p; i <= i__3; ++i) {
/* L640: */
                xu += rv6[i] * z[i + j * z_dim1];
            }

            i__3 = q;
            for (i = p; i <= i__3; ++i) {
/* L660: */
                rv6[i] -= xu * z[i + j * z_dim1];
            }

/* L680: */
        }

        L700:
        norm = 0.;

        i__2 = q;
        for (i = p; i <= i__2; ++i) {
/* L720: */
            norm += (d__1 = rv6[i], abs(d__1));
        }

        if (norm >= 1.) {
            goto L840;
        }
/*     :::::::::: forward substitution :::::::::: */
        if (its == 5) {
            goto L830;
        }
        if (norm != 0.) {
            goto L740;
        }
        rv6[s] = eps4;
        ++s;
        if (s > q) {
            s = p;
        }
        goto L780;
        L740:
        xu = eps4 / norm;

        i__2 = q;
        for (i = p; i <= i__2; ++i) {
/* L760: */
            rv6[i] *= xu;
        }
/*     :::::::::: elimination operations on next vector */
/*                iterate :::::::::: */
        L780:
        i__2 = q;
        for (i = ip; i <= i__2; ++i) {
            u = rv6[i];
/*     :::::::::: if rv1(i-1) .eq. e(i), a row interchange */
/*                was performed earlier in the */
/*                triangularization process :::::::::: */
            if (rv1[i - 1] != e[i]) {
                goto L800;
            }
            u = rv6[i - 1];
            rv6[i - 1] = rv6[i];
            L800:
            rv6[i] = u - rv4[i] * rv6[i - 1];
/* L820: */
        }

        ++its;
        goto L600;
/*     :::::::::: set error -- non-converged eigenvector :::::::::: */
        L830:
        *ierr = -r;
        xu = 0.;
        goto L870;
/*     :::::::::: normalize so that sum of squares is */
/*                1 and expand to full order :::::::::: */
        L840:
        u = 0.;

        i__2 = q;
        for (i = p; i <= i__2; ++i) {
/* L860: */
/* Computing 2nd power */
            d__1 = rv6[i];
            u += d__1 * d__1;
        }

        xu = 1. / sqrt(u);

        L870:
        i__2 = *n;
        for (i = 1; i <= i__2; ++i) {
/* L880: */
            z[i + r * z_dim1] = 0.;
        }

        i__2 = q;
        for (i = p; i <= i__2; ++i) {
/* L900: */
            z[i + r * z_dim1] = rv6[i] * xu;
        }

        x0 = x1;
        L920:
        ;
    }

    if (q < *n) {
        goto L100;
    }
    L1001:
    return 0;
/*     :::::::::: last card of tinvit :::::::::: */
} /* tinvit_ */

int dpss::_jtridib(int *n, double *eps1, double *d, double *e, double *e2, double *lb, double *ub, int *m11, int *m,
                   double *w, int *ind, int *ierr, double *rv4, double *rv5) {
    /* Initialized data */

    static double machep = 1.25e-15;

    /* System generated locals */
    int i__1, i__2;
    double d__1, d__2, d__3;

    /* Local variables */
    static int i, j, k, l, p, q, r, s;
    static double u, v;
    static int m1, m2;
    static double t1, t2, x0, x1;
    static int m22, ii;
    static double xu;
    static int isturm, tag;



/*     this subroutine is a translation of the algol procedure bisect, */
/*     num. math. 9, 386-393(1967) by barth, martin, and wilkinson. */
/*     handbook for auto. comp., vol.ii-linear algebra, 249-256(1971). */

/*     this subroutine finds those eigenvalues of a tridiagonal */
/*     symmetric matrix between specified boundary indices, */
/*     using bisection. */

/*     on input: */

/*        n is the order of the matrix; */

/*        eps1 is an absolute error tolerance for the computed */
/*          eigenvalues.  if the input eps1 is non-positive, */
/*          it is reset for each submatrix to a default value, */
/*          namely, minus the product of the relative machine */
/*          precision and the 1-norm of the submatrix; */

/*        d contains the diagonal elements of the input matrix; */

/*        e contains the subdiagonal elements of the input matrix */
/*          in its last n-1 positions.  e(1) is arbitrary; */

/*        e2 contains the squares of the corresponding elements of e. */
/*          e2(1) is arbitrary; */

/*        m11 specifies the lower boundary index for the desired */
/*          eigenvalues; */

/*        m specifies the number of eigenvalues desired.  the upper */
/*          boundary index m22 is then obtained as m22=m11+m-1. */

/*     on output: */

/*        eps1 is unaltered unless it has been reset to its */
/*          (last) default value; */

/*        d and e are unaltered; */

/*        elements of e2, corresponding to elements of e regarded */
/*          as negligible, have been replaced by zero causing the */
/*          matrix to split into a direct sum of submatrices. */
/*          e2(1) is also set to zero; */

/*        lb and ub define an interval containing exactly the desired */
/*          eigenvalues; */

/*        w contains, in its first m positions, the eigenvalues */
/*          between indices m11 and m22 in ascending order; */

/*        ind contains in its first m positions the submatrix indices */
/*          associated with the corresponding eigenvalues in w -- */
/*          1 for eigenvalues belonging to the first submatrix from */
/*          the top, 2 for those belonging to the second submatrix, etc.;
*/

/*        ierr is set to */
/*          zero       for normal return, */
/*          3*n+1      if multiple eigenvalues at index m11 make */
/*                     unique selection impossible, */
/*          3*n+2      if multiple eigenvalues at index m22 make */
/*                     unique selection impossible; */

/*        rv4 and rv5 are temporary storage arrays. */

/*     note that subroutine tql1, imtql1, or tqlrat is generally faster */
/*     than tridib, if more than n/4 eigenvalues are to be found. */

/*     questions and comments should be directed to b. s. garbow, */
/*     applied mathematics division, argonne national laboratory */

/*     ------------------------------------------------------------------
*/

/*     :::::::::: machep is a machine dependent parameter specifying */
/*                the relative precision of floating point arithmetic. */
/*                machep = 16.0d0**(-13) for long form arithmetic */
/*                on s360 :::::::::: */
/*  for f_floating dec fortran */
/*      data machep/1.1d-16/ */
/*  for g_floating dec fortran */
    /* Parameter adjustments */
    --rv5;
    --rv4;
    --e2;
    --e;
    --d;
    --ind;
    --w;

    /* Function Body */

    *ierr = 0;
    tag = 0;
    xu = d[1];
    x0 = d[1];
    u = 0.;
/*     :::::::::: look for small sub-diagonal entries and determine an */
/*                interval containing all the eigenvalues :::::::::: */
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
        x1 = u;
        u = 0.;
        if (i != *n) {
            u = (d__1 = e[i + 1], abs(d__1));
        }
/* Computing MIN */
        d__1 = d[i] - (x1 + u);
        xu = std::min(d__1,xu);
/* Computing MAX */
        d__1 = d[i] + (x1 + u);
        x0 = std::max(d__1,x0);
        if (i == 1) {
            goto L20;
        }
        if ((d__1 = e[i], abs(d__1)) > machep * ((d__2 = d[i], abs(d__2)) + (
                d__3 = d[i - 1], abs(d__3)))) {
            goto L40;
        }
        L20:
        e2[i] = 0.;
        L40:
        ;
    }

/* Computing MAX */
    d__1 = abs(xu), d__2 = abs(x0);
    x1 = std::max(d__1,d__2) * machep * (double) (*n);
    xu -= x1;
    t1 = xu;
    x0 += x1;
    t2 = x0;
/*     :::::::::: determine an interval containing exactly */
/*                the desired eigenvalues :::::::::: */
    p = 1;
    q = *n;
    m1 = *m11 - 1;
    if (m1 == 0) {
        goto L75;
    }
    isturm = 1;
    L50:
    v = x1;
    x1 = xu + (x0 - xu) * .5;
    if (x1 == v) {
        goto L980;
    }
    goto L320;
    L60:
    if ((i__1 = s - m1) < 0) {
        goto L65;
    } else if (i__1 == 0) {
        goto L73;
    } else {
        goto L70;
    }
    L65:
    xu = x1;
    goto L50;
    L70:
    x0 = x1;
    goto L50;
    L73:
    xu = x1;
    t1 = x1;
    L75:
    m22 = m1 + *m;
    if (m22 == *n) {
        goto L90;
    }
    x0 = t2;
    isturm = 2;
    goto L50;
    L80:
    if ((i__1 = s - m22) < 0) {
        goto L65;
    } else if (i__1 == 0) {
        goto L85;
    } else {
        goto L70;
    }
    L85:
    t2 = x1;
    L90:
    q = 0;
    r = 0;
/*     :::::::::: establish and process next submatrix, refining */
/*                interval by the gerschgorin bounds :::::::::: */
    L100:
    if (r == *m) {
        goto L1001;
    }
    ++tag;
    p = q + 1;
    xu = d[p];
    x0 = d[p];
    u = 0.;

    i__1 = *n;
    for (q = p; q <= i__1; ++q) {
        x1 = u;
        u = 0.;
        v = 0.;
        if (q == *n) {
            goto L110;
        }
        u = (d__1 = e[q + 1], abs(d__1));
        v = e2[q + 1];
        L110:
/* Computing MIN */
        d__1 = d[q] - (x1 + u);
        xu = std::min(d__1,xu);
/* Computing MAX */
        d__1 = d[q] + (x1 + u);
        x0 = std::max(d__1,x0);
        if (v == 0.) {
            goto L140;
        }
/* L120: */
    }

    L140:
/* Computing MAX */
    d__1 = abs(xu), d__2 = abs(x0);
    x1 = std::max(d__1,d__2) * machep;
    if (*eps1 <= 0.) {
        *eps1 = -x1;
    }
    if (p != q) {
        goto L180;
    }
/*     :::::::::: check for isolated root within interval :::::::::: */
    if (t1 > d[p] || d[p] >= t2) {
        goto L940;
    }
    m1 = p;
    m2 = p;
    rv5[p] = d[p];
    goto L900;
    L180:
    x1 *= (double) (q - p + 1);
/* Computing MAX */
    d__1 = t1, d__2 = xu - x1;
    *lb = std::max(d__1,d__2);
/* Computing MIN */
    d__1 = t2, d__2 = x0 + x1;
    *ub = std::min(d__1,d__2);
    x1 = *lb;
    isturm = 3;
    goto L320;
    L200:
    m1 = s + 1;
    x1 = *ub;
    isturm = 4;
    goto L320;
    L220:
    m2 = s;
    if (m1 > m2) {
        goto L940;
    }
/*     :::::::::: find roots by bisection :::::::::: */
    x0 = *ub;
    isturm = 5;

    i__1 = m2;
    for (i = m1; i <= i__1; ++i) {
        rv5[i] = *ub;
        rv4[i] = *lb;
/* L240: */
    }
/*     :::::::::: loop for k-th eigenvalue */
/*                for k=m2 step -1 until m1 do -- */
/*                (-do- not used to legalize -computed go to-) ::::::::::
*/
    k = m2;
    L250:
    xu = *lb;
/*     :::::::::: for i=k step -1 until m1 do -- :::::::::: */
    i__1 = k;
    for (ii = m1; ii <= i__1; ++ii) {
        i = m1 + k - ii;
        if (xu >= rv4[i]) {
            goto L260;
        }
        xu = rv4[i];
        goto L280;
        L260:
        ;
    }

    L280:
    if (x0 > rv5[k]) {
        x0 = rv5[k];
    }
/*     :::::::::: next bisection step :::::::::: */
    L300:
    x1 = (xu + x0) * .5;
    if (x0 - xu <= machep * 2. * (abs(xu) + abs(x0)) + abs(*eps1)) {
        goto L420;
    }
/*     :::::::::: in-line procedure for sturm sequence :::::::::: */
    L320:
    s = p - 1;
    u = 1.;

    i__1 = q;
    for (i = p; i <= i__1; ++i) {
        if (u != 0.) {
            goto L325;
        }
        v = (d__1 = e[i], abs(d__1)) / machep;
        if (e2[i] == 0.) {
            v = 0.;
        }
        goto L330;
        L325:
        v = e2[i] / u;
        L330:
        u = d[i] - x1 - v;
        if (u < 0.) {
            ++s;
        }
/* L340: */
    }

    switch ((int)isturm) {
        case 1:  goto L60;
        case 2:  goto L80;
        case 3:  goto L200;
        case 4:  goto L220;
        case 5:  goto L360;
    }
/*     :::::::::: refine intervals :::::::::: */
    L360:
    if (s >= k) {
        goto L400;
    }
    xu = x1;
    if (s >= m1) {
        goto L380;
    }
    rv4[m1] = x1;
    goto L300;
    L380:
    rv4[s + 1] = x1;
    if (rv5[s] > x1) {
        rv5[s] = x1;
    }
    goto L300;
    L400:
    x0 = x1;
    goto L300;
/*     :::::::::: k-th eigenvalue found :::::::::: */
    L420:
    rv5[k] = x1;
    --k;
    if (k >= m1) {
        goto L250;
    }
/*     :::::::::: order eigenvalues tagged with their */
/*                submatrix associations :::::::::: */
    L900:
    s = r;
    r = r + m2 - m1 + 1;
    j = 1;
    k = m1;

    i__1 = r;
    for (l = 1; l <= i__1; ++l) {
        if (j > s) {
            goto L910;
        }
        if (k > m2) {
            goto L940;
        }
        if (rv5[k] >= w[l]) {
            goto L915;
        }

        i__2 = s;
        for (ii = j; ii <= i__2; ++ii) {
            i = l + s - ii;
            w[i + 1] = w[i];
            ind[i + 1] = ind[i];
/* L905: */
        }

        L910:
        w[l] = rv5[k];
        ind[l] = tag;
        ++k;
        goto L920;
        L915:
        ++j;
        L920:
        ;
    }

    L940:
    if (q < *n) {
        goto L100;
    }
    goto L1001;
/*     :::::::::: set error -- interval cannot be found containing */
/*                exactly the desired eigenvalues :::::::::: */
    L980:
    *ierr = *n * 3 + isturm;
    L1001:
    *lb = t1;
    *ub = t2;
    return 0;
/*     :::::::::: last card of tridib :::::::::: */
}


/* tridib_ */