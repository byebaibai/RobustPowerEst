//
// Created by heyanbai on 2021/9/12.
//

#ifndef GENERIC_PSE_DPSS_H
#define GENERIC_PSE_DPSS_H

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include "gtest/gtest.h"


class dpss {
private:
    uint16_t _seq_length;
    double _time_halfbandwidth;
    uint16_t _num_seq;

    Eigen::MatrixXd _tapers;
    Eigen::VectorXd _eigvals;
    Eigen::VectorXd _tapersum;

    bool _compute(int num_points, int nwin, double *lam, float npi, double *tapers, double *tapsum);

    // helper functions
    int _jtinvit(int *nm, int *n, double *d, double *e, double *e2,
                 int *m, double *w, int *ind,double *z, int *ierr, double *rv1, double *rv2,
                     double *rv3, double *rv4, double *rv6);

    int _jtridib(int *n, double *eps1, double *d, double *e, double *e2, double *lb, double *ub, int *m11, int *m, double *w, int *ind, int *ierr,
                 double *rv4, double *rv5);


public:
    dpss(uint16_t seq_length, double time_halfbandwidth, uint16_t num_seq = 0);
};


#endif //GENERIC_PSE_DPSS_H
