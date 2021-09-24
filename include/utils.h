//
// Created by heyanbai on 2021/9/19.
//

#ifndef GENERIC_PSE_UTILS_H
#define GENERIC_PSE_UTILS_H

#include <iostream>

#include <memory>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "gtest/gtest.h"
#include <cstdint>
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "kfr/dft.hpp"
#include "yaml-cpp/yaml.h"

class utils {
private:
    static bool _len_guards(const int16_t & M);
    static int16_t _extend(const int16_t & M, const bool & sym, bool & needs_trunc);
    Eigen::MatrixXcd _fftautocorr(const Eigen::MatrixXcd & x);
    static void _eigh_tridiagonal(const Eigen::VectorXd &d, const Eigen::VectorXd &e, Eigen::VectorXd &w,
                                         Eigen::MatrixXd &v, const std::pair<uint16_t, uint16_t> & select_range);
    static void _dstebz(int *N, double *VL, double *VU,
                        int *IL, int *IU, double *ABSTOL, double *D, double *E,
                        int *M, double *W, int *IBLOCK, int *ISPLIT);
    static void _dstein(int *N, double *D, double *E, int *M, double *W, int *IBLOCK, int *ISPLIT, double *Z, int *LDZ);

public:
    // TODO: return_ratios
    static Eigen::MatrixXd dpss(int16_t M, int16_t NW, int16_t Kmax = -1, bool sym = true, std::string norm = "", bool return_ratios = false);
    static Eigen::MatrixXd dpsschk(const int & N, const YAML::Node & mts_args);
    static void getEigentoData(Eigen::MatrixXd& src, char* pathAndName);
};


#endif //GENERIC_PSE_UTILS_H
