//
// Created by heyanbai on 2021/9/19.
//

#ifndef GENERIC_PSE_UTILS_H
#define GENERIC_PSE_UTILS_H

#include <string>
#include <cstdint>
#include <cmath>
#include "yaml-cpp/yaml.h"
#include <armadillo>

namespace utils {
    namespace helper {
        bool len_guards(const int16_t &M);

        int16_t extend(const int16_t &M, const bool &sym, bool &needs_trunc);

        // TODO: implement
//        arma::cx_dmat fftautocorr(const arma::cx_dmat & x);

        void eigh_tridiagonal(const arma::dvec &d, const arma::dvec &e, arma::dvec &w,
                              arma::dmat &v, const std::pair<uint16_t, uint16_t> &select_range);

        void dstebz(int *N, double *VL, double *VU,
                    int *IL, int *IU, double *ABSTOL, double *D, double *E,
                    int *M, double *W, int *IBLOCK, int *ISPLIT);

        void dstein(int *N, double *D, double *E, int *M, double *W, int *IBLOCK, int *ISPLIT, double *Z, int *LDZ);
    }
    namespace multitaper {
        // TODO: return_ratios
        arma::dmat dpss(int16_t M, double NW, int16_t Kmax = -1, bool sym = true, std::string norm = "", bool return_ratios = false);
    }

    void getArmaMat2Txt(const arma::dmat & src, const std::string & pathAndName);
    void getArmaVec2Txt(const arma::dvec & src, const std::string & pathAndName);
};

#endif //GENERIC_PSE_UTILS_H
