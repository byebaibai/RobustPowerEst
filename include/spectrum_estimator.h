//
// Created by heyanbai on 2021/9/29.
//

#ifndef GENERIC_PSE_SPECTRUM_ESTIMATOR_H
#define GENERIC_PSE_SPECTRUM_ESTIMATOR_H

#include "yaml-cpp/yaml.h"
#include <armadillo>
#include <cstdint>
#include "utils.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class spectrum_estimator {

    enum ScaleFactorCalcType{
        kAnalytic,
        kMonteCarlo,
        kNone
    };

    enum EstimatorCalcType{
        kSpectrum,
        kError
    };

    struct FreqGrid {
        arma::dvec freqs;
        arma::uvec freqidxs;

        FreqGrid(arma::dvec f, arma::uvec findx):freqs(f), freqidxs(findx){};
    };

private:
    YAML::Node method_args_;
    YAML::Node mts_args_;
    ScaleFactorCalcType scalefactor_calculation_;

    std::pair<double, double> fpass_;
    std::pair<double, double> err_;
    std::pair<double, int16_t> tapers_;
    int16_t trialave_{};
    int16_t pad_;
    int16_t sample_rate_;

    py::scoped_interpreter m_guard_;
    py::module scipySpecial_;
    py::module scipyDistribution_;
    py::function chi2Ppf_;
    py::function beta_;


    FreqGrid getFreqGrid_(const uint16_t & n_fft);
    arma::dmat dpsschk_(const uint16_t & N);
    arma::cx_dcube mtfftc_(const arma::dcube & data, const arma::dmat & tapers, const uint16_t & n_fft);
    arma::dcube permute_(const arma::dcube & data);
    void scaleFactorCompute_(const arma::cx_dcube & data);
    double analyticalScaleFactorCompute_(uint16_t ntapers, uint16_t ntrials, double h);

    arma::dvec computeChi2Ppf_(const arma::dvec & svals, const double & ndegs);

    double computeBeta_(const double & a, const double & b);

    arma::dvec taperedEstimate_(const arma::dcube & data, const YAML::Node & method, const EstimatorCalcType & calcType, uint16_t trialave = 1);
    arma::dmat estimate_(const arma::dcube & data, const std::string & tiermethod);
    arma::dvec estimate_(const arma::dmat & data, const std::string & tiermethod, const double & h = 0.5);

    std::string toString_(EstimatorCalcType calcType){
        static std::vector<std::string> table = {"spectrum", "error"};
        return table[calcType];
    }
public:
    struct Result {
        arma::dvec spectrum;
        arma::dvec freqs;
        arma::dmat spectrumErr;
        Result(arma::dvec S, arma::dvec f, arma::dmat Serr):spectrum(S), freqs(f), spectrumErr(Serr){};
        Result(arma::dvec S, arma::dvec f):spectrum(S), freqs(f){};
        Result(){};
    };

    spectrum_estimator(YAML::Node method_args, YAML::Node mts_args);

    Result compute(const arma::dcube & data);

};


#endif //GENERIC_PSE_SPECTRUM_ESTIMATOR_H
