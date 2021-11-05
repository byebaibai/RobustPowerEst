//
// Created by heyanbai on 2021/9/29.
//

#include "spectrum_estimator.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include <vector>

arma::dvec spectrum_estimator::computeChi2Ppf_(const arma::dvec & svals, const double & ndegs){
    std::vector<double> svalsVector = arma::conv_to<std::vector<double>>::from(svals);
    py::array_t<double> svalsNp = py::cast(svalsVector);
    py::array_t<double> ppfRes = chi2Ppf_(svalsNp, ndegs);
    double* chi2PpfRes = static_cast<double*>(ppfRes.request().ptr);

    arma::dvec res(chi2PpfRes, svals.size());

    return res;
}

double spectrum_estimator::computeBeta_(const double & a, const double & b){
    return beta_(a, b).cast<double>();
}

spectrum_estimator::spectrum_estimator(YAML::Node method_args, YAML::Node mts_args){

    scipySpecial_ = py::module::import("scipy.special");
    scipyDistribution_ = py::module::import("scipy.stats.distributions");
    beta_ = scipySpecial_.attr("beta");
    chi2Ppf_ = scipyDistribution_.attr("chi2").attr("ppf");

    if (!mts_args["trialave"]){
        mts_args["trialave"] = 1;
        trialave_ = 1;
    }else{
        trialave_ = mts_args["trialave"].as<int16_t>();
    }

    if(((method_args["class"].as<std::string>() == "two-tier" &&
            method_args["tier"][0]["estimator"].as<std::string>() == "mean" &&
            (method_args["tier"][1]["estimator"].as<std::string>() == "mean" ||
                    method_args["tier"][1]["estimator"].as<std::string>() == "quantile"))) ||
            (method_args["tier"][1]["estimator"].as<std::string>() == "mean" ||
             method_args["tier"][1]["estimator"].as<std::string>() == "median") ){
        scalefactor_calculation_ = ScaleFactorCalcType::kAnalytic;
    }else{
        scalefactor_calculation_ = ScaleFactorCalcType::kMonteCarlo;
    }

    if (mts_args["tapers"]["nw"] && mts_args["tapers"]["kmax"]){
        tapers_ = std::make_pair(mts_args["tapers"]["nw"].as<double>(),
                mts_args["tapers"]["kmax"].as<int16_t>());
    }else{
        tapers_ = std::make_pair(3.0, 5);
    }

    if (mts_args["pad"]){
        pad_ = mts_args["pad"].as<int16_t>();
    }else{
        pad_ = 0;
    }

    if (mts_args["Fs"]){
        sample_rate_ = mts_args["Fs"].as<int16_t>();
    }else{
        sample_rate_ = 1;
    }

    if (mts_args["fpass"][0] && mts_args["fpass"][1]){
        fpass_ = std::make_pair(mts_args["fpass"][0].as<double>(), mts_args["fpass"][1].as<double>());
    }else{
        fpass_ = std::make_pair(0.0, double(sample_rate_)/2.0);
    }

    if (mts_args["err"][0] && mts_args["err"][1]){
        err_ = std::make_pair(mts_args["err"][0].as<double>(), mts_args["err"][1].as<double>());
    }else{
        err_ = std::make_pair(0.0, 0.0);
    }

    this->method_args_ = method_args;
    this->mts_args_ = mts_args;

}

spectrum_estimator::FreqGrid spectrum_estimator::getFreqGrid_(const uint16_t & n_fft){
    double df = double(sample_rate_) / double(n_fft);
    arma::dvec freqs = arma::regspace(0, 1, ceil(double(sample_rate_)/df)) * df;
    freqs = freqs.subvec(0, n_fft);

    arma::uvec findx = arma::find( freqs >= fpass_.first && freqs <= fpass_.second );
    freqs = freqs(findx);

    return spectrum_estimator::FreqGrid(freqs, findx);
}

arma::dmat spectrum_estimator::dpsschk_(const uint16_t & N){
    arma::dmat tapers = utils::multitaper::dpss(N, tapers_.first, tapers_.second);
    tapers = tapers * sqrt(sample_rate_);
    return tapers.t();
}

arma::dcube spectrum_estimator::permute_(const arma::dcube & data){
    arma::uword n_rows = data.n_rows, n_cols = data.n_cols, n_slices = data.n_slices;
    arma::dcube result(n_rows, n_slices, n_cols);
    const double *basePtr = data.memptr();
    double *resPtr = result.memptr();

    for(arma::uword i = 0; i < n_slices; ++i){
        double *tmpPtr = resPtr + n_rows * i;
        for(arma::uword j = 0; j < n_cols; ++j){
            memcpy(tmpPtr, basePtr, n_rows * sizeof(*basePtr));
            basePtr = basePtr + n_rows;
            tmpPtr = tmpPtr + n_rows * n_slices;
        }
    }

    return result;
}

arma::cx_dcube spectrum_estimator::mtfftc_(const arma::dcube & data, const arma::dmat & tapers, const uint16_t & n_fft){
    uint16_t NC = data.n_cols, C = data.n_rows;
    uint16_t NK = tapers.n_rows, K = tapers.n_cols;

    if( NK != NC ){
        throw std::runtime_error("length of tapers is incompatible with length of data");
    }

    arma::dcube tapers4mul(NK, K, C);
    for(uint16_t i = 0; i < C; ++i){
        tapers4mul.slice(i) = tapers;
    }
    arma::dcube data4temp(NC, C, K);
    for (uint16_t i = 0; i < K; ++i){
        data4temp.slice(i) = data.slice(0).t();
    }
    arma::dcube data4mul = permute_(data4temp);

    arma::dcube data2proj = data4mul % tapers4mul;
    arma::cx_dcube result(n_fft, K, C);

    for(uint16_t i = 0; i < C; ++i){
        result.slice(i) = arma::fft(data2proj.slice(i), n_fft);
    }

    return result/double(sample_rate_);
}

spectrum_estimator::Result spectrum_estimator::compute(const arma::dcube & data){
    uint16_t n_pnts = data.n_cols;
    uint16_t n_fft = std::max(uint16_t(pow(2, int16_t(ceil(log2(n_pnts))))), n_pnts);

    FreqGrid freqGrid = getFreqGrid_(n_fft);
    arma::dmat tapers = dpsschk_(n_pnts);
    arma::cx_dcube J = mtfftc_(data, tapers, n_fft);
    arma::cx_dcube J_tmp(freqGrid.freqidxs.n_elem, J.n_cols, J.n_slices);

    for( uint16_t i = 0; i < J.n_slices; ++i ){
        J_tmp.slice(i) = J.slice(i).rows(freqGrid.freqidxs.front(), freqGrid.freqidxs.back());
    }

    scaleFactorCompute_(J_tmp);

    arma::dcube J_est = arma::conv_to<arma::dcube>::from(J_tmp % arma::conj(J_tmp));

    arma::dvec S = taperedEstimate_(J_est, method_args_, EstimatorCalcType::kSpectrum, trialave_);

    return Result(S, freqGrid.freqs);
}

double spectrum_estimator::analyticalScaleFactorCompute_(uint16_t ntapers, uint16_t ntrials, double h){
    double ds = 1e-5;

    arma::dvec svals = arma::regspace(0.0, ds, 1.0);
    svals = svals.subvec(1, svals.size() - 1);
    arma::dvec chiinv_vals = computeChi2Ppf_(svals, ntapers);

    uint16_t N = ntrials - 1;
    double k = std::min(std::max(0.0, h * double(N + 1) - 0.5), double(N));
    double f_analytic = 0;
    if ( fabs(floor(k) - k) < 1e-5 ){
        f_analytic = (1.0/computeBeta_(k+1,N-k+1)) * ds *
                arma::sum(arma::pow(svals, k) % (arma::pow(1 - svals, N-k) % chiinv_vals)) / ntapers;
    }else{
        double k_lo = floor(k);
        double k_hi = k_lo + 1;
        double f_lo = (1.0/computeBeta_(k_lo+1,N-k_lo+1)) * ds * arma::sum(arma::pow(svals, k_lo) % (arma::pow(1 - svals, N-k_lo) % chiinv_vals)) / ntapers;
        double f_hi = (1.0/computeBeta_(k_hi+1,N-k_hi+1)) * ds * arma::sum(arma::pow(svals, k_hi) % (arma::pow(1 - svals, N-k_hi) % chiinv_vals)) / ntapers;

        f_analytic = (k_hi - k) * f_lo + (k - k_lo) * f_hi;
    }
    return f_analytic;
}

void spectrum_estimator::scaleFactorCompute_(const arma::cx_dcube & data){
    uint16_t nfreqs = data.n_rows, ntapers = data.n_cols, ntrials = data.n_slices;
    arma::cube dims_real(1, ntapers, ntrials);
    arma::cube dims_imag(1, ntapers, ntrials);

    method_args_["scalefactor"]["spectrum"] = std::make_pair(1.0, 1.0);
    method_args_["scalefactor"]["error"] = std::make_pair(1.0, 1.0);

    std::pair<double, double> calib_ratio_s, calib_ratio_e;

    if(method_args_["class"].as<std::string>() == "standard"){
        calib_ratio_s = std::make_pair(1.0, 1.0);
        calib_ratio_e = std::make_pair(1.0, 1.0);
        scalefactor_calculation_ = ScaleFactorCalcType::kNone;
    }else if( scalefactor_calculation_ == ScaleFactorCalcType::kAnalytic &&
            ((method_args_["tier"][0]["estimator"].as<std::string>() != "mean") ||
                (method_args_["tier"][1]["estimator"].as<std::string>() == "mean")) ){
        fmt::print("Defaulting to monte-carlo scale factor.");
        scalefactor_calculation_ = ScaleFactorCalcType::kMonteCarlo;
    }
    if ( scalefactor_calculation_ == ScaleFactorCalcType::kAnalytic ){

        double h = 0.5;
        if ( method_args_["class"].as<std::string>() == "one-tier" ){
            if ( method_args_["tier"][0]["estimator"].as<std::string>() == "median" ){
                h = 0.5;
            }else if ( method_args_["tier"][0]["estimator"].as<std::string>() == "quantile" ){
                h = method_args_["tier"][0]["h"].as<double>();
            }
            ntrials = ntrials * ntapers;
            ntapers = 1;
        } else if ( method_args_["tier"][1]["estimator"].as<std::string>() == "median" ){
            h = 0.5;
        } else if ( method_args_["tier"][1]["estimator"].as<std::string>() == "quantile" ){
            h = method_args_["tier"][1]["h"].as<double>();
        }

        double s_dc = analyticalScaleFactorCompute_(1*ntapers, ntrials, h);
        double s_el = analyticalScaleFactorCompute_(2*ntapers, ntrials, h);
        double e_dc = analyticalScaleFactorCompute_(1, ntapers*ntrials, h);
        double e_el = analyticalScaleFactorCompute_(1, 2*ntapers*ntrials, h);;

        calib_ratio_s = std::make_pair(s_dc, s_el);
        calib_ratio_e = std::make_pair(e_dc, e_el);
    }else{
        uint32_t nruns = 1000;
        arma::cx_dmat simruns_nrobust_s(2, nruns, arma::fill::zeros);
        arma::cx_dmat simruns_robust_s(2, nruns, arma::fill::zeros);
        arma::cx_dmat simruns_nrobust_e(2, nruns, arma::fill::zeros);
        arma::cx_dmat simruns_robust_e(2, nruns, arma::fill::zeros);

        arma::field<arma::cx_dcube> simJ(nruns);
        for (uint32_t ri = 0; ri < nruns; ++ri){
            simJ(ri) = arma::zeros<arma::cx_dcube>(2, ntapers, ntrials);
            simJ(ri).row(0) += 1i * arma::zeros<arma::cx_dcube>(1, ntapers, ntrials);
            simJ(ri).row(1) = arma::randn<arma::cx_dcube>(1, ntapers, ntrials) + 1i * arma::randn<arma::cx_dcube>(1, ntapers, ntrials);
            simJ(ri) = simJ(ri) % arma::conj(simJ(ri));

        }
    }

    std::vector<double> calibRatioS = std::vector<double>(nfreqs, calib_ratio_s.second);
    std::vector<double> calibRatioE = std::vector<double>(nfreqs, calib_ratio_e.second);
    calibRatioS.front() = calib_ratio_s.first;
    calibRatioS.front() = calib_ratio_s.first;
    calibRatioE.back() = calib_ratio_e.first;
    calibRatioE.back() = calib_ratio_e.first;

    method_args_["scalefactor"]["spectrum"] = calibRatioS;
    method_args_["scalefactor"]["error"] = calibRatioE;

}
arma::dmat spectrum_estimator::estimate_(const arma::dcube & data, const std::string & tiermethod){
    return arma::mean(data, 1);
}
arma::dvec spectrum_estimator::estimate_(const arma::dmat & data, const std::string & tiermethod, const double & h){
    if( tiermethod == "mean" ){
        return arma::mean(data, 1);
    } else if ( tiermethod == "median" ){
        return arma::median(data, 1);
    } else if ( tiermethod == "quantile" ){
        arma::dvec hVector = {h};
        return arma::quantile(data, hVector, 1);
    }
    return arma::mean(data, 1);
}
arma::dvec spectrum_estimator::taperedEstimate_(const arma::dcube & data, const YAML::Node & method,
                                const spectrum_estimator::EstimatorCalcType & calcType, uint16_t trialave){
    uint16_t nfreqs = data.n_rows, ntapers = data.n_cols, ntrials = data.n_slices;
    arma::dmat avgOverTapers;
    arma::dvec avgOverTrials;

    if(method["class"].as<std::string>() == "standard"){
        avgOverTapers = arma::mean(data, 1);
        if(trialave && calcType == spectrum_estimator::EstimatorCalcType::kSpectrum){
            avgOverTrials = arma::mean(avgOverTapers, 1);
        }
    } else if(method["class"].as<std::string>() == "two-tier"){
        arma::dvec scalefact(method["scalefactor"][toString_(calcType)].as<std::vector<double>>());
        if(calcType == spectrum_estimator::EstimatorCalcType::kSpectrum){
            avgOverTapers = estimate_(data, method["tier"][0]["estimator"].as<std::string>());
            if( trialave ){
                avgOverTrials = estimate_(avgOverTapers, method["tier"][1]["estimator"].as<std::string>(), method["tier"][1]["h"].as<double>());
                avgOverTrials = avgOverTrials / scalefact;
            }else{
                avgOverTapers = avgOverTapers.each_col() / scalefact;
            }
        }else if(calcType == spectrum_estimator::EstimatorCalcType::kError){
            if( trialave ){
                avgOverTapers = arma::dmat(const_cast<double*>(data.memptr()) , nfreqs, ntapers*ntrials, false, true);
                avgOverTrials = estimate_(avgOverTapers, method["tier"][1]["estimator"].as<std::string>(), method["tier"][1]["h"].as<double>());
                avgOverTrials = avgOverTrials / scalefact;
            }else{
                avgOverTapers = estimate_(data, method["tier"][0]["estimator"].as<std::string>());
                avgOverTapers = avgOverTapers.each_col() / scalefact;
            }
        }
    } else if(method["class"].as<std::string>() == "one-tier"){
        arma::dvec scalefact(method["scalefactor"][toString_(calcType)].as<std::vector<double>>());
        if(trialave){
            avgOverTapers = arma::dmat(const_cast<double*>(data.memptr()) , nfreqs, ntapers*ntrials, false, true);
            avgOverTrials = estimate_(data, method["tier"][0]["estimator"].as<std::string>());
            avgOverTrials = avgOverTrials / scalefact;
        }else{
            avgOverTapers = estimate_(data, method["tier"][0]["estimator"].as<std::string>());
            avgOverTapers = avgOverTapers.each_col() / scalefact;
        }
    }
    return avgOverTrials;
}
