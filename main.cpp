#include "utils.h"
#include "spectrum_estimator.h"

int main()
{
    YAML::Node mts_args = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/multitaper.yaml");
    YAML::Node robust_args = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/robust.yaml");
    YAML::Node standard_args = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/standard.yaml");

    arma::dmat tapers = utils::multitaper::dpss(750, 5, 8, true, "subsample");

    std::string hdf5Filename = "/mnt/c/Users/heyanbai/CLionProjects/generic-pse/sample/HumanDemoData.h5";
    arma::dcube clean100ch16, clean75ch16, clean50ch16;
    clean100ch16.load(arma::hdf5_name(hdf5Filename, "clean100ch16"));
    clean75ch16.load(arma::hdf5_name(hdf5Filename, "clean75ch16"));
    clean50ch16.load(arma::hdf5_name(hdf5Filename, "clean50ch16"));

    spectrum_estimator robustEstimator(robust_args, mts_args);
//    spectrum_estimator standardEstimator(standard_args, mts_args);

    spectrum_estimator::Result robust100Result = robustEstimator.compute(clean100ch16);
    spectrum_estimator::Result robust75Result = robustEstimator.compute(clean75ch16);
    spectrum_estimator::Result robust50Result = robustEstimator.compute(clean50ch16);

//    spectrum_estimator::Result standard100Result = standardEstimator.compute(clean100ch16);
//    spectrum_estimator::Result standard75Result = standardEstimator.compute(clean75ch16);
//    spectrum_estimator::Result standard50Result = standardEstimator.compute(clean50ch16);
//    std::cout<<clean100ch16.n_rows << " " << clean100ch16.n_cols << " " << clean100ch16.n_slices << std::endl;

    utils::getArmaMat2Txt(tapers, "./tapers.txt");
    utils::getArmaVec2Txt(robust100Result.spectrum, "./r100.txt");
    utils::getArmaVec2Txt(robust75Result.spectrum, "./r75.txt");
    utils::getArmaVec2Txt(robust50Result.spectrum, "./r50.txt");
//    utils::getArmaVec2Txt(standard100Result.spectrum, "./s100.txt");
//    utils::getArmaVec2Txt(standard75Result.spectrum, "./s75.txt");
//    utils::getArmaVec2Txt(standard50Result.spectrum, "./s50.txt");
//    utils::getArmaVec2Txt(standard50Result.freqs, "./freqs.txt");
}
