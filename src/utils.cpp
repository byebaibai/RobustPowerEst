//
// Created by heyanbai on 2021/9/19.
//

#include "utils.h"

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>

extern "C" {
    void dstebz_(char *RANGE, char *ORDER, int *N, double *VL, double *VU,
                 int *IL, int *IU, double *ABSTOL, double *D, double *E, int *M,
                 int *NSPLIT, double *W, int *IBLOCK, int *ISPLIT, double *WORK, int *IWORK, int *INFO);

    void dstein_(int *N, double *D, double *E, int *M, double *W, int *IBLOCK, int *ISPLIT, double *Z,
                 int *LDZ, double *WORK, int *IWORK, int *IFAIL, int *INFO);
}

void utils::helper::dstebz(int *N, double *VL, double *VU,
                     int *IL, int *IU, double *ABSTOL, double *D, double *E,
                     int *M, double *W, int *IBLOCK, int *ISPLIT){

    int INFO, NSPLIT;
    char range = 'I', order = 'B';
    std::shared_ptr<double> WORK(new double[4*(*N)], std::default_delete<double[]>());
    std::shared_ptr<int> IWORK(new int[3*(*N)], std::default_delete<int[]>());
    dstebz_(&range, &order, N, VL, VU, IL, IU, ABSTOL,
            D, E, M, &NSPLIT, W, IBLOCK, ISPLIT, WORK.get(), IWORK.get(), &INFO);
    if (INFO != 0) {
        if ( INFO < 0 ){
            printf("%d--th argument had an illegal value\n", -INFO);
        } else {
            switch (INFO){
                case 4:
                    printf("the Gershgorin interval initially used was too small. No eigenvalues were computed.\n");
                    break;
                case 2:
                    printf("Not all of the eigenvalues IL:IU were found..\n");
                    break;
                case 1:
                    printf("Bisection failed to converge for some eigenvalues\n");
                    break;
                case 3:
                    printf("Not all of the eigenvalues IL:IU were found..\n");
                    printf("Bisection failed to converge for some eigenvalues\n");
                    break;
            }
        }
    }
}
void utils::helper::dstein(int *N, double *D, double *E, int *M, double *W, int *IBLOCK, int *ISPLIT, double *Z, int *LDZ){
    int INFO;
    std::shared_ptr<double> WORK(new double[5*(*N)], std::default_delete<double[]>());
    std::shared_ptr<int> IWORK(new int[*N], std::default_delete<int[]>());
    std::shared_ptr<int> IFAIL(new int[*M], std::default_delete<int[]>());
    dstein_(N, D, E, M, W, IBLOCK, ISPLIT, Z, LDZ, WORK.get(), IWORK.get(), IFAIL.get(), &INFO);
    if (INFO != 0) {
        if ( INFO < 0 ){
            printf("%d--th argument had an illegal value\n", -INFO);
        } else {
            printf("%d eigenvectors failed to converge in MAXITS iterations.\n", -INFO);
        }
    }
}

bool utils::helper::len_guards(const int16_t &M) {
    if ( M < 0 ){
        throw std::runtime_error("Window length M must be a non-negative integer");
    }
    return M <= 1;
}

int16_t utils::helper::extend(const int16_t & M, const bool & sym, bool & needs_trunc) {
    if ( !sym ){
        needs_trunc = true;
        return M + 1;
    }else{
        needs_trunc = false;
        return M;
    }
}

void utils::helper::eigh_tridiagonal(const arma::dvec &d, const arma::dvec &e, arma::dvec &w, arma::dmat &v,
                              const std::pair<uint16_t, uint16_t> &select_range) {
    int N = int(d.size());
    double vl = 0.0, vu = 1.0, tol = 0.0;
    int il = select_range.first + 1, iu = select_range.second + 1, M;
    std::shared_ptr<double> W(new double[N], std::default_delete<double[]>());
    std::shared_ptr<int> IBLOCK(new int[N], std::default_delete<int[]>());
    std::shared_ptr<int> ISPLIT(new int[N], std::default_delete<int[]>());
    auto D = const_cast<double *>(d.memptr());
    auto E = const_cast<double *>(e.memptr());

    dstebz(&N, &vl, &vu, &il, &iu, &tol, D, E, &M, W.get(), IBLOCK.get(), ISPLIT.get());

    std::shared_ptr<double> Z(new double[N * M], std::default_delete<double[]>());
    dstein(&N, D, E, &M, W.get(), IBLOCK.get(), ISPLIT.get(), Z.get(), &N);

    w = arma::dvec(W.get(), M);
    v = arma::dmat(Z.get(), N, M);
    inplace_trans(v);
}

void utils::getArmaMat2Txt(const arma::dmat & src, const std::string & pathAndName){
    std::ofstream fichier(pathAndName, std::ios::out | std::ios::trunc);
    if(fichier)  // si l'ouverture a réussi
    {
        // instructions
        fichier << src << "\n";
        fichier.close();  // on referme le fichier
    }
    else  // sinon
    {
        std::cerr << "Some wrong with getArmaMat2Txt !" << std::endl;
    }
}
void utils::getArmaVec2Txt(const arma::dvec & src, const std::string & pathAndName){
    std::ofstream fichier(pathAndName, std::ios::out | std::ios::trunc);
    if(fichier)  // si l'ouverture a réussi
    {
        // instructions
        fichier << src << "\n";
        fichier.close();  // on referme le fichier
    }
    else  // sinon
    {
        std::cerr << "Some wrong with getArmaMat2Txt !" << std::endl;
    }
}
arma::dmat utils::multitaper::dpss(int16_t M, double NW, int16_t Kmax, bool sym, std::string norm, bool return_ratios) {
    if ( helper::len_guards(M) ){
        return arma::dmat(M, 1, arma::fill::ones);
    }
    if ( norm.empty() ){
        if ( Kmax < 0 ){
            norm = "approximate";
        }else{
            norm = "2";
        }
    }
    std::vector<std::string> known_norms = {"2", "approximate", "subsample"};
    if ( std::find(known_norms.begin(), known_norms.end(), norm)== known_norms.end() ){
        throw std::runtime_error("\'" + norm + "\' is not an available norm type, please select it from \'2\', "
                                               "\'approximate\' and \'subsample\'");
    }
    bool singleton;
    if ( Kmax < 0 ){
        singleton = true;
        Kmax = 1.0;
    }else{
        singleton = false;
    }

    if ( Kmax <= 0 || Kmax > M ){
        throw std::runtime_error("Kmax must be greater than 0 and less than M");
    }
    if ( NW >= double(M/2.0) ){
        throw std::runtime_error("NW must be less than M/2");
    }
    if( NW <= 0 ){
        throw std::runtime_error("NW must be positive");
    }
    bool needs_trunc;
    M = helper::extend(M, sym, needs_trunc);
    double W = NW/double(M);

    arma::dvec nidx = arma::linspace(0, M - 1, M);
    arma::dvec d = arma::square((M - 1.0 - 2.0 * nidx) / 2.0 ) * std::cos( 2.0 * M_PI * W);
    arma::dvec e = nidx.subvec(1, nidx.index_max()) % (M - nidx.subvec(1, nidx.index_max())) / 2.0;

    arma::dvec w;
    arma::dmat windows;
    helper::eigh_tridiagonal(d, e, w, windows, std::make_pair<uint16_t, uint16_t>(M - Kmax, M - 1));
    w = arma::reverse(w);
    windows = arma::reverse(windows, 0);


    arma::dmat evenRows = windows.rows(arma::regspace<arma::uvec>(0, 2, windows.n_rows - 1));
    for (uint16_t i = 0; i < evenRows.n_rows; ++i ){
        if ( arma::sum(evenRows.row(i)) < 0 ){
            windows.row(i * 2) *= -1;
        }
    }

    double thresh = std::max(1e-7, 1.0 / double(M));
    arma::dmat oddRows = windows.rows(arma::regspace<arma::uvec>(1, 2, windows.n_rows - 1));
    for (uint16_t i = 0; i < oddRows.n_rows; ++i ){
        arma::uvec mask = arma::find( arma::square(oddRows.row(i)) > thresh );
        if ( oddRows.row(i)(mask(0)) < 0 ){
            windows.row(i * 2 + 1) = windows.row(i * 2 + 1) * -1;
        }
    }

    // TODO: implement
    if ( return_ratios ){

    }

    if ( norm != "2" ){
        windows /= windows.max();
        if ( M % 2 == 0 ){
            double correction;
            if ( norm == "approximate" ){
                correction = pow(M, 2) / double( pow(M, 2) + NW);
            }else{
                arma::dvec data2fft = windows.row(0).as_col();
                arma::cx_dvec s = arma::fft(data2fft);
                s = s.subvec(0, M/2);
                arma::cx_dvec shift = -(1 - 1.0/double(M)) * arma::conv_to<arma::cx_dvec>::from(
                        arma::linspace(1, M/2 + 1, M/2));
                s.subvec(1, M/2) %= 2 * arma::exp(-1 * M_PI * shift * 1i);
                correction = double(M) / arma::sum(arma::real(s));
            }
            windows *= correction;
        }
    }

    if ( needs_trunc ){
        windows = windows.cols(0, windows.n_cols - 1);
    }
    if ( singleton ){
        windows = windows.row(0);
    }
    return windows;
}
