#include <iostream>
#include <memory>
#include "utils.h"

int main()
{
    YAML::Node mts_args = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/multitaper.yaml");
    YAML::Node method = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/method.yaml");

    arma::dmat tapers = utils::dpss(750, 5, 8);

    utils::getEigentoData(tapers, "./tapers.txt");
}
