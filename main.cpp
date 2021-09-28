#include "utils.h"

int main()
{
    YAML::Node mts_args = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/multitaper.yaml");
    YAML::Node method = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/method.yaml");

    arma::dmat tapers = utils::multitaper::dpss(750, 5, 8);

    utils::getArmaMat2Txt(tapers, "./tapers.txt");
}
