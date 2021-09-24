#include <iostream>
#include <memory>
#include "Eigen/Dense"
#include "utils.h"

int main()
{
    YAML::Node mts_args = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/multitaper.yaml");
    YAML::Node method = YAML::LoadFile("/mnt/c/Users/heyanbai/CLionProjects/generic-pse/method.yaml");

    Eigen::MatrixXd tapers = utils::dpss(750, 3, 5);

    std::cout<<tapers.rows()<<" "<<tapers.cols()<<std::endl;
    utils::getEigentoData(tapers, "./tapers.txt");

}
