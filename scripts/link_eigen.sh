#!/bin/bash
# DROID-SLAM's eigen submodule is a GitLab checkout this environment cannot
# reach. The extension include path wants the Eigen headers next to the sources.
set -euo pipefail
apt-get update
apt-get install -y libeigen3-dev
for d in \
  thirdparty/DROID-SLAM/thirdparty/eigen \
  thirdparty/DROID-SLAM/thirdparty/lietorch/eigen
do
  mkdir -p "$d"
  ln -sfn /usr/include/eigen3/Eigen "$d/Eigen"
  ln -sfn /usr/include/eigen3/unsupported "$d/unsupported"
  ln -sfn /usr/include/eigen3/signature_of_eigen3_matrix_library "$d/signature_of_eigen3_matrix_library"
done
echo "eigen headers linked"
