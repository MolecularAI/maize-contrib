#!/bin/bash

echo "Fetching Vina..."
wget https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz
echo "Unpacking Vina..."
tar -xvzf autodock_vina_1_1_2_linux_x86.tgz
echo "Fetching Gypsum-DL"
git clone https://github.com/durrantlab/gypsum_dl
