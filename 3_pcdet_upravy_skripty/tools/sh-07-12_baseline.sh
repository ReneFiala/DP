#!/bin/bash
./sh-train-nested.sh ./cfgs/base_pointrcnn_mdl.yaml ./results/baseline_pointrcnn 300 repo_rnd_0 100 25 ap_sum_0.3 max
./sh-train-nested.sh ./cfgs/base_second_mdl.yaml ./results/baseline_second 300 repo_rnd_0 100 25 ap_sum_0.3 max
./sh-train-nested.sh ./cfgs/base_pillarnet_mdl.yaml ./results/baseline_pillarnet 300 repo_rnd_0 100 25 ap_sum_0.3 max
sudo shutdown -h now
