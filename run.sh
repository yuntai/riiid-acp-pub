#!/bin/bash

set -euxo pipefail

OPTARGS=""
if [[ $(hostname) == "nipa2020-0909" ]]; then
  echo "running on $(hostname)"
  OPTARGS="-e OPENBLAS_CORETYPE=nehalem"
fi

docker run --gpus all --shm-size=1g --ulimit memlock=-1 -d --ulimit stack=67108864 --rm \
  --ipc=host \
  --name riiid \
  -v $PWD:/workspace/riiid \
  -v /mnt/tmp/input:/workspace/riiid/input \
  -p 0.0.0.0:8888:8888 \
  $OPTARGS riiid
