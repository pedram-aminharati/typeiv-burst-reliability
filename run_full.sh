#!/bin/bash
cd .
python -u -m rel.study_full H5-ref H5-e0125 H7-ref H7-out20 H7-out10 H7-out15 H7-out25 H7-out30 >> rel_results/full.log 2>&1
python -u -m rel.study_full >> rel_results/full.log 2>&1
echo DONE >> rel_results/full.log
