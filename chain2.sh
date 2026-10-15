#!/bin/bash
cd .
while ! grep -q DONE rel_results/full.log; do sleep 120; done
python -u -m rel.direct_mc H7-out20 >> rel_results/chain2.log 2>&1
python -u -m rel.betamap >> rel_results/chain2.log 2>&1
echo DONE2 >> rel_results/chain2.log
