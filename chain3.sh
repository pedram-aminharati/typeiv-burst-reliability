#!/bin/bash
cd .
while ! grep -q DONE2 rel_results/chain2.log 2>/dev/null; do sleep 120; done
python -u -m rel.thermal_full >> rel_results/chain3.log 2>&1
echo DONE3 >> rel_results/chain3.log
