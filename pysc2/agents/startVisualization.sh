#!/bin/bash

./updateAgents.sh
python -m pysc2.bin.agent --map DefeatRoaches --agent pysc2.agents.RunAwAI_Agent.RunAwAI
