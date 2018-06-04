#!/bin/bash

./updateAgents.sh # Remove this line if running on your machine (not our test env)
python -m pysc2.bin.agent --map DefeatRoaches --agent pysc2.agents.RunAwAI_Agent.RunAwAI --norender &> out.log
