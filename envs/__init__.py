#!usr/bin/env python3
# -*- coding: utf-8 -*-
from envs.guidneceEnv import GuidenceEnvOverload

REGISTRY = {}

from envs.airCombateEnv import *
REGISTRY["airCombate"] = AirCombatEnv
REGISTRY["airCombateNvsM"] = AirCombatEnvMultiUnit
REGISTRY["guidence"] = GuidenceEnvOverload

def make(name):
	return REGISTRY[name]()




