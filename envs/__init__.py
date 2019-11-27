#!usr/bin/env python3
# -*- coding: utf-8 -*-

REGISTRY = {}

from envs.airCombateEnv import *
REGISTRY["airCombate"] = AirCombatEnv
REGISTRY["airCombateNvsM"] = AirCombatEnvMultiUnit
REGISTRY["guidence"] = AirCombatEnvOverload

def make(name):
	return REGISTRY[name]()




