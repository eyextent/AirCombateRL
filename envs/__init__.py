#!usr/bin/env python3
# -*- coding: utf-8 -*-
from envs.airCombateEnv.airCombateEnv import *
from envs.landingGuidanceEnv.guidneceEnv import GuidenceEnvOverload

REGISTRY = {}
REGISTRY["airCombate"] = AirCombatEnv
REGISTRY["airCombateNvsM"] = AirCombatEnvMultiUnit
REGISTRY["guidence"] = GuidenceEnvOverload

def make(name):
	return REGISTRY[name]()