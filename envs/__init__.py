#!usr/bin/env python3
# -*- coding: utf-8 -*-

REGISTRY = {}

from envs.airCombateEnv import *
from envs.aircombatenv_new import *
REGISTRY["airCombate"] = AirCombatEnv
REGISTRY["airCombateNvsM"] = AirCombatEnvMultiUnit
REGISTRY["airCombateNew"] = AirCombatEnv_new


def make(name):
	return REGISTRY[name]()

