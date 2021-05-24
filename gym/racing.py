#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:25:50 2020

@author: hamzamahdi
"""
import gym

env = gym.make("CarRacing-v0").env

env.render()
