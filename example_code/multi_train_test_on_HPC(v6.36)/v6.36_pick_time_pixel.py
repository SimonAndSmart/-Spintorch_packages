#!/usr/bin/env python
# coding: utf-8

# # SpinTorch (Local/Json version6.36) 
# Use multi virtual probes with different weighting to get intensity readings.
# v6.32
# - fixed loss value representation during testing.
# 
# v6.33
# - changed 'weighting_cut_off' into 'dist_cut_off' so now we define the radius of the probe directlly.
# - added ability to make mutiple accuracy test
# - better way to chose equal probe measure method
# 
# v6.34
# - bug fix
# 
# v6.35
# - add max intensity classify for accuracy test back
# 
# v6.36
# - devided the one sum-intensity value into a set of 2ns period intensity sum
# - added a extra layer over regression training so it will pick the region gives the best acc on training data.
# - detailed acc result for each choice of time pixle saved into txt files
# 
# Latter
# - fix ROC curve

# # Set all parameters
