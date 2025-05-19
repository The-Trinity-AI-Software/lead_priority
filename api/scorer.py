# -*- coding: utf-8 -*-
"""
Created on May 12 19:03:29 2025

@author: HP
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.features import extract_features
from model.predictor import predict

def score_lead(lead):
    features = extract_features(lead)
    result = predict(features)
    return result
