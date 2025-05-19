# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:08:08 2025

@author: HP
"""

import pandas as pd
import os

def load_raw_logs(app_path, web_path, crm_path):
    app_df = pd.read_csv(app_path)
    web_df = pd.read_csv(web_path)
    crm_df = pd.read_csv(crm_path)
    return app_df, web_df, crm_df