# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:08:45 2025

@author: HP
"""

import pandas as pd

def clean_logs(app_df, web_df, crm_df):
    # Fill missing values, standardize columns, join on user_id/email, etc.
    app_df.fillna(0, inplace=True)
    web_df.fillna(0, inplace=True)
    crm_df.fillna(0, inplace=True)

    merged = app_df.merge(web_df, on='user_id', how='left')
    merged = merged.merge(crm_df, on='user_id', how='left')

    # Feature engineering placeholder (derive age, engagement rates, etc.)
    merged['age'] = 2025 - pd.to_datetime(merged['dob'], errors='coerce').dt.year
    merged['email_engagement'] = merged['email_open_rate'] * merged['email_click_rate']

    return merged
