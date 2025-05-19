# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:01:15 2025

@author: HP
"""

def extract_features(raw):
    feature_keys = [
        'first_name', 'last_name', 'address', 'mobile', 'drivers_license', 'dob', 'age', 'home_owner',
        'monthly_income', 'income_verified', 'employer', 'total_session_duration', 'avg_pages_per_session',
        'bounce_rate_flag', 'scroll_depth_pct', 'repeat_visitor_count', 'device_switching',
        'traffic_paid_pct', 'traffic_organic_pct', 'time_outside_hours_pct',
        'time_step1', 'time_step2', 'time_step3', 'inter_step_latency', 'drop_off_step', 'form_edits',
        'funnel_revisits', 'funnel_friction_score', 'email_open_rate', 'email_click_through_rate',
        'time_to_first_email_open', 'sms_bounce_rate', 'sms_reply_rate', 'click_to_call_ratio',
        'reply_sentiment_score', 'opt_out_flag', 'days_since_last_application', 'days_since_last_session',
        'days_since_last_email_open', 'days_since_last_sms_reply', 'event_recency_bucket',
        'frequency_last_30_days', 'inverse_recency_weight', 'recency_frequency_interaction',
        'referral_code_used', 'geo_behavior_distance', 'support_chat_activity', 'credit_check_days_ago',
        'risk_proxy_flag'
    ]
    return {key: raw.get(key, 0) for key in feature_keys}
