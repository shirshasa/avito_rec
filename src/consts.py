
DATA_DIR = 'data/'

# Generator consts

event2weight = {
    17 : 1, #    88.851425
    11: 3, #    6.276978
    12 : 3, #   1.223713
    3: 2,    # 0.463616
    8: 2,   # 0.218001,
    16: 2,   #  0.149694
        
    10 : 20, #   1.216446
    15: 10,
    5: 10,
    19: 10,
    4: 10,
    13:10,
    14: 10,
    2:10,
    0:10,
    9: 10
}

event2weight = {
        17 : 1, #    88.851425
        11: 1, #    6.276978
        12 : 2, #   1.223713
        3: 2,    # 0.463616
        8: 2,   # 0.218001,
        16: 2,   #  0.149694
            
        10 : 50, #   1.216446
        15: 20,
        5: 20,
        19: 20,
        4: 10,
        13:10,
        14: 10,
        2:10,
        0:10,
        9: 10
}

# Ranker const

cat = [
    'node_category',
    'most_freq_surface_CAT',
    'most_freq_event_CAT',
    'most_freq_event_contact_CAT',
    
    # 'location_top_last_contact_CAT',
    # 'most_freq_top_location_contact_CAT',
    
    # 'most_freq_surface_LOC',
    # 'most_freq_event_LOC',
    # 'most_freq_event_contact_LOC'
]

nodes = [
    'node_last_contact_CAT',
    'most_freq_node_contact_CAT',
    'node_last_contact_LOC',
    'most_freq_node_contact_LOC'
]

feats = [
    'cookie',
    'node_category',
    'node_location_num',
    'node_item_num',

    # 'num_contacts_CAT',
    'pr_contact_CAT',
    'surface_unique_counts_CAT',
    'location_unique_counts_CAT',

    'most_freq_surface_CAT', #
    'most_freq_event_CAT', #
    'most_freq_event_contact_CAT',#

    # 'location_top_last_contact_CAT', #
    # 'most_freq_top_location_contact_CAT', #

    # 'num_contacts_LOC',
    'pr_contact_LOC',
    'surface_unique_counts_LOC',

    # 'most_freq_surface_LOC', #
    # 'most_freq_event_LOC', #
    # 'most_freq_event_contact_LOC', #

    'cosine_sim_node_last_contact_CAT',
    'cosine_sim_most_freq_node_contact_CAT',
    'cosine_sim_node_last_contact_LOC',
    'cosine_sim_most_freq_node_contact_LOC',

    'contact_ratio', 'contact_active_days_ratio', 'active_days_pr', 'contact_active_events_ratio',
    'contact_active_last_week_events_ratio', 'pr_since_last_contact', 'is_contact_last', 'pr_since_last',
    'category_0_pr', 'category_10_pr', 'category_11_pr', 'category_12_pr', 'category_13_pr', 'category_14_pr', 
    'category_15_pr', 'category_17_pr', 'category_18_pr', 'category_19_pr', 'category_1_pr', 'category_20_pr', 
    'category_23_pr', 'category_24_pr', 'category_25_pr', 'category_26_pr', 'category_28_pr', 'category_29_pr', 
    'category_2_pr', 'category_30_pr', 'category_31_pr', 'category_32_pr', 'category_33_pr', 'category_34_pr', 
    'category_35_pr', 'category_37_pr', 'category_40_pr', 'category_41_pr', 'category_42_pr', 'category_43_pr', 
    'category_44_pr', 'category_46_pr', 'category_47_pr', 'category_48_pr', 'category_49_pr', 'category_50_pr', 
    'category_51_pr', 'category_52_pr', 'category_53_pr', 'category_57_pr', 'category_58_pr', 'category_59_pr', 
    'category_5_pr', 'category_61_pr', 'category_62_pr', 'category_63_pr', 'category_64_pr', 'category_6_pr', 
    'category_7_pr', 'category_8_pr', 'category_9_pr', 'event_0_pr', 'event_10_pr', 'event_11_pr', 'event_12_pr', 
    'event_13_pr', 'event_14_pr', 'event_15_pr', 'event_16_pr', 'event_17_pr', 'event_18_pr', 'event_19_pr', 'event_1_pr', 
    'event_2_pr', 'event_3_pr', 'event_4_pr', 'event_5_pr', 'event_6_pr', 'event_8_pr', 'event_9_pr', 'surface_0_pr', 'surface_10_pr', 
    'surface_11_pr', 'surface_12_pr', 'surface_13_pr', 'surface_14_pr', 'surface_15_pr', 'surface_16_pr', 'surface_17_pr', 
    'surface_18_pr', 'surface_1_pr', 'surface_2_pr', 'surface_3_pr', 'surface_4_pr', 'surface_5_pr', 'surface_6_pr', 'surface_7_pr', 
    'surface_8_pr', 'surface_9_pr', 'ctr'
]
