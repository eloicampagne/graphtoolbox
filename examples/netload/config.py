import pandas as pd
import numpy as np

features_load = ['temperature', 'temperature_lisse_990', 'temperature_lisse_950', 'wind', 'nebulosity', 
                'toy', 'year', 'month', 'day_type_jf', 'day_type_week', 'period_holiday', 'period_hour_changed',
                'period_holiday_zone_a', 'period_holiday_zone_b', 'period_holiday_zone_c', 'period_christmas', 'period_summer']
features_wind = ['wind', 'wind_by_wind_power_weights', 'toy', 'year', 'month']
features_solar = ['nebulosity', 'nebulosity_by_solar_power_weights', 'toy', 'year', 'month']
features = np.unique(list(set(features_load) | set(features_wind) | set(features_solar)))

model_kwargs = {
    'model_name': 'GraphSAGE',
    'lr': 0.001,
    'num_epochs': 200,
    'hidden_channels': 97,
    'out_channels': 1,
    'num_layers': 1,
    'batch_size': 314,
}

data_kwargs = {
    'day_sup_train': '2018-01-01',
    'day_inf_val': '2018-01-01',
    'node_var': 'Region',
    'dummies': ['tod', 'day_type_week']

}

dataset_kwargs = {
    'adj_matrix': 'dtw',
    'batch_size': 512,
    'features_base': features,
    'target_base': 'NetLoad'
}

optim_kwargs = {
    'num_layers': (1, 5),
    'hidden_channels': (32, 128),
    'batch_size': (256, 1024),
    'heads': (1, 8)
}

explain_kwargs = {
    'months': {0: 'January', 7200: 'June', 10128: 'August', 14496: 'November'},
}

df_pos = pd.DataFrame(
    {'VILLE': ['LILLE', 'ROUEN', 'PARIS', 'STRASBOURG', 'BREST', 'NANTES', 'ORLEANS', 'DIJON', 'BORDEAUX', 'LYON',
              'TOULOUSE', 'MARSEILLE'],
    'LATITUDE': [50.6365654, 49.4404591, 48.862725, 48.584614, 
                 48.3905283, 47.2186371, 47.9027336, 47.3215806, 
                 44.841225, 45.7578137, 43.6044622, 43.2961743],
    'LONGITUDE': [3.0635282, 1.0939658, 2.287592, 7.7507127, 
                  -4.4860088, -1.5541362, 1.9086066, 5.0414701, 
                  -0.5800364, 4.8320114, 1.4442469, 5.3699525],
    'REGION': ['haut_de_france', 'normandie', 'ile_de_france', 'grand_est', 'bretagne', 'pays_de_loire', 
                'centre_val_de_loire', 'bourgogne_franche_comte', 'nouvelle_aquitaine', 'auvergne_rhone_alpes', 
                'occitanie', 'paca'],
    'SUPERFICIE_REGION': [31813, 29906, 12011, 57433, 27208, 32082, 39151, 47784, 83809, 69711, 72724, 31400],
    'POPULATION_REGION': [5987172, 3307286, 12395148, 5542094, 3402932, 3873096, 2564915, 2785393, 6081985,
                          8153233, 6053548, 5131187]
    }
).sort_values(by='REGION').reset_index(drop=True)
