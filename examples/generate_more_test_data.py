"""
Generate additional artificial test data
Stakeholders said: "You are more than welcome to generate 
additional safety observations similar to the provided ones"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Finnish safety observation templates
OBSERVATION_TEMPLATES = {
    'slipping': [
        'Liukas lattia {location}',
        'Vesilammikko {location} aiheuttaa liukastumisvaaran',
        'Lattia liukas {location}, vaatii huomiota',
        '{location} lattia erittäin liukas',
        'Liukastumisvaara havaittu {location}'
    ],
    'stairs': [
        'Portaat liukkaat {location}',
        'Rappusten kaide heiluu {location}',
        'Portaisiin jätetty tavaraa {location}',
        '{location} portaat huonossa kunnossa',
        'Portaiden valaistus puutteellinen {location}'
    ],
    'equipment': [
        'Suojavaruste puuttuu {location}',
        'Työväline rikki {location}',
        'Hengityssuojain ei käytössä {location}',
        'Turvavarusteet puutteelliset {location}',
        'Henkilökohtaiset suojaimet puuttuvat {location}'
    ],
    'electrical': [
        'Kaapeli kulkuväylällä {location}',
        'Sähköjohto irti {location}',
        'Pistorasia vaurioitunut {location}',
        'Sähkölaitteiden johto rikki {location}',
        'Valojen kytkentä rikki {location}'
    ],
    'fire': [
        'Palovaroitin ei toimi {location}',
        'Hätäpoistumistie tukossa {location}',
        'Sammutuslaite puuttuu {location}',
        'Palovaroittimen paristo lopussa {location}',
        'Hätä-seis-painike peitetty {location}'
    ],
    'maintenance': [
        'Vuotava hana {location}',
        'Rikkinäinen ovi {location}',
        'Lattian reikä {location}',
        'Ikkunan lasi haljennut {location}',
        'Vaurio seinässä {location}'
    ]
}

LOCATIONS = [
    'tuotannossa',
    'varastossa',
    'käytävällä',
    'toimistossa',
    'taukotilassa',
    'lähettämössä',
    'laboratoriossa',
    'pakkausosastolla',
    'BT01:ssä',
    'rakennuksessa A'
]

STATUSES = ['Archived', 'Implemented', 'Commenting', 'In Progress']
DIVISIONS = ['Production', 'Logistics', 'Maintenance', 'Laboratory']


def generate_observation(obs_type: str, date: datetime) -> dict:
    """Generate a single observation"""
    template = random.choice(OBSERVATION_TEMPLATES[obs_type])
    location = random.choice(LOCATIONS)
    title = template.format(location=location)
    
    # Generate handling time (1-10 days, most are 1-3)
    handling_days = int(np.random.exponential(2) + 1)
    handling_days = min(handling_days, 10)
    
    return {
        'title': title,
        'observation': title + f'. Havaittu {date.strftime("%d.%m.%Y")}.',
        'observation_date': date.strftime('%m/%d/%Y'),
        'observation_handled_date': (date + timedelta(days=handling_days)).strftime('%m/%d/%Y'),
        'status': random.choice(STATUSES),
        'division': random.choice(DIVISIONS),
        'observation_type': obs_type.title()
    }


def generate_dataset(name: str, n_observations: int = 20, 
                    start_date: str = '2024-01-01',
                    obs_types: list = None) -> pd.DataFrame:
    """
    Generate a complete test dataset
    
    Args:
        name: Dataset name
        n_observations: Number of observations to generate
        start_date: Start date for observations
        obs_types: Types of observations to include (default: all)
    """
    if obs_types is None:
        obs_types = list(OBSERVATION_TEMPLATES.keys())
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    observations = []
    for i in range(n_observations):
        # Random date within 90 days
        days_offset = random.randint(0, 90)
        obs_date = start + timedelta(days=days_offset)
        
        # Random observation type
        obs_type = random.choice(obs_types)
        
        obs = generate_observation(obs_type, obs_date)
        observations.append(obs)
    
    df = pd.DataFrame(observations)
    
    print(f"Generated '{name}' dataset:")
    print(f"  - {len(df)} observations")
    print(f"  - Date range: {df['observation_date'].min()} to {df['observation_date'].max()}")
    print(f"  - Types: {df['observation_type'].value_counts().to_dict()}")
    print()
    
    return df


def generate_all_test_datasets():
    """Generate comprehensive test datasets"""
    
    datasets = {}
    
    # 1. Stairs-focused dataset
    datasets['stairs'] = generate_dataset(
        'Stairs (Rappuset)',
        n_observations=15,
        start_date='2024-03-01',
        obs_types=['stairs', 'sl