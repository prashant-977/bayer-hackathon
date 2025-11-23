"""
Test script for the visualization module using mock HSE data
Run this to test your module without needing the full application
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from context2visual import generate_visualization
import plotly.io as pio

# Set plotly to open in browser for testing
pio.renderers.default = "browser"


def create_mock_data_scenario_1():
    """
    Scenario 1: Safety observations with status distribution
    Prompt: "Tee yhteenveto viime kuun turvallisuushavainnoista"
    (Make a summary of last month's safety observations)
    """
    data = {
        'id': range(18507, 18527),
        'created': [int((datetime.now() - timedelta(days=np.random.randint(1, 30))).timestamp() * 1000) 
                    for _ in range(20)],
        'lastupdate': [int(datetime.now().timestamp() * 1000) for _ in range(20)],
        'name': [
            'Lähettämön sisääntulon portaat todella liukkaat',
            'Liukkaat portaat BT01',
            'Huoltosillake',
            'Juoksentelua portaissa',
            'Ruokalan portaat',
            'Kaiteesta ei pidetty kiinni',
            'Henkilö osoitti hyvää esimerkkiä',
            'Turvallisuushavainto',
            'Kompastumisvaara käytävällä',
            'Liukas lattia',
            'Portaiden kaide rikki',
            'Hyvä turvallisuushavainto',
            'Liukastumisvaara',
            'Kaatumisvaara',
            'Portaat liukkaat',
            'Turvallisuus parantunut',
            'Varoitus merkki puuttuu',
            'Lattia liukas',
            'Portaat huonossa kunnossa',
            'Turvallisuustarkastus tehty'
        ],
        'status': np.random.choice(
            ['Archived', 'Implemented', 'Commenting', 'In Progress'],
            20,
            p=[0.5, 0.3, 0.15, 0.05]
        ),
        'division': np.random.choice(['Production', 'Logistics', 'Maintenance'], 20),
        'observationtype': ['Safety observation'] * 20,
        'viewscount': np.random.randint(1, 100, 20)
    }
    
    return pd.DataFrame(data)


def create_mock_data_scenario_2():
    """
    Scenario 2: Time series data
    Prompt: "Näytä turvallisuushavaintojen trendi viimeisen 3 kuukauden ajalta"
    (Show the trend of safety observations over the last 3 months)
    """
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
    
    data = {
        'id': range(1, len(dates) + 1),
        'created': [int(d.timestamp() * 1000) for d in dates],
        'lastupdate': [int(datetime.now().timestamp() * 1000) for _ in dates],
        'name': ['Observation ' + str(i) for i in range(len(dates))],
        'status': np.random.choice(['Archived', 'Implemented', 'Commenting'], len(dates)),
        'division': np.random.choice(['Production', 'Logistics'], len(dates)),
        'observationtype': ['Safety observation'] * len(dates)
    }
    
    return pd.DataFrame(data)


def create_mock_data_scenario_3():
    """
    Scenario 3: Category distribution without explicit categorical column
    Prompt: "Mitkä ovat yleisimmät turvallisuushavainnot?"
    (What are the most common types of safety observations?)
    """
    names = [
        'Liukkaat portaat', 'Kompastumisvaara', 'Liukas lattia',
        'Kaatumisvaara', 'Liukastuminen', 'Portaat liukkaat',
        'Juokseminen portaissa', 'Huolimattomuus', 'Kiirehtiminen',
        'Kaide puuttuu', 'Turvavaruste puuttuu', 'Varoitusmerkki puuttuu',
        'Ergonominen ongelma', 'Työasento huono', 'Liikenne ongelma',
        'Liukkaus', 'Kaatuminen', 'Kompastuminen', 'Portaiden ongelma',
        'Lattian ongelma'
    ] * 3
    
    data = {
        'id': range(1, len(names) + 1),
        'created': [int((datetime.now() - timedelta(days=np.random.randint(1, 30))).timestamp() * 1000) 
                    for _ in range(len(names))],
        'name': names,
        'status': np.random.choice(['Archived', 'Implemented'], len(names)),
        'division': np.random.choice(['Production', 'Logistics', 'Maintenance'], len(names))
    }
    
    return pd.DataFrame(data)


def test_scenario(scenario_num, prompt, data_func):
    """Test a specific scenario"""
    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_num}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}\n")
    
    # Generate mock data
    df = data_func()
    print(f"Generated DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Generate visualization
    print(f"\n{'→'*40}")
    print("Generating visualization...")
    print(f"{'→'*40}\n")
    
    try:
        fig = generate_visualization(prompt, df)
        
        if fig:
            print("✅ Visualization generated successfully!")
            print(f"Type: {type(fig)}")
            print(f"Title: {fig.layout.title.text if fig.layout.title else 'No title'}")
            
            # Show in browser
            fig.show()
            print("\n✓ Visualization opened in browser")
            
        else:
            print("ℹ️ No visualization was generated (this may be expected for some data)")
            
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}\n")


def main():
    """Run all test scenarios"""
    print("\n" + "="*80)
    print(" HSE VISUALIZATION MODULE - TEST SUITE")
    print("="*80)
    
    scenarios = [
        {
            'num': 1,
            'prompt': "Tee yhteenveto viime kuun turvallisuushavainnoista",
            'data_func': create_mock_data_scenario_1
        },
        {
            'num': 2,
            'prompt': "Näytä turvallisuushavaintojen trendi viimeisen 3 kuukauden ajalta",
            'data_func': create_mock_data_scenario_2
        },
        {
            'num': 3,
            'prompt': "Mitkä ovat yleisimmät turvallisuushavainnot?",
            'data_func': create_mock_data_scenario_3
        }
    ]
    
    for scenario in scenarios:
        test_scenario(
            scenario['num'],
            scenario['prompt'],
            scenario['data_func']
        )
    
    print("\n" + "="*80)
    print(" ALL TESTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()