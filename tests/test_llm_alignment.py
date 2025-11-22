"""
Test that visualizations extract same insights as LLM summaries
"""

import pandas as pd
from context2visual import VisualizationGenerator
import json


def test_handling_time_insights():
    """Test: Handling time analysis"""
    
    # Sample data
    df = pd.DataFrame({
        'title': ['Obs1', 'Obs2', 'Obs3', 'Obs4', 'Obs5'],
        'observation': ['Issue A', 'Issue B', 'Issue C', 'Issue D', 'Issue E'],
        'havainto_pvm': ['2024-01-10', '2024-01-18', '2024-01-25', '2024-02-02', '2024-02-09'],
        'havainto_käsitelty_pvm': ['2024-01-11', '2024-01-25', '2024-01-26', '2024-02-03', '2024-02-10']
    })
    
    prompt = "What was the average processing time for incidents?"
    
    # Expected LLM insights
    expected_insights = {
        'total_count': 5,
        'average_days': 2.8,  # (1+7+1+1+1)/5
        'range': '1-7 days'
    }
    
    # Generate visualization and extract insights
    gen = VisualizationGenerator()
    context = gen.analyze_context(prompt, df)
    
    # Prepare data (converts dates, calculates duration)
    prepared = gen.prepare_data(df, 'histogram', context)
    
    # Extract insights
    insights = gen.extract_insights(df, context)
    
    print("="*60)
    print("TEST: Handling Time Insights Alignment")
    print("="*60)
    
    print(f"\n📝 Expected (from LLM):")
    print(f"  Total: {expected_insights['total_count']}")
    print(f"  Average: {expected_insights['average_days']} days")
    print(f"  Range: {expected_insights['range']}")
    
    print(f"\n📊 Extracted (from our module):")
    print(f"  Total: {insights.get('total_count')}")
    print(f"  Average: {insights.get('handling_time', {}).get('average', 0):.1f} days")
    print(f"  Range: {insights.get('handling_time', {}).get('min', 0):.0f}-{insights.get('handling_time', {}).get('max', 0):.0f} days")
    
    # Validation
    assert insights['total_count'] == expected_insights['total_count'], "Total count mismatch"
    
    avg_diff = abs(insights['handling_time']['average'] - expected_insights['average_days'])
    assert avg_diff < 0.5, f"Average differs by {avg_diff:.1f} days"
    
    print(f"\n✅ PASS: Insights match LLM summary")


def test_distribution_insights():
    """Test: Category distribution"""
    
    df = pd.DataFrame({
        'observation': [
            'Liukas lattia',
            'Liukas lattia käytävällä',
            'Portaat liukkaat',
            'Kaapeli kulkuväylällä',
            'Vuotava hana'
        ],
        'status': ['Archived', 'Archived', 'Implemented', 'Archived', 'Commenting']
    })
    
    prompt = "Show status distribution"
    
    # Expected insights
    expected = {
        'total': 5,
        'most_common': 'Archived',
        'most_common_count': 3,
        'most_common_percentage': 60.0
    }
    
    gen = VisualizationGenerator()
    context = gen.analyze_context(prompt, df)
    insights = gen.extract_insights(df, context)
    
    print("\n" + "="*60)
    print("TEST: Distribution Insights Alignment")
    print("="*60)
    
    print(f"\n📝 Expected:")
    print(f"  Total: {expected['total']}")
    print(f"  Most common: {expected['most_common']} ({expected['most_common_count']})")
    print(f"  Percentage: {expected['most_common_percentage']}%")
    
    print(f"\n📊 Extracted:")
    dist = insights.get('distribution', {})
    print(f"  Total: {insights.get('total_count')}")
    print(f"  Most common: {dist.get('most_common')} ({dist.get('most_common_count')})")
    print(f"  Percentage: {dist.get('most_common_percentage')}%")
    
    # Validation
    assert insights['total_count'] == expected['total']
    assert dist['most_common'] == expected['most_common']
    assert dist['most_common_count'] == expected['most_common_count']
    assert dist['most_common_percentage'] == expected['most_common_percentage']
    
    print(f"\n✅ PASS: Distribution insights match")


def test_same_data_different_visualizations():
    """Test: Same data, different analytical angles"""
    
    df = pd.DataFrame({
        'title': ['Obs1', 'Obs2', 'Obs3', 'Obs4'],
        'observation': ['Liukas lattia', 'Portaat', 'Kaapeli', 'Liukas portaat'],
        'created': pd.date_range('2024-05-01', periods=4, freq='W'),
        'handled': pd.date_range('2024-05-03', periods=4, freq='W'),
        'status': ['Archived', 'Implemented', 'Archived', 'Archived']
    })
    
    test_cases = [
        {
            'prompt': 'Show status distribution',
            'expected_viz': 'bar_chart',
            'should_show': 'Status categories'
        },
        {
            'prompt': 'Show trend over time',
            'expected_viz': 'time_series',
            'should_show': 'Observations per date'
        },
        {
            'prompt': 'What was average handling time?',
            'expected_viz': 'histogram',
            'should_show': 'Duration distribution'
        }
    ]
    
    gen = VisualizationGenerator()
    
    print("\n" + "="*60)
    print("TEST: Same Data, Different Analytical Angles")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Prompt: '{test_case['prompt']}'")
        
        context = gen.analyze_context(test_case['prompt'], df)
        viz_type = gen.select_visualization_type(context)
        
        print(f"   Expected: {test_case['expected_viz']}")
        print(f"   Got: {viz_type}")
        print(f"   Shows: {test_case['should_show']}")
        
        assert viz_type == test_case['expected_viz'], f"Viz type mismatch for: {test_case['prompt']}"
        print(f"   ✅ PASS")
    
    print(f"\n✅ ALL TESTS PASSED: Different questions → different visualizations")


if __name__ == "__main__":
    test_handling_time_insights()
    test_distribution_insights()
    test_same_data_different_visualizations()
    
    print("\n" + "="*60)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("="*60)
    print("\nConclusion:")
    print("- Insights extracted match LLM summaries")
    print("- Same data produces different viz based on question")
    print("- Solution is ready for integration")