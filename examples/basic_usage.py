"""
Basic Usage Examples for Context2Visual Module

This file demonstrates simple usage patterns for the visualization module.
All examples use inline data - no external files needed.

Run: python examples/basic_usage.py
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from context2visual import generate_visualization, VisualizationGenerator


def example_1_bar_chart():
    """
    Example 1: Bar Chart - Status Distribution
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Bar Chart - Status Distribution")
    print("="*60)
    
    # Create sample data
    df = pd.DataFrame({
        'observation': [
            'Liukas lattia käytävällä',
            'Portaat liukkaat',
            'Kaapeli kulkuväylällä',
            'Vuotava hana',
            'Liukas lattia taukotilassa',
            'Hylly ylikuormitettu',
            'Palovaroitin ei toimi',
            'Kemikaaliroiske lattialla'
        ],
        'status': [
            'Archived', 'Implemented', 'Archived', 'Commenting',
            'Archived', 'Implemented', 'In Progress', 'Archived'
        ],
        'division': [
            'Production', 'Logistics', 'Production', 'Maintenance',
            'Production', 'Logistics', 'Production', 'Laboratory'
        ]
    })
    
    print(f"\nDataFrame: {len(df)} observations")
    print(df[['observation', 'status']].head(3))
    
    # Generate visualization
    prompt = "Show status distribution"
    print(f"\nPrompt: '{prompt}'")
    
    fig = generate_visualization(prompt, df)
    
    if fig:
        print(f"✅ Generated: {type(fig).__name__}")
        print(f"   Title: {fig.layout.title.text}")
        
        # Save to HTML
        fig.write_html('example_1_bar_chart.html')
        print("   📊 Saved to: example_1_bar_chart.html")
    else:
        print("❌ No visualization generated")


def example_2_time_series():
    """
    Example 2: Time Series - Observations Over Time
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Time Series - Trend Over Time")
    print("="*60)
    
    # Create sample data with dates
    df = pd.DataFrame({
        'title': [f'Observation {i}' for i in range(1, 16)],
        'observation_date': pd.date_range('2024-05-01', periods=15, freq='2D'),
        'status': ['Archived'] * 15
    })
    
    print(f"\nDataFrame: {len(df)} observations")
    print(f"Date range: {df['observation_date'].min().date()} to {df['observation_date'].max().date()}")
    
    # Generate visualization
    prompt = "Show trend of observations over time"
    print(f"\nPrompt: '{prompt}'")
    
    fig = generate_visualization(prompt, df)
    
    if fig:
        print(f"✅ Generated: {type(fig).__name__}")
        print(f"   Title: {fig.layout.title.text}")
        
        # Save to HTML
        fig.write_html('example_2_time_series.html')
        print("   📊 Saved to: example_2_time_series.html")
    else:
        print("❌ No visualization generated")


def example_3_histogram():
    """
    Example 3: Histogram - Handling Time Distribution
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Histogram - Handling Time Analysis")
    print("="*60)
    
    # Create sample data with two dates
    df = pd.DataFrame({
        'title': [
            'Vesilammikko käytävällä',
            'Heiluva kaide rappusissa',
            'Puutteellinen suoja-asuste',
            'Tavaraa portaisiin',
            'Kemikaaliroiske',
            'Vuotava hana',
            'Rikkinäinen ovi',
            'Liukas lattia',
            'Kaapeli kulkuväylällä',
            'Portaat liukkaat'
        ],
        'created': pd.date_range('2024-01-10', periods=10, freq='7D'),
        'handled': pd.date_range('2024-01-12', periods=10, freq='7D')
    })
    
    # Adjust some handling dates for variety
    df.loc[1, 'handled'] = df.loc[1, 'created'] + pd.Timedelta(days=7)
    df.loc[5, 'handled'] = df.loc[5, 'created'] + pd.Timedelta(days=1)
    df.loc[8, 'handled'] = df.loc[8, 'created'] + pd.Timedelta(days=5)
    
    print(f"\nDataFrame: {len(df)} observations")
    print(f"First few rows:")
    print(df[['title', 'created', 'handled']].head(3))
    
    # Generate visualization
    prompt = "What was the average handling time?"
    print(f"\nPrompt: '{prompt}'")
    
    fig = generate_visualization(prompt, df)
    
    if fig:
        print(f"✅ Generated: {type(fig).__name__}")
        print(f"   Title: {fig.layout.title.text}")
        
        # Save to HTML
        fig.write_html('example_3_histogram.html')
        print("   📊 Saved to: example_3_histogram.html")
    else:
        print("❌ No visualization generated")


def example_4_text_extraction():
    """
    Example 4: Category Extraction from Finnish Text
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Category Extraction from Text")
    print("="*60)
    
    # Create sample data with only text (no explicit categories)
    df = pd.DataFrame({
        'observation': [
            'Liukas lattia käytävällä, liukastumisvaara',
            'Portaat erittäin liukkaat talvella',
            'Kaapeli kulkuväylällä kompastumisvaara',
            'Verkkokaapeli käytävän poikki',
            'Vuotava hana taukotilassa',
            'Vesivahinko vuotavan hanan takia',
            'Liukas lattia taukotilassa',
            'Portaiden kaide heiluu vaarallisesti',
            'Sähköjohto irti seinästä',
            'Kaapeli lattialla toimistossa'
        ]
    })
    
    print(f"\nDataFrame: {len(df)} observations (text only)")
    print("First few observations:")
    for i, obs in enumerate(df['observation'].head(3), 1):
        print(f"  {i}. {obs}")
    
    # Generate visualization
    prompt = "What are the most common types of observations?"
    print(f"\nPrompt: '{prompt}'")
    
    fig = generate_visualization(prompt, df)
    
    if fig:
        print(f"✅ Generated: {type(fig).__name__}")
        print(f"   Title: {fig.layout.title.text}")
        print("   Categories extracted from text using TF-IDF!")
        
        # Save to HTML
        fig.write_html('example_4_text_extraction.html')
        print("   📊 Saved to: example_4_text_extraction.html")
    else:
        print("❌ No visualization generated")


def example_5_same_data_different_questions():
    """
    Example 5: Same Data, Different Questions → Different Visualizations
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Same Data, Different Analytical Angles")
    print("="*60)
    
    # Create comprehensive dataset
    df = pd.DataFrame({
        'observation': [
            'Liukas lattia', 'Portaat liukkaat', 'Kaapeli kulkuväylällä',
            'Vuotava hana', 'Liukas lattia', 'Portaat',
            'Kaapeli', 'Vuoto', 'Liukas', 'Portaat'
        ] * 2,  # 20 observations
        'status': ['Archived', 'Implemented', 'Archived', 'Commenting', 'Archived'] * 4,
        'created': pd.date_range('2024-05-01', periods=20, freq='D'),
        'handled': pd.date_range('2024-05-03', periods=20, freq='D'),
        'division': ['Production', 'Logistics', 'Maintenance', 'Laboratory'] * 5
    })
    
    print(f"\nSame DataFrame: {len(df)} observations")
    print(f"Columns: {df.columns.tolist()}")
    
    # Different questions on SAME data
    questions = [
        "Show status distribution",
        "Show trend over time",
        "What was average handling time?",
        "Show by division"
    ]
    
    for i, prompt in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Prompt: '{prompt}'")
        
        fig = generate_visualization(prompt, df)
        
        if fig:
            print(f"✅ Generated: {type(fig).__name__}")
            print(f"   Title: {fig.layout.title.text}")
            fig.write_html(f'example_5_question_{i}.html')
            print(f"   📊 Saved to: example_5_question_{i}.html")
        else:
            print("❌ No visualization generated")
    
    print("\n🎯 Result: Same data → 4 different visualizations based on question!")


def example_6_advanced_usage():
    """
    Example 6: Advanced Usage with VisualizationGenerator Class
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Advanced Usage - Custom Configuration")
    print("="*60)
    
    # Create sample data
    df = pd.DataFrame({
        'observation': [f'Issue {i}' for i in range(20)],
        'category': ['Safety', 'Equipment', 'Maintenance', 'Other'] * 5,
        'priority': ['High', 'Medium', 'Low'] * 6 + ['High', 'Medium']
    })
    
    print(f"\nDataFrame: {len(df)} observations")
    
    # Option 1: Use with custom configuration
    custom_config = {
        'max_categories': 10,     # Show max 10 categories
        'min_word_length': 5,      # Only words >= 5 chars
        'use_bigrams': False       # Single words only
    }
    
    gen = VisualizationGenerator(config=custom_config)
    
    prompt = "Show category breakdown"
    print(f"\nPrompt: '{prompt}'")
    print("Using custom configuration...")
    
    fig = gen.generate_visualization(prompt, df)
    
    if fig:
        print(f"✅ Generated: {type(fig).__name__}")
        
        # Option 2: Extract insights separately
        context = gen.analyze_context(prompt, df)
        insights = gen.extract_insights(df, context)
        
        print("\nExtracted Insights:")
        print(f"  Total: {insights.get('total_count')}")
        if 'distribution' in insights:
            print(f"  Most common: {insights['distribution'].get('most_common')}")
            print(f"  Percentage: {insights['distribution'].get('most_common_percentage')}%")
        
        fig.write_html('example_6_advanced.html')
        print("\n📊 Saved to: example_6_advanced.html")
    else:
        print("❌ No visualization generated")


def example_7_error_handling():
    """
    Example 7: Error Handling and Edge Cases
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling and Edge Cases")
    print("="*60)
    
    # Test case 1: Empty dataframe
    print("\nTest 1: Empty DataFrame")
    df_empty = pd.DataFrame()
    fig = generate_visualization("Show data", df_empty)
    print(f"Result: {fig if fig else 'None (expected - no data)'}")
    
    # Test case 2: Single row
    print("\nTest 2: Single Row")
    df_single = pd.DataFrame({'status': ['Archived']})
    fig = generate_visualization("Show status", df_single)
    print(f"Result: {fig if fig else 'None (expected - not enough data)'}")
    
    # Test case 3: No suitable columns
    print("\nTest 3: No Suitable Columns")
    df_ids = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
    fig = generate_visualization("Show distribution", df_ids)
    print(f"Result: {fig if fig else 'None (expected - no visualizable columns)'}")
    
    # Test case 4: Valid data
    print("\nTest 4: Valid Data")
    df_valid = pd.DataFrame({
        'status': ['Archived', 'Implemented', 'Archived'],
        'count': [10, 5, 3]
    })
    fig = generate_visualization("Show status", df_valid)
    print(f"Result: {'Generated successfully!' if fig else 'Failed'}")
    
    print("\n✅ Error handling works correctly!")


def main():
    """
    Run all examples
    """
    print("="*60)
    print("CONTEXT2VISUAL - BASIC USAGE EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates common usage patterns.")
    print("HTML files will be saved to the current directory.")
    print("Open them in your browser to view the visualizations.")
    
    # Run examples
    example_1_bar_chart()
    example_2_time_series()
    example_3_histogram()
    example_4_text_extraction()
    example_5_same_data_different_questions()
    example_6_advanced_usage()
    example_7_error_handling()
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    print("  • example_1_bar_chart.html")
    print("  • example_2_time_series.html")
    print("  • example_3_histogram.html")
    print("  • example_4_text_extraction.html")
    print("  • example_5_question_1.html to example_5_question_4.html")
    print("  • example_6_advanced.html")
    print("\n📊 Open these files in your browser to view visualizations!")
    print("\nKey Takeaways:")
    print("  1. Simple one-line usage: generate_visualization(prompt, df)")
    print("  2. Automatically chooses chart type based on question")
    print("  3. Works with text-only data (extracts categories)")
    print("  4. Same data → different viz based on question")
    print("  5. Handles edge cases gracefully")


if __name__ == "__main__":
    main()