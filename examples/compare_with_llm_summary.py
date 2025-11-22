"""
Validate that visualizations highlight same insights as LLM summary
"""

import pandas as pd
from context2visual import VisualizationGenerator

# Example: Käsittelyaika (Handling Time)
df = pd.DataFrame({
    'title': ['Incident 1', 'Incident 2', 'Incident 3', 'Incident 4', 'Incident 5'],
    'observation': [...],
    'created': ['2024-01-10', '2024-01-18', '2024-01-25', '2024-02-02', '2024-02-09'],
    'handled': ['2024-01-11', '2024-01-25', '2024-01-26', '2024-02-03', '2024-02-10']
})

prompt = "What was the average processing time? Are there trends?"

# LLM Summary (what stakeholders will compare against)
llm_summary = """
Analysis of 2024 security incidents:
- Average processing time: 2.8 days
- Range: 1-7 days
- Most incidents (60%) were processed within 1-2 days
- One outlier: 7 days processing time
- Trend: Slightly increasing processing times over the period
"""

# Your visualization
gen = VisualizationGenerator()
fig = gen.generate_visualization(prompt, df)

# Extract insights
insights = gen._extract_key_insights(df, gen.analyze_context(prompt, df))

print("="*60)
print("LLM SUMMARY vs VISUALIZATION INSIGHTS")
print("="*60)

print("\n📝 LLM Says:")
print(llm_summary)

print("\n📊 Your Visualization Shows:")
print(f"- Total observations: {insights['total_count']}")
print(f"- Average handling time: {insights['handling_time']['average']:.1f} days")
print(f"- Range: {insights['handling_time']['min']}-{insights['handling_time']['max']} days")
print(f"- Median: {insights['handling_time']['median']:.1f} days")

print("\n✅ Match Check:")
print(f"  Average time: {'✅ MATCH' if abs(insights['handling_time']['average'] - 2.8) < 0.5 else '❌ MISMATCH'}")
print(f"  Range captured: {'✅ MATCH' if insights['handling_time']['max'] == 7 else '❌ MISMATCH'}")

print("\n📈 Visualization Type:", type(fig).__name__)
print("Expected: Histogram showing distribution of handling times")