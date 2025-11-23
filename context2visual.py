"""
Context2Visual: Enhanced with Insight Extraction and Non-Technical Focus
Aligns with LLM summaries, optimized for non-technical users
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, Optional, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np


class VisualizationGenerator:
    """
    Intelligent visualization generator with insight extraction
    Designed for non-technical users
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Default configuration
        self.viz_config = {
            'color_scheme': 'Blues',
            'default_height': 500,
            'max_categories': 15,
            'min_word_length': 4,
            'top_n_words': 10,
            'use_bigrams': True,
            'show_insights': True  # Show quantitative insights on charts
        }
        
        if config:
            self.viz_config.update(config)
        
        # Intent keywords for visualization selection
        self.intent_keywords = {
            'temporal': ['trend', 'trendi', 'over time', 'ajan', 'kehitys', 'change', 'muutos'],
            'distribution': ['distribution', 'jakauma', 'breakdown', 'jako'],
            'status': ['status', 'tila', 'state'],
            'duration': ['time', 'aika', 'duration', 'kesto', 'how long', 'kauanko',
                        'average', 'keskiarvo', 'processing', 'käsittelyaika', 'handling'],
            'count': ['how many', 'montako', 'kuinka monta', 'count', 'määrä'],
            'top': ['most common', 'yleisimmät', 'most', 'eniten', 'top'],
            'comparison': ['compare', 'vertaa', 'difference', 'ero', 'vs']
        }
    
    def generate_visualization(self, prompt: str, df: pd.DataFrame) -> Optional[Any]:
        """
        Main entry point with insight extraction
        """
        if df is None or df.empty or len(df) < 2:
            return None
        
        # Analyze context
        context = self.analyze_context(prompt, df)
        
        # Extract quantitative insights (like LLM would)
        insights = self.extract_insights(df, context)
        context['insights'] = insights
        
        # Select visualization type based on question
        viz_type = self.select_visualization_type(context)
        
        if viz_type == 'none':
            return None
        
        # Prepare data
        prepared_data = self.prepare_data(df, viz_type, context)
        
        if prepared_data is None or prepared_data.empty:
            return None
        
        # Create visualization with insights
        fig = self.create_visualization(prepared_data, viz_type, context)
        
        return fig
    
    def extract_insights(self, df: pd.DataFrame, context: Dict) -> Dict:
        """
        Extract quantitative insights that would appear in LLM summary
        These help validate that visualization shows same story as text
        """
        insights = {}
        
        # Basic count
        insights['total_count'] = len(df)
        
        # Temporal insights
        if context['temporal_cols']:
            date_col = context['temporal_cols'][0]
            df_dates = df[date_col].dropna()
            
            if len(df_dates) > 0:
                insights['temporal'] = {
                    'start_date': df_dates.min(),
                    'end_date': df_dates.max(),
                    'span_days': (df_dates.max() - df_dates.min()).days,
                    'observations_per_day': len(df_dates) / max((df_dates.max() - df_dates.min()).days, 1)
                }
        
        # Handling time insights (if two date columns)
        if len(context['temporal_cols']) >= 2:
            start_col = context['temporal_cols'][0]
            end_col = context['temporal_cols'][1]
            
            durations = (df[end_col] - df[start_col]).dt.total_seconds() / (24 * 3600)  # Convert to days
            durations = durations[durations >= 0]
            
            if len(durations) > 0:
                insights['handling_time'] = {
                    'average': durations.mean(),
                    'median': durations.median(),
                    'min': durations.min(),
                    'max': durations.max(),
                    'std': durations.std(),
                    'within_1_day': (durations <= 1).sum(),
                    'within_3_days': (durations <= 3).sum(),
                    'over_5_days': (durations > 5).sum()
                }
        
        # Categorical distribution insights
        if context['categorical_cols']:
            cat_col = context['categorical_cols'][0]
            value_counts = df[cat_col].value_counts()
            
            insights['distribution'] = {
                'most_common': value_counts.index[0],
                'most_common_count': int(value_counts.values[0]),
                'most_common_percentage': round((value_counts.values[0] / len(df)) * 100, 1),
                'unique_categories': len(value_counts),
                'top_3': [
                    {
                        'category': value_counts.index[i],
                        'count': int(value_counts.values[i]),
                        'percentage': round((value_counts.values[i] / len(df)) * 100, 1)
                    }
                    for i in range(min(3, len(value_counts)))
                ]
            }
        
        return insights
    
    def analyze_context(self, prompt: str, df: pd.DataFrame) -> Dict:
        """Enhanced context analysis with better column detection"""
        prompt_lower = prompt.lower()
        
        # Detect intents
        detected_intents = []
        for intent_type, keywords in self.intent_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_intents.append(intent_type)
        
        temporal_cols = []
        text_cols = []
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype
            
            # Temporal columns
            if dtype in ['datetime64[ns]', 'datetime64']:
                temporal_cols.append(col)
            elif any(term in col_lower for term in [
                'date', 'time', 'datum', 'pvm', 'päivä', 'aika',
                'created', 'updated', 'handled', 'käsitelty'
            ]):
                # Try to convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    temporal_cols.append(col)
                except:
                    pass
            
            # Text vs categorical detection
            elif dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                unique_ratio = df[col].nunique() / len(df)
                
                if avg_length > 30 and unique_ratio > 0.3:
                    # Long, unique text
                    text_cols.append(col)
                elif unique_ratio < 0.5:
                    # Repeated values = categorical
                    categorical_cols.append(col)
                elif any(term in col_lower for term in ['observation', 'havainto', 'description', 'kuvaus']):
                    # Explicitly text column
                    text_cols.append(col)
                else:
                    text_cols.append(col)
            
            # Numerical
            elif dtype in ['int64', 'float64']:
                if col_lower not in ['id', 'index']:
                    numerical_cols.append(col)
        
        return {
            'prompt': prompt,
            'prompt_lower': prompt_lower,
            'detected_intents': detected_intents,
            'has_temporal': len(temporal_cols) > 0,
            'has_categorical': len(categorical_cols) > 0,
            'has_text': len(text_cols) > 0,
            'temporal_cols': temporal_cols,
            'categorical_cols': categorical_cols,
            'text_cols': text_cols,
            'numerical_cols': numerical_cols,
            'row_count': len(df),
            'columns': df.columns.tolist()
        }
    
    def select_visualization_type(self, context: Dict) -> str:
        """
        Enhanced: Select based on WHAT ASPECT is being queried
        Same data → different viz based on question
        """
        intents = context['detected_intents']
        prompt_lower = context['prompt_lower']
        
        # DURATION/TIME questions → Histogram
        if 'duration' in intents or any(term in prompt_lower for term in [
            'average time', 'handling time', 'processing time',
            'keskiarvo', 'käsittelyaika', 'how long', 'kauanko'
        ]):
            if len(context['temporal_cols']) >= 2:
                return 'histogram'
        
        # TREND questions → Time series
        if 'temporal' in intents or any(term in prompt_lower for term in [
            'trend', 'trendi', 'over time', 'change', 'muutos', 'kehitys'
        ]):
            if context['temporal_cols']:
                return 'time_series'
        
        # STATUS/DISTRIBUTION questions → Bar chart
        if 'status' in intents or 'distribution' in intents or any(term in prompt_lower for term in [
            'status', 'tila', 'distribution', 'jakauma', 'breakdown'
        ]):
            if context['categorical_cols'] or context['text_cols']:
                return 'bar_chart'
        
        # COUNT/TOP questions → Bar chart
        if 'count' in intents or 'top' in intents or any(term in prompt_lower for term in [
            'how many', 'montako', 'most common', 'yleisimmät', 'eniten'
        ]):
            if context['categorical_cols'] or context['text_cols']:
                return 'bar_chart'
        
        # Default logic
        if context['has_categorical'] or context['has_text']:
            return 'bar_chart'
        
        if context['has_temporal']:
            return 'time_series'
        
        return 'none'
    
    def prepare_data(self, df: pd.DataFrame, viz_type: str, context: Dict) -> Optional[pd.DataFrame]:
        """Data preparation with universal preprocessing"""
        df_copy = df.copy()
        
        # Universal: Convert temporal columns
        for col in context['temporal_cols']:
            if df_copy[col].dtype not in ['datetime64[ns]', 'datetime64']:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        # Universal: Calculate duration if 2+ date columns
        if len(context['temporal_cols']) >= 2:
            start_col = context['temporal_cols'][0]
            end_col = context['temporal_cols'][1]
            df_copy['calculated_duration_days'] = (
                df_copy[end_col] - df_copy[start_col]
            ).dt.total_seconds() / (24 * 3600)
            
            # Remove negative durations
            df_copy = df_copy[df_copy['calculated_duration_days'] >= 0]
        
        # Viz-specific preparation
        if viz_type == 'histogram':
            return self._prepare_histogram_data(df_copy, context)
        elif viz_type == 'time_series':
            return self._prepare_timeseries_data(df_copy, context)
        elif viz_type == 'bar_chart':
            return self._prepare_categorical_data(df_copy, context)
        
        return df_copy
    
    def _prepare_histogram_data(self, df: pd.DataFrame, context: Dict) -> Optional[pd.DataFrame]:
        """Prepare for duration histogram"""
        if 'calculated_duration_days' in df.columns:
            target_col = 'calculated_duration_days'
        elif context['numerical_cols']:
            target_col = context['numerical_cols'][0]
        else:
            return None
        
        # Filter extreme outliers (optional, conservative)
        q99 = df[target_col].quantile(0.99)
        df_filtered = df[df[target_col] <= q99].copy()
        
        return df_filtered[[target_col]]
    
    def _prepare_timeseries_data(self, df: pd.DataFrame, context: Dict) -> Optional[pd.DataFrame]:
        """Prepare for time series"""
        if not context['temporal_cols']:
            return None
        
        date_col = context['temporal_cols'][0]
        df['date_only'] = df[date_col].dt.date
        
        # Aggregate by date
        aggregated = df.groupby('date_only').size().reset_index(name='count')
        aggregated = aggregated.sort_values('date_only')
        aggregated.columns = ['date', 'count']
        
        return aggregated
    
    def _prepare_categorical_data(self, df: pd.DataFrame, context: Dict) -> Optional[pd.DataFrame]:
        """Prepare categorical data - use existing or extract from text"""
        
        # Use existing categorical column
        if context['categorical_cols']:
            cat_col = context['categorical_cols'][0]
            counts = df[cat_col].value_counts().reset_index()
            counts.columns = ['category', 'count']
            counts = counts.head(self.viz_config['max_categories'])
            return counts
        
        # Extract from text
        if context['text_cols']:
            return self._extract_by_tfidf(df, context)
        
        return None
    
    def _extract_by_tfidf(self, df: pd.DataFrame, context: Dict) -> Optional[pd.DataFrame]:
        """TF-IDF based category extraction"""
        text_col = self._select_best_text_column(df, context)
        if text_col is None:
            return None
        
        texts = df[text_col].fillna('').astype(str).tolist()
        
        if len(texts) < 3:
            return None
        
        try:
            ngram_range = (1, 2) if self.viz_config['use_bigrams'] else (1, 1)
            
            vectorizer = TfidfVectorizer(
                max_features=self.viz_config['top_n_words'],
                lowercase=True,
                token_pattern=r'\b\w{' + str(self.viz_config['min_word_length']) + r',}\b',
                ngram_range=ngram_range
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            categories = []
            for i in range(len(texts)):
                doc_scores = tfidf_matrix[i].toarray()[0]
                
                if doc_scores.max() == 0:
                    categories.append('Other')
                    continue
                
                top_idx = doc_scores.argmax()
                top_word = feature_names[top_idx]
                category = top_word.title()
                categories.append(category)
            
            category_counts = Counter(categories)
            result = pd.DataFrame(
                list(category_counts.items()),
                columns=['category', 'count']
            ).sort_values('count', ascending=False)
            
            result = result.head(self.viz_config['max_categories'])
            
            return result
            
        except Exception as e:
            print(f"TF-IDF failed: {e}")
            return self._extract_by_word_frequency(df, text_col)
    
    def _extract_by_word_frequency(self, df: pd.DataFrame, text_col: str) -> Optional[pd.DataFrame]:
        """Fallback: word frequency"""
        all_words = []
        
        for text in df[text_col].fillna('').astype(str):
            words = re.findall(
                r'\b\w{' + str(self.viz_config['min_word_length']) + r',}\b',
                text.lower()
            )
            all_words.extend(words)
        
        if not all_words:
            return None
        
        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(self.viz_config['top_n_words'])]
        
        categories = []
        for text in df[text_col].fillna('').astype(str):
            text_lower = text.lower()
            assigned = False
            
            for word in top_words:
                if word in text_lower:
                    categories.append(word.title())
                    assigned = True
                    break
            
            if not assigned:
                categories.append('Other')
        
        category_counts = Counter(categories)
        result = pd.DataFrame(
            list(category_counts.items()),
            columns=['category', 'count']
        ).sort_values('count', ascending=False)
        
        return result.head(self.viz_config['max_categories'])
    
    def _select_best_text_column(self, df: pd.DataFrame, context: Dict) -> Optional[str]:
        """Select most descriptive text column"""
        if not context['text_cols']:
            return None
        
        # Prefer observation/description columns
        for col in context['text_cols']:
            col_lower = col.lower()
            if any(term in col_lower for term in ['observation', 'havainto', 'description', 'kuvaus']):
                return col
        
        # Otherwise longest text
        max_length = 0
        best_col = None
        for col in context['text_cols']:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > max_length:
                max_length = avg_len
                best_col = col
        
        return best_col
    
    def create_visualization(self, data: pd.DataFrame, viz_type: str, context: Dict) -> Optional[Any]:
        """Create visualization with insights"""
        try:
            if viz_type == 'histogram':
                return self._create_histogram(data, context)
            elif viz_type == 'time_series':
                return self._create_time_series(data, context)
            elif viz_type == 'bar_chart':
                return self._create_bar_chart(data, context)
            elif viz_type == 'pie_chart':
                return self._create_pie_chart(data, context)
            else:
                return None
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def _create_histogram(self, data: pd.DataFrame, context: Dict) -> go.Figure:
        """
        Histogram with clear insights for non-technical users
        Shows same numbers that appear in LLM summary
        """
        num_col = 'calculated_duration_days' if 'calculated_duration_days' in data.columns else data.columns[0]
        
        fig = px.histogram(
            data,
            x=num_col,
            title='Processing Time Distribution',
            labels={num_col: 'Days to Complete'},
            nbins=min(20, len(data)),
            color_discrete_sequence=['#1f77b4']
        )
        
        # Add insights from context
        insights = context.get('insights', {})
        handling_time = insights.get('handling_time', {})
        
        if handling_time:
            avg_val = handling_time.get('average', data[num_col].mean())
            median_val = handling_time.get('median', data[num_col].median())
            
            # Add average line
            fig.add_vline(
                x=avg_val,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Average: {avg_val:.1f} days",
                annotation_position="top right"
            )
            
            # Add median line
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"Median: {median_val:.1f} days",
                annotation_position="bottom right"
            )
            
            # Add text box with key stats
            stats_text = f"""
            <b>Key Statistics:</b><br>
            Total: {insights.get('total_count', len(data))} observations<br>
            Average: {avg_val:.1f} days<br>
            Median: {median_val:.1f} days<br>
            Range: {handling_time.get('min', 0):.1f} - {handling_time.get('max', 0):.1f} days
            """
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=11)
            )
        
        fig.update_layout(
            height=self.viz_config['default_height'],
            template='plotly_white',
            showlegend=False,
            font=dict(size=12),
            title_font_size=16,
            xaxis_title='Days to Complete',
            yaxis_title='Number of Observations'
        )
        
        return fig
    
    def _create_time_series(self, data: pd.DataFrame, context: Dict) -> go.Figure:
        """
        Time series with trend insights for non-technical users
        """
        fig = px.line(
            data,
            x='date',
            y='count',
            title='Observations Over Time',
            labels={'date': 'Date', 'count': 'Number of Observations'},
            markers=True,
            line_shape='linear'
        )
        
        fig.update_traces(
            line_color='#1f77b4',
            line_width=2,
            marker=dict(size=8)
        )
        
        # Add insights
        insights = context.get('insights', {})
        temporal_info = insights.get('temporal', {})
        
        if temporal_info:
            total = insights.get('total_count', 0)
            span_days = temporal_info.get('span_days', 0)
            avg_per_day = temporal_info.get('observations_per_day', 0)
            
            # Peak detection
            peak_date = data.loc[data['count'].idxmax(), 'date']
            peak_count = data['count'].max()
            
            # Add annotation for peak
            fig.add_annotation(
                x=peak_date,
                y=peak_count,
                text=f"Peak: {peak_count} obs.",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=30,
                ay=-40,
                bgcolor="white",
                bordercolor="red"
            )
            
            # Add summary box
            summary_text = f"""
            <b>Summary:</b><br>
            Total: {total} observations<br>
            Period: {span_days} days<br>
            Average: {avg_per_day:.1f} per day<br>
            Peak: {peak_count} on {peak_date}
            """
            
            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=11)
            )
        
        fig.update_layout(
            height=self.viz_config['default_height'],
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def _create_bar_chart(self, data: pd.DataFrame, context: Dict) -> go.Figure:
        """
        Bar chart with clear labels and insights for non-technical users
        """
        x_col = data.columns[0]
        y_col = 'count'
        
        # Sort by count
        data_sorted = data.sort_values(y_col, ascending=False).head(self.viz_config['max_categories'])
        
        fig = px.bar(
            data_sorted,
            x=x_col,
            y=y_col,
            title=f'Distribution by {x_col.replace("_", " ").title()}',
            labels={
                x_col: x_col.replace('_', ' ').title(),
                y_col: 'Number of Observations'
            },
            color=y_col,
            color_continuous_scale=self.viz_config['color_scheme'],
            text=y_col
        )
        
        # Show numbers on bars
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=12
        )
        
        # Add insights
        insights = context.get('insights', {})
        distribution = insights.get('distribution', {})
        
        total = insights.get('total_count', data_sorted[y_col].sum())
        
        # Add summary annotation
        if distribution:
            most_common = distribution.get('most_common', '')
            most_common_pct = distribution.get('most_common_percentage', 0)
            
            summary_text = f"""
            <b>Summary:</b><br>
            Total: {total} observations<br>
            Most common: {most_common}<br>
            ({most_common_pct:.1f}% of total)
            """
        else:
            summary_text = f"""
            <b>Total:</b> {total} observations<br>
            Showing top {len(data_sorted)} categories
            """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        )
        
        fig.update_layout(
            height=self.viz_config['default_height'],
            xaxis_tickangle=-45,
            template='plotly_white',
            showlegend=False,
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, context: Dict) -> go.Figure:
        """
        Pie chart for proportions (non-technical friendly)
        """
        label_col = data.columns[0]
        value_col = 'count'
        
        fig = px.pie(
            data,
            names=label_col,
            values=value_col,
            title=f'Distribution by {label_col.replace("_", " ").title()}'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12
        )
        
        # Add total in subtitle
        total = data[value_col].sum()
        fig.update_layout(
            height=self.viz_config['default_height'],
            template='plotly_white',
            font=dict(size=12),
            title={
                'text': f'{label_col.replace("_", " ").title()} Distribution<br><sub>Total: {total} observations</sub>',
                'x': 0.5,
                'xanchor': 'center'
            }
        )
        
        return fig


# Convenience function
def generate_visualization(prompt: str, dataframe: pd.DataFrame) -> Optional[Any]:
    """
    One-line usage for easy integration
    
    Example:
        from context2visual import generate_visualization
        fig = generate_visualization(user_prompt, df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    """
    generator = VisualizationGenerator()
    return generator.generate_visualization(prompt, dataframe)