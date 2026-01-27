"""
Chart generation and data analysis service - FIXED VERSION
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os
import logging
import math
from typing import Tuple, Dict, Optional

from config import Config

logger = logging.getLogger(__name__)

# Configure matplotlib
plt.switch_backend('Agg')
plt.style.use(Config.CHART_STYLE)

# Global data store
excel_data_store = {}

def clean_for_json(obj):
    """
    FIXED: Recursively clean data for JSON serialization
    Replaces NaN, inf, -inf with None
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj

def process_data_file(filepath, filename: str) -> Dict:
    """Process uploaded data file (CSV/Excel) - FIXED VERSION"""
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        
        # Read file
        if ext == 'csv':
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # CRITICAL FIX: Replace NaN/inf values immediately after reading
        df = df.replace([np.nan, np.inf, -np.inf], None)
        
        # Store in memory
        excel_data_store[filename] = df
        
        logger.info(f"📊 Data file processed: {filename} - {len(df)} rows, {len(df.columns)} columns")
        
        # FIXED: Clean all data before returning
        result = {
            'message': f'Data file uploaded successfully - {len(df)} rows, {len(df.columns)} columns',
            'filename': filename,
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'data_types': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'sample_data': df.head(3).to_dict()
        }
        
        # Clean the result recursively
        result = clean_for_json(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing data file: {str(e)}")
        raise

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get numeric column names"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def suggest_chart_columns(df: pd.DataFrame, question: str) -> Tuple:
    """
    Suggest appropriate chart type and columns based on question
    
    Returns:
        (chart_type, x_column, y_column, reasoning)
    """
    numeric_cols = get_numeric_columns(df)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols:
        return None, None, None, "No numeric columns found"
    
    question_lower = question.lower()
    
    # Chart type detection
    if 'pie' in question_lower and categorical_cols:
        return 'pie', categorical_cols[0], numeric_cols[0], "Pie chart showing distribution"
    elif 'line' in question_lower and len(numeric_cols) >= 2:
        return 'line', numeric_cols[0], numeric_cols[1], "Line chart showing trends"
    elif 'scatter' in question_lower and len(numeric_cols) >= 2:
        return 'scatter', numeric_cols[0], numeric_cols[1], "Scatter plot showing correlation"
    elif 'histogram' in question_lower or 'distribution' in question_lower:
        return 'histogram', None, numeric_cols[0], "Histogram showing distribution"
    elif categorical_cols and numeric_cols:
        return 'bar', categorical_cols[0], numeric_cols[0], "Bar chart with categorical and numeric data"
    elif len(numeric_cols) >= 2:
        return 'scatter', numeric_cols[0], numeric_cols[1], "Scatter plot with two numeric columns"
    else:
        return 'histogram', None, numeric_cols[0], "Distribution histogram"

def generate_smart_chart(df: pd.DataFrame, filename: str, question: str) -> Tuple[Optional[str], str]:
    """
    Generate chart with improved visualization
    
    Returns:
        (chart_url, chart_info)
    """
    chart_type, x_col, y_col, reasoning = suggest_chart_columns(df, question)
    
    if not chart_type:
        return None, reasoning
    
    try:
        chart_id = uuid.uuid4().hex
        
        # Setup plot
        plt.figure(figsize=(12, 8))
        
        # Color palette
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
        
        # Generate chart based on type
        if chart_type == 'bar' and x_col and y_col:
            _create_bar_chart(df, x_col, y_col, colors)
            
        elif chart_type == 'scatter' and x_col and y_col:
            _create_scatter_chart(df, x_col, y_col, colors)
            
        elif chart_type == 'histogram' and y_col:
            _create_histogram(df, y_col, colors)
            
        elif chart_type == 'pie' and x_col and y_col:
            _create_pie_chart(df, x_col, y_col, colors)
            
        elif chart_type == 'line' and x_col and y_col:
            _create_line_chart(df, x_col, y_col, colors)
        
        # Save chart
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        chart_path = Config.PLOTS_FOLDER / f"{chart_id}.png"
        plt.savefig(
            str(chart_path),
            dpi=Config.CHART_DPI,
            bbox_inches='tight',
            facecolor='#1a1f2e',
            edgecolor='none',
            transparent=False
        )
        plt.close()
        
        if os.path.exists(chart_path):
            logger.info(f"✓ Chart generated: {chart_id}.png")
            chart_url = f"http://localhost:5000/api/files/plots/{chart_id}.png"
            return chart_url, f"Generated {chart_type} chart: {reasoning}"
        else:
            return None, "Failed to save chart"
            
    except Exception as e:
        plt.close()
        logger.error(f"Chart generation error: {str(e)}")
        return None, f"Error: {str(e)}"

def _create_bar_chart(df, x_col, y_col, colors):
    """Create bar chart"""
    if df[x_col].dtype == 'object':
        grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
        bars = plt.bar(range(len(grouped)), grouped.values, color=colors[0], alpha=0.8)
        plt.xticks(range(len(grouped)), grouped.index, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    else:
        plt.bar(df[x_col], df[y_col], color=colors[0], alpha=0.8)
    
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'{y_col} by {x_col}', fontsize=14, fontweight='bold')

def _create_scatter_chart(df, x_col, y_col, colors):
    """Create scatter plot"""
    plt.scatter(df[x_col], df[y_col], alpha=0.7, color=colors[0], s=60)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
    
    # Add trend line
    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) > 1:
        z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
        p = np.poly1d(z)
        plt.plot(valid_data[x_col], p(valid_data[x_col]), "--", color=colors[1], alpha=0.8)

def _create_histogram(df, y_col, colors):
    """Create histogram"""
    plt.hist(df[y_col].dropna(), bins=20, color=colors[0], alpha=0.7, edgecolor='white')
    plt.xlabel(y_col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {y_col}', fontsize=14, fontweight='bold')

def _create_pie_chart(df, x_col, y_col, colors):
    """Create pie chart"""
    if df[x_col].dtype == 'object':
        pie_data = df.groupby(x_col)[y_col].sum().head(8)
        plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        plt.title(f'{y_col} Distribution by {x_col}', fontsize=14, fontweight='bold')

def _create_line_chart(df, x_col, y_col, colors):
    """Create line chart"""
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=6, color=colors[0])
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'{y_col} over {x_col}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)