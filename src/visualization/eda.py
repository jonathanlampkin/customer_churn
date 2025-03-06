"""
Enhanced Exploratory Data Analysis for Churn Modeling
Uses modern visualization libraries and techniques.
"""

import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import logging
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import List, Dict, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)

class ChurnEDA:
    """
    Class for comprehensive exploratory data analysis on churn dataset.
    """
    
    def __init__(self, config_path: str = 'configs/config.yml'):
        """Initialize with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.df = None
        self.target_col = self.config['training']['target_column']
        
        # Create output directory for visualizations
        os.makedirs('reports/figures', exist_ok=True)
        logger.info("EDA module initialized")
    
    def load_data(self, processed: bool = True) -> pl.DataFrame:
        """
        Load either raw or processed data.
        
        Args:
            processed: Whether to load processed data (True) or raw data (False)
            
        Returns:
            Loaded DataFrame
        """
        if processed:
            file_path = self.config['data']['processed_path']
            logger.info(f"Loading processed data from {file_path}")
            self.df = pl.read_parquet(file_path)
        else:
            file_path = self.config['data']['raw_path']
            logger.info(f"Loading raw data from {file_path}")
            
            import pandas as pd
            from scipy.io import arff
            
            data, meta = arff.loadarff(file_path)
            pdf = pd.DataFrame(data)
            
            # Convert byte strings to regular strings
            string_columns = [col for col in pdf.columns if pdf[col].dtype == object]
            for col in string_columns:
                pdf[col] = pdf[col].str.decode('utf-8')
            
            self.df = pl.from_pandas(pdf)
        
        logger.info(f"Data loaded with shape: {self.df.shape}")
        return self.df
    
    def basic_stats(self) -> Dict:
        """
        Compute basic statistics for the dataset.
        
        Returns:
            Dictionary of statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        stats = {}
        
        # Basic shape information
        stats['n_rows'] = self.df.height
        stats['n_columns'] = self.df.width
        
        # Column type counts
        dtype_counts = {}
        for _, dtype in self.df.schema.items():
            dtype_str = str(dtype)
            if dtype_str in dtype_counts:
                dtype_counts[dtype_str] += 1
            else:
                dtype_counts[dtype_str] = 1
        stats['dtype_counts'] = dtype_counts
        
        # Target distribution
        if self.target_col in self.df.columns:
            target_counts = self.df.group_by(self.target_col).agg(
                pl.count().alias('count')
            )
            
            target_values = target_counts[self.target_col].to_list()
            target_counts = target_counts['count'].to_list()
            stats['target_distribution'] = dict(zip(target_values, target_counts))
            
            # Calculate class imbalance
            if len(target_counts) == 2:
                imbalance_ratio = max(target_counts) / min(target_counts)
                stats['class_imbalance_ratio'] = imbalance_ratio
        
        # Missing value summary
        null_counts = self.df.null_count()
        total_nulls = sum(null_counts.values())
        stats['total_nulls'] = total_nulls
        stats['columns_with_nulls'] = {k: v for k, v in null_counts.items() if v > 0}
        
        # Numerical column statistics
        numeric_cols = [name for name, dtype in self.df.schema.items() 
                      if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        stats['numerical_stats'] = {}
        for col in numeric_cols:
            col_stats = {
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'skew': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis())
            }
            stats['numerical_stats'][col] = col_stats
        
        logger.info("Basic statistics calculated")
        return stats
    
    def plot_target_distribution(self, interactive: bool = True) -> None:
        """
        Plot the distribution of the target variable.
        
        Args:
            interactive: Whether to create an interactive plotly visualization
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.target_col not in self.df.columns:
            logger.warning(f"Target column '{self.target_col}' not found in data")
            return
        
        # Get target counts
        target_counts = self.df.group_by(self.target_col).agg(
            pl.count().alias('count')
        ).with_columns(
            (pl.col('count') / self.df.height * 100).alias('percentage')
        )
        
        # Log the distribution
        logger.info(f"Target distribution:\n{target_counts}")
        
        # Create static visualization
        plt.figure(figsize=(10, 6))
        
        # Convert to pandas for easier plotting
        target_counts_pd = target_counts.to_pandas()
        
        ax = sns.barplot(x=self.target_col, y='percentage', data=target_counts_pd)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=12)
        
        plt.title(f'Distribution of {self.target_col}', fontsize=15)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xlabel(self.target_col, fontsize=12)
        plt.tight_layout()
        plt.savefig(f'reports/figures/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive visualization if requested
        if interactive:
            fig = px.bar(
                target_counts_pd, 
                x=self.target_col, 
                y='percentage',
                text=target_counts_pd['percentage'].apply(lambda x: f'{x:.1f}%'),
                title=f'Distribution of {self.target_col}',
                labels={'percentage': 'Percentage (%)'},
                color=self.target_col,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            fig.update_layout(
                title_font_size=20,
                xaxis_title_font_size=15,
                yaxis_title_font_size=15
            )
            
            # Save as HTML
            fig.write_html('reports/figures/target_distribution_interactive.html')
        
        logger.info("Target distribution plot created")
    
    def plot_correlation_heatmap(self, interactive: bool = True, method: str = 'pearson') -> None:
        """
        Create a correlation heatmap for numerical features.
        
        Args:
            interactive: Whether to create an interactive plotly visualization
            method: Correlation method ('pearson', 'spearman', or 'kendall')
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select only numeric columns
        numeric_cols = [name for name, dtype in self.df.schema.items() 
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for correlation analysis")
            return
        
        logger.info(f"Creating correlation heatmap with {method} method")
        
        # Convert to pandas for correlation calculation
        pdf = self.df.select(numeric_cols).to_pandas()
        
        # Calculate correlation matrix
        correlation_matrix = pdf.corr(method=method)
        
        # Create static visualization
        plt.figure(figsize=(16, 14))
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap
        sns.heatmap(
            correlation_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            annot=False, 
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title(f'{method.capitalize()} Correlation Heatmap', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'reports/figures/correlation_heatmap_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive visualization if requested
        if interactive:
            # Create a beautiful correlation heatmap with plotly
            fig = px.imshow(
                correlation_matrix,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title=f'{method.capitalize()} Correlation Heatmap'
            )
            
            fig.update_layout(
                title_font_size=20,
                xaxis_title_font_size=15,
                yaxis_title_font_size=15,
                height=800,
                width=900
            )
            
            # Save as HTML
            fig.write_html(f'reports/figures/correlation_heatmap_{method}_interactive.html')
        
        logger.info("Correlation analysis plots created")
    
    def plot_feature_importance_to_target(self, n_features: int = 20, interactive: bool = True) -> None:
        """
        Calculate and plot feature importance relative to the target.
        
        Args:
            n_features: Number of top features to display
            interactive: Whether to create interactive plots
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.target_col not in self.df.columns:
            logger.warning(f"Target column '{self.target_col}' not found in data")
            return
        
        # Select only numeric columns
        numeric_cols = [name for name, dtype in self.df.schema.items() 
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric features found for importance analysis")
            return
        
        logger.info("Calculating feature importance to target...")
        
        # Use Random Forest for feature importance
        from sklearn.ensemble import RandomForestClassifier
        
        # Convert to pandas for sklearn
        X = self.df.select(numeric_cols).to_pandas()
        y = self.df.select(self.target_col).to_pandas()
        
        # Train a random forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y.values.ravel())
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create dataframe of features and importances
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = feature_importance.head(n_features)
        
        # Create static visualization
        plt.figure(figsize=(12, 8))
        
        ax = sns.barplot(x='importance', y='feature', data=top_features)
        
        # Add value labels to bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_width():.3f}', 
                       (p.get_width(), p.get_y() + p.get_height() / 2.), 
                       ha='left', va='center', fontsize=10, xytext=(5, 0), 
                       textcoords='offset points')
        
        plt.title(f'Top {n_features} Features by Importance to {self.target_col}', fontsize=15)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive visualization if requested
        if interactive:
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f'Top {n_features} Features by Importance to {self.target_col}',
                labels={'importance': 'Importance', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                title_font_size=20,
                xaxis_title_font_size=15,
                yaxis_title_font_size=15
            )
            
            # Save as HTML
            fig.write_html('reports/figures/feature_importance_interactive.html')
        
        logger.info("Feature importance plot created")
        
        return feature_importance
    
    def plot_dimensionality_reduction(self, method: str = 'pca', interactive: bool = True) -> None:
        """
        Create dimensionality reduction plots (PCA, t-SNE, or UMAP) colored by target.
        
        Args:
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
            interactive: Whether to create interactive plots
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.target_col not in self.df.columns:
            logger.warning(f"Target column '{self.target_col}' not found in data")
            return
        
        # Select only numeric columns
        numeric_cols = [name for name, dtype in self.df.schema.items() 
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric features for dimensionality reduction")
            return
        
        logger.info(f"Creating {method.upper()} dimensionality reduction plot...")
        
        # Convert to pandas for sklearn
        X = self.df.select(numeric_cols).to_pandas()
        y = self.df.select(self.target_col).to_pandas()
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_scaled)
            variance_explained = reducer.explained_variance_ratio_
            title = f'PCA Projection (Explained Variance: {variance_explained[0]:.2f}, {variance_explained[1]:.2f})'
        
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            embedding = reducer.fit_transform(X_scaled)
            title = 't-SNE Projection'
        
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_scaled)
            title = 'UMAP Projection'
        
        else:
            logger.error(f"Unknown dimensionality reduction method: {method}")
            return
        
        # Create dataframe for plotting
        projection_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'target': y[self.target_col]
        })
        
        # Create static visualization
        plt.figure(figsize=(12, 10))
        
        scatter = sns.scatterplot(
            x='x', 
            y='y', 
            hue='target', 
            data=projection_df,
            alpha=0.7,
            s=50,
            palette='viridis'
        )
        
        plt.title(title, fontsize=15)
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        
        # Add legend with title
        scatter.legend(title=self.target_col)
        
        plt.tight_layout()
        plt.savefig(f'reports/figures/{method}_projection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive visualization if requested
        if interactive:
            fig = px.scatter(
                projection_df,
                x='x',
                y='y',
                color='target',
                title=title,
                labels={
                    'x': f'{method.upper()} Component 1',
                    'y': f'{method.upper()} Component 2',
                    'target': self.target_col
                },
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            fig.update_layout(
                title_font_size=20,
                xaxis_title_font_size=15,
                yaxis_title_font_size=15
            )
            
            # Save as HTML
            fig.write_html(f'reports/figures/{method}_projection_interactive.html')
        
        logger.info(f"{method.upper()} projection plot created")
    
    def run_all_analyses(self) -> None:
        """Run all EDA analyses and create visualizations."""
        logger.info("Running all EDA analyses...")
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Run analyses
        self.basic_stats()
        self.plot_target_distribution()
        self.plot_correlation_heatmap(method='pearson')
        self.plot_correlation_heatmap(method='spearman')
        self.plot_feature_importance_to_target()
        
        # Run dimensionality reduction
        self.plot_dimensionality_reduction(method='pca')
        self.plot_dimensionality_reduction(method='tsne')
        self.plot_dimensionality_reduction(method='umap')
        
        logger.info("All EDA analyses completed")

def main():
    """Main execution function"""
    eda = ChurnEDA()
    eda.run_all_analyses()
    
    logger.info("EDA completed successfully")

if __name__ == "__main__":
    main() 