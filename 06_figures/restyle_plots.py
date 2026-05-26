#!/usr/bin/env python
"""
Re-styles and exports existing UMAP HTML plots to high-quality PNGs.

This script scans the static and dynamic output directories for Plotly UMAP
visualizations (.html files), applies a clean, minimalist, publication-ready
stylesheet to them, and saves the results as high-resolution PNG images.

This is a lightweight utility script designed to be run after the main
analysis scripts, allowing for rapid iteration on visual styles without
needing to re-run the time-consuming embedding and analysis steps.
"""
from __future__ import annotations

import json
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
from bs4 import BeautifulSoup


def restyle_and_export_plot(html_path: pathlib.Path, output_dir: pathlib.Path):
    """Loads a Plotly HTML file, applies styling, and saves as PNG to a specific directory."""
    try:
        # Read the HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse the HTML to find the JSON data for the plot
        soup = BeautifulSoup(html_content, 'lxml')
        plotly_script = soup.find('script', {'type': 'application/json'})
        
        if not plotly_script:
            raise ValueError("Could not find Plotly JSON data in the HTML file.")
            
        fig_data = json.loads(plotly_script.string)
        
        # Create a figure from the extracted data
        fig = go.Figure(fig_data)

        # Apply the minimalist, high-contrast styling
        fig.update_traces(
            opacity=0.8,
            marker=dict(size=5)
        )
        fig.update_layout(
            title_text='',      # Remove title
            showlegend=False,   # Remove legend
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,  # Hide entire x-axis
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,  # Hide entire y-axis
            ),
            plot_bgcolor='rgba(255, 255, 255, 1)', # Explicitly set white background
        )

        # Define the output path for the PNG in the specified output directory
        png_path = output_dir / html_path.with_suffix('.png').name
        
        # Save the styled figure as a high-resolution PNG
        fig.write_image(png_path, scale=3)
        
        print(f"Successfully re-styled and saved to {png_path}")

    except Exception as e:
        print(f"Failed to process {html_path.name}: {e}")


def main():
    """Main function to find and process all UMAP plots."""
    print("--- Starting UMAP Plot Re-styling and Export ---")
    
    # Define directories to scan and the new output directory for PNGs
    static_dir = pathlib.Path("data/outputs/analysis/static")
    dynamic_dir = pathlib.Path("data/outputs/analysis/dynamic")
    png_output_dir = pathlib.Path("data/outputs/analysis/styled_plots")
    
    # Create the output directory if it doesn't exist
    png_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all UMAP html files
    static_plots = list(static_dir.glob("umap_visualization_*.html"))
    dynamic_plots = list(dynamic_dir.glob("umap_visualization_*.html"))
    all_plots = static_plots + dynamic_plots
    
    if not all_plots:
        print("No UMAP HTML plots found to re-style.")
        return
        
    print(f"Found {len(all_plots)} UMAP plots to process...")
    
    for plot_path in all_plots:
        restyle_and_export_plot(plot_path, png_output_dir)
        
    print(f"\n--- Re-styling complete. Styled PNGs are in {png_output_dir} ---")


if __name__ == "__main__":
    main() 