import pandas as pd
import os

def escape_latex(text):
    """
    Escapes special LaTeX characters in a string.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Escape backslashes first
    text = text.replace('\\', r'\textbackslash{}')
    # Escape other special characters
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')
    text = text.replace("'", "'") # Apostrophe
    text = text.replace('"', "''") # Double quotes
    
    return text

def floats_to_str(df, precision=3):
    """
    Converts all float columns in a DataFrame to strings with a specified precision.
    """
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].apply(lambda x: f"{x:.{precision}f}")
    return df

def df_to_latex(df, output_file, caption, label, single_page, text_col_width='8cm', columns=None):
    """
    Converts a pandas DataFrame to a LaTeX table, saves it to a .tex file.
    Can create a multi-page longtable or a single-page tabular.
    """
    if columns:
        df = df[columns]

    num_cols = len(df.columns)
    
    # Define column specifications
    col_specs = ""
    for col_name in df.columns:
        if col_name == 'text':
            col_specs += f"p{{{text_col_width}}} "
        elif pd.api.types.is_numeric_dtype(df[col_name]) or col_name == 'distinction_score':
            col_specs += "r "
        else:
            col_specs += "l "
            
    with open(output_file, 'w', encoding='utf-8') as f:
        if single_page:
            f.write("% To use this table in your main LaTeX document, include the following packages:\n")
            f.write("% \\usepackage{booktabs}\n")
            f.write("% To place in a normal page, use 'table'. For a full-page width table, use 'table*'.\n")
            f.write("% \\begin{table}[htbp]\n")
            f.write("%   \\centering\n")
            f.write(f"%   \\caption{{{caption}}}\n")
            f.write(f"%   \\label{{{label}}}\n")
            f.write(f"%   \\input{{{output_file}}}\n")
            f.write("% \\end{table}\n\n")

            # Table start
            f.write("\\begin{tabular}{" + col_specs.strip() + "}\n")
            f.write("\\toprule\n")
            
            # Header
            header = " & ".join([escape_latex(col.replace('_', ' ').title()) for col in df.columns])
            f.write(header + " \\\\\n")
            f.write("\\midrule\n")
            
            # Body
            df = floats_to_str(df.copy())
            for _, row in df.iterrows():
                row_values = [escape_latex(cell) for cell in row]
                f.write(" & ".join(row_values) + " \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
        else: # longtable logic
            f.write("% To use this table in your main LaTeX document, include the following packages:\n")
            f.write("% \\usepackage{longtable}\n")
            f.write("% \\usepackage{booktabs}\n")
            f.write("% \\usepackage{pdflscape}\n")
            f.write("% Then, in your appendix, use the following code:\n")
            f.write("% \\begin{landscape}\n")
            f.write("%   \\centering\n")
            f.write(f"%   \\captionof{{table}}{{{caption}}}\n")
            f.write(f"%   \\label{{{label}}}\n")
            f.write(f"%   \\input{{{output_file}}}\n")
            f.write("% \\end{landscape}\n\n")

            # Table start
            f.write("\\begin{longtable}{" + col_specs.strip() + "}\n")
            f.write(f"\\caption[]{{{caption}}} \\label{{{label}}} \\\\\n")
            f.write("\\toprule\n")
            
            # Header
            header = " & ".join([escape_latex(col.replace('_', ' ').title()) for col in df.columns])
            f.write(header + " \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endfirsthead\n")
            
            # Continued table header
            f.write(f"\\caption[]{{(continued)}} \\\\\n")
            f.write("\\toprule\n")
            f.write(header + " \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endhead\n")
            
            # Footer for all but last page
            f.write("\\midrule\n")
            f.write(f"\\multicolumn{{{num_cols}}}{{r}}{{(Continued on next page)}} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\endfoot\n")

            # Footer for last page
            f.write("\\bottomrule\n")
            f.write("\\endlastfoot\n")
            
            # Body
            df = floats_to_str(df.copy())
            for _, row in df.iterrows():
                row_values = [escape_latex(cell) for cell in row]
                f.write(" & ".join(row_values) + " \\\\\n")
                
            # Table end
            f.write("\\end{longtable}\n")

def main():
    """
    Main function to generate LaTeX tables from CSV files.
    """
    input_dir = 'data/outputs/analysis'
    output_dir = 'data/outputs/analysis'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Compact Conclusion-style Tables (Top 10) ---
    conclusion_cols = ['distinction_score', 'text', 'source']
    
    # Process first file for conclusion
    csv1_path = os.path.join(input_dir, 'leaderboard_model_based_top_50.csv')
    tex1_conclusion_path = os.path.join(output_dir, 'leaderboard_top_10_conclusion.tex')
    df1_c = pd.read_csv(csv1_path).head(10)
    df_to_latex(
        df1_c, tex1_conclusion_path, 
        caption='Top 10 Quotes by Distinction Score (with Sentiment).', 
        label='tab:conclusion_top10_with_sentiment',
        single_page=True,
        text_col_width='0.6\\textwidth',
        columns=conclusion_cols
    )
    print(f"Generated Conclusion table at {tex1_conclusion_path}")

    # Process second file for conclusion
    csv2_path = os.path.join(input_dir, 'leaderboard_model_based_top_50_NoSentiment.csv')
    tex2_conclusion_path = os.path.join(output_dir, 'leaderboard_top_10_NoSentiment_conclusion.tex')
    df2_c = pd.read_csv(csv2_path).head(10)
    df_to_latex(
        df2_c, tex2_conclusion_path, 
        caption='Top 10 Quotes by Distinction Score (without Sentiment).',
        label='tab:conclusion_top10_no_sentiment',
        single_page=True,
        text_col_width='0.6\\textwidth',
        columns=conclusion_cols
    )
    print(f"Generated Conclusion table at {tex2_conclusion_path}")

    # --- Full Appendix-style Tables (Top 20) ---
    # Process first file
    tex1_appendix_path = os.path.join(output_dir, 'leaderboard_model_based_top_20_appendix.tex')
    df1_a = pd.read_csv(csv1_path).head(20)
    if all(col in df1_a.columns for col in ['ff', 'ff_x', 'ff_y']):
      df1_a = df1_a.drop(columns=['ff_x', 'ff_y'])

    df_to_latex(
        df1_a, tex1_appendix_path, 
        caption='Model-Based Top 20 Leaderboard with Sentiment Metrics.', 
        label='tab:appendix_top20_with_sentiment',
        single_page=True,
        text_col_width='7.5cm'
    )
    print(f"Generated Appendix table at {tex1_appendix_path}")

    # Process second file
    tex2_appendix_path = os.path.join(output_dir, 'leaderboard_model_based_top_20_NoSentiment_appendix.tex')
    df2_a = pd.read_csv(csv2_path).head(20)
    if all(col in df2_a.columns for col in ['ff', 'ff_x', 'ff_y']):
      df2_a = df2_a.drop(columns=['ff_x', 'ff_y'])

    df_to_latex(
        df2_a, tex2_appendix_path, 
        caption='Model-Based Top 20 Leaderboard without Sentiment Metrics.',
        label='tab:appendix_top20_no_sentiment',
        single_page=True,
        text_col_width='8.5cm'
    )
    print(f"Generated Appendix table at {tex2_appendix_path}")


if __name__ == '__main__':
    main() 