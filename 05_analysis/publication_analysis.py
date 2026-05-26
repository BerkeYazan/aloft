
import argparse
import logging
import pathlib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
import warnings
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# Suppress the specific UserWarning from XGBoost that clutters the console
# We use a more forceful method to ensure a clean console during grid search.
# warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

def get_col_name(prefix: str, stem: str) -> str:
    """
    Gets the correct column name for a given metric, handling exceptions.
    The 'lex_div' metric has inconsistent naming across sources.
    """
    if stem == 'lex_div':
        # Handle the special naming convention for lexical diversity columns
        special_names = {
            'sample': 'sample_lex_div', 'popular': 'popular_lex_div',
            'snippet': 'matched_snippet_lex_div', 't50': 't50_quote_lex_div',
            't50freelength': 't50_quote_free_context_length_matched_lex_div',
            'nonlit': 'nonlit_baseline_lex_div', 'page': 'page_text_lex_div',
            't50free': 't50_quote_free_context_lex_div'
        }
        return special_names.get(prefix, f"{prefix}_{stem}")
    
    # Default naming convention for all other metrics
    return f"{prefix}_{stem}"

class PublicationAnalysis:
    """
    A class to perform, evaluate, and explain classification tasks
    on the ALOFT dataset to identify features of extraordinary language.
    """

    def __init__(self, data_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Initializes the analysis with data and output paths.

        Args:
            data_path: Path to the ALOFT_master_metrics.csv file.
            output_dir: Directory to save all generated reports and plots.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(f"Loading data from {data_path}")
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            LOG.error(f"FATAL: Data file not found at {data_path}")
            raise

        # Generic metric names (without prefixes like 'sample_')
        self.metric_names = self._extract_metric_names()
        LOG.info(f"Identified {len(self.metric_names)} unique metric stems.")

    def _extract_metric_names(self) -> list[str]:
        """
        Identifies the base metric names from the DataFrame columns.
        
        Example: 'sample_flesch' -> 'flesch', 'popular_pmi' -> 'pmi'.
        It uses the 'sample_' prefix as a reference.
        """
        sample_cols = [col for col in self.df.columns if col.startswith("sample_")]
        return [col.replace("sample_", "") for col in sample_cols]

    def _prepare_data(
        self,
        pos_prefixes: list[str],
        neg_prefixes: list[str],
        excluded_metrics: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepares the feature matrix (X) and target vector (y) for a classification task.

        This function creates a standardized feature set by selecting metrics for positive
        and negative classes from multiple source prefixes, renaming them to a generic 
        format, and combining them.

        Args:
            pos_prefixes: A list of prefixes for the positive class columns (e.g., ['popular', 'sample']).
            neg_prefixes: A list of prefixes for the negative class columns (e.g., ['snippet', 'nonlit']).
            excluded_metrics: A list of base metric names to exclude from the analysis.

        Returns:
            A tuple containing the feature DataFrame (X) and the label Series (y).
        """
        LOG.info(f"Preparing data for task: {pos_prefixes} (positive) vs. {neg_prefixes} (negative)")

        metrics_to_use = self.metric_names
        if excluded_metrics:
            LOG.info(f"Excluding metrics: {excluded_metrics}")
            metrics_to_use = [m for m in self.metric_names if m not in excluded_metrics]
        
        # --- Positive Class (Label=1) ---
        all_pos_dfs = []
        for prefix in pos_prefixes:
            pos_cols = {stem: get_col_name(prefix, stem) for stem in metrics_to_use}
            if not all(col in self.df.columns for col in pos_cols.values()):
                LOG.warning(f"Skipping positive source '{prefix}' due to missing metric columns.")
                continue
            df_pos_single = self.df[list(pos_cols.values())].copy().rename(columns={v: k for k, v in pos_cols.items()})
            all_pos_dfs.append(df_pos_single)
        
        if not all_pos_dfs:
            raise ValueError("No valid data found for any positive class prefixes.")
        df_pos = pd.concat(all_pos_dfs, ignore_index=True)
        df_pos['label'] = 1

        # --- Negative Class (Label=0) ---
        all_neg_dfs = []
        for prefix in neg_prefixes:
            neg_cols = {stem: get_col_name(prefix, stem) for stem in metrics_to_use}
            if not all(col in self.df.columns for col in neg_cols.values()):
                LOG.warning(f"Skipping negative source '{prefix}' due to missing metric columns.")
                continue
            df_neg_single = self.df[list(neg_cols.values())].copy().rename(columns={v: k for k, v in neg_cols.items()})
            all_neg_dfs.append(df_neg_single)

        if not all_neg_dfs:
            raise ValueError("No valid data found for any negative class prefixes.")
        df_neg = pd.concat(all_neg_dfs, ignore_index=True)
        df_neg['label'] = 0

        # --- Align Columns & Combine ---
        common_metrics = list(set(df_pos.columns) & set(df_neg.columns))
        common_metrics.remove('label')

        if 'sentiment_label' in common_metrics:
            common_metrics.remove('sentiment_label')
        
        LOG.info(f"Using {len(common_metrics)} common metrics for this task.")
        
        combined_df = pd.concat([
            df_pos[['label'] + common_metrics],
            df_neg[['label'] + common_metrics]
        ], ignore_index=True).dropna()

        X = combined_df[common_metrics]
        y = combined_df['label']

        LOG.info(f"Data prepared. Shape: {X.shape}. Class distribution: {y.value_counts(normalize=True).to_dict()}")
        return X, y

    def train_and_get_model(
        self,
        pos_prefixes: list[str],
        neg_prefixes: list[str],
        excluded_metrics: list[str] | None = None
    ) -> tuple[xgb.XGBClassifier, list[str]]:
        """
        Trains and returns an XGBoost model for a given task.

        This function encapsulates the data preparation, hyperparameter tuning,
        and final model training, returning a robust classifier ready for inference.

        Args:
            pos_prefixes: A list of prefixes for the positive class (e.g., ['popular', 'sample']).
            neg_prefixes: A list of prefixes for the negative class (e.g., ['snippet', 'nonlit']).
            excluded_metrics: A list of base metric names to exclude.

        Returns:
            A tuple containing the trained XGBoost model and the list of feature names used.
        """
        LOG.info(f"Starting model training pipeline for task: '{pos_prefixes}' vs. '{neg_prefixes}'")

        # 1. Prepare data using the class's internal method
        X, y = self._prepare_data(pos_prefixes, neg_prefixes, excluded_metrics)
        feature_names = X.columns.tolist()

        # 2. Hyperparameter Tuning with GridSearchCV
        LOG.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 250], 'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=cv, verbose=0, n_jobs=-1)
        
        # Suppress noisy output from grid search
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = open(os.devnull, 'w'), open(os.devnull, 'w')
        try:
            grid_search.fit(X, y) # Fit on the full dataset to find best params
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout, sys.stderr = original_stdout, original_stderr

        LOG.info(f"Best parameters found: {grid_search.best_params_}")
        
        # 3. Train the final model using the best parameters on the full dataset
        LOG.info("Training final model with best parameters on the full dataset.")
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            **grid_search.best_params_
        )
        final_model.fit(X, y)

        LOG.info("Model training complete.")
        return final_model, feature_names

    def run_experiment(
        self,
        task_name: str,
        pos_prefix: list[str] | str = None,
        neg_prefix: list[str] | str = None,
        pos_prefixes: list[str] | str = None,
        neg_prefixes: list[str] | str = None,
        excluded_metrics: list[str] | None = None
    ):
        """
        Runs a full classification experiment: tune, train, evaluate, and explain.

        Args:
            task_name: A descriptive name for the task (used for output directories).
            pos_prefix/pos_prefixes: The column prefix(es) for the positive class.
            neg_prefix/neg_prefixes: The column prefix(es) for the negative class.
            excluded_metrics: A list of base metric names to exclude from this run.
        
        Returns:
            A dictionary with key performance metrics, or None if the experiment is skipped.
        """
        # Consolidate old and new argument names for backward compatibility
        pos = pos_prefixes or pos_prefix
        neg = neg_prefixes or neg_prefix
        if not pos or not neg:
            raise ValueError("Positive and negative prefixes must be provided.")

        LOG.info(f"\n{'='*80}\nStarting Experiment: {task_name}\n{'='*80}")
        
        # Ensure prefixes are lists for internal processing
        if isinstance(pos, str):
            pos = [pos]
        if isinstance(neg, str):
            neg = [neg]

        task_output_dir = self.output_dir / task_name
        
        # --- Check if results already exist ---
        if task_output_dir.exists():
            LOG.info(f"Results for experiment '{task_name}' already exist. Skipping.")
            return None
            
        task_output_dir.mkdir(exist_ok=True)

        # 1. Prepare Data
        X, y = self._prepare_data(pos, neg, excluded_metrics)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        LOG.info(f"Data prepared. Shape: {X.shape}. Class distribution: {y.value_counts(normalize=True).to_dict()}")
        
        # 2. Hyperparameter Tuning with Cross-Validation
        LOG.info("Performing hyperparameter tuning with GridSearchCV...")
        
        # Expanded grid for a more thorough search, balancing performance and accuracy.
        param_grid = {
            'n_estimators': [100, 250, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }

        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=xgb_model, param_grid=param_grid,
            scoring='roc_auc', cv=cv, verbose=0, n_jobs=-1 # Set verbose=0
        )

        # --- Suppress stdout/stderr during the noisy grid search fitting ---
        # Store original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Redirect to /dev/null (a black hole for output)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        try:
            grid_search.fit(X_train, y_train)
        finally:
            # --- Restore original streams ---
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        LOG.info(f"Best parameters found: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # 3. Final Evaluation on Test Set
        LOG.info("Evaluating final model on the hold-out test set...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        self._save_classification_report(y_test, y_pred, y_pred_proba, grid_search.best_params_, task_output_dir)
        self._plot_roc_curve(y_test, y_pred_proba, task_name, task_output_dir)
        self._plot_confusion_matrix(y_test, y_pred, task_name, task_output_dir)

        # 4. Explain Model with SHAP
        LOG.info("Calculating and plotting SHAP values for feature importance...")
        self._plot_shap_summary(best_model, X_train, task_name, task_output_dir)

        LOG.info(f"Experiment '{task_name}' complete. Outputs saved to {task_output_dir}")

        # --- Collect and return key metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        return {
            "Experimental Task": task_name,
            "AUC Score": roc_auc,
            "Accuracy": accuracy,
            "Macro F1-Score": macro_f1,
            "Excluded Metrics": ", ".join(sorted(excluded_metrics)) if excluded_metrics else "None"
        }

    def _save_classification_report(self, y_test, y_pred, y_pred_proba, best_params, output_dir):
        """Saves a detailed classification report, including best hyperparameters."""
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        params_str = "\n".join([f"  - {k}: {v}" for k, v in best_params.items()])

        report_str = f"""============================================================
                 CLASSIFICATION RESULTS
============================================================

Best Hyperparameters:
---------------------
{params_str}

Evaluation Metrics:
-------------------
  - AUC Score: {roc_auc:.4f}
  - Accuracy:  {accuracy:.4f}

Classification Report:
----------------------
{report}
============================================================
"""
        LOG.info(f"Classification Metrics:\n{report_str}")
        
        report_path = output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            # Use strip to remove leading/trailing whitespace from the multiline string
            f.write(report_str.strip())

    def _plot_roc_curve(self, y_test, y_pred_proba, task_name, output_dir):
        """Plots and saves the ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {task_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        sns.despine()
        
        plot_path = output_dir / "roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, y_test, y_pred, task_name, output_dir):
        """Plots and saves the confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {task_name}')
        
        plot_path = output_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_shap_summary(self, model, X_train, task_name, output_dir):
        """Calculates and plots SHAP summary and bar plots."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Summary Plot (violin style)
        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="violin", show=False)
        plt.title(f'SHAP Feature Importance - {task_name}', y=1.05)
        summary_plot_path = output_dir / "shap_summary_plot.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar Plot (mean absolute SHAP value)
        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.title(f'Mean SHAP Value Magnitude - {task_name}', y=1.05)
        bar_plot_path = output_dir / "shap_bar_plot.png"
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Run publication-quality classification analysis on the ALOFT dataset."
    )
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        # Run from the repository root so this default resolves. See the project README.
        default=pathlib.Path("data/outputs/master_metrics/ALOFT_master_metrics.csv"),
        help="Path to the master metrics CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        # Run from the repository root so this default resolves. See the project README.
        default=pathlib.Path("data/outputs/analysis/publication_analysis"),
        help="Directory to save analysis results."
    )
    args = parser.parse_args()

    # --- Pre-run Check ---
    if not args.data_path.is_file():
        LOG.error(f"FATAL: The specified data file does not exist: {args.data_path}")
        LOG.error("Please ensure the file path is correct and the file is accessible.")
        # Exit gracefully with a non-zero code to indicate failure.
        exit(1)

    analysis = PublicationAnalysis(args.data_path, args.output_dir)

    # List to hold results from all experiments
    all_results = []

    # --- Define Experiment Sets ---
    # We define permanently excluded metrics here.
    permanently_excluded_metrics = ["pmi"]
    LOG.info(f"Permanently excluding metrics due to methodological concerns: {permanently_excluded_metrics}")

    experiments = [
        {
            "task_name": "Quote_vs_MatchedSnippet",
            "pos_prefixes": ["sample"],
            "neg_prefixes": ["snippet"]
        },
        {
            "task_name": "Popular_vs_SampleQuote",
            "pos_prefixes": ["popular"],
            "neg_prefixes": ["sample"]
        },
        {
            "task_name": "Quote_vs_NonLiterary",
            "pos_prefixes": ["sample"],
            "neg_prefixes": ["nonlit"]
        },
        {
            "task_name": "T50Quote_vs_T50MatchedContext",
            "pos_prefixes": ["t50"],
            "neg_prefixes": ["t50freelength"]
        },
        {
            "task_name": "SampleQuote_vs_T50Quote",
            "pos_prefixes": ["sample"],
            "neg_prefixes": ["t50"]
        }
    ]

    # --- Run Original Experiments (Full Feature Set) ---
    LOG.info("\n" + "="*40 + " RUNNING ORIGINAL EXPERIMENTS (ALL FEATURES) " + "="*40)
    for experiment in tqdm(experiments, desc="Running Full-Feature Experiments"):
        exp_config = experiment.copy()
        # Append a suffix to distinguish from runs that included PMI and SWD
        exp_config['task_name'] = f"{exp_config['task_name']}_NoPMI_NoSWD"
        exp_config['excluded_metrics'] = permanently_excluded_metrics
        results = analysis.run_experiment(**exp_config)
        if results:
            all_results.append(results)

    # --- Run New Experiments (Excluding Sentiment Features) ---
    LOG.info("\n" + "="*40 + " RUNNING EXPERIMENTS AGAIN (NO SENTIMENT FEATURES) " + "="*40)
    sentiment_metrics = ['sentiment_polarity', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'sentiment_label']
    sentiment_excluded_metrics = list(set(permanently_excluded_metrics + sentiment_metrics))

    for experiment in tqdm(experiments, desc="Running No-Sentiment Experiments"):
        exp_config = experiment.copy()
        exp_config['task_name'] = f"{exp_config['task_name']}_NoPMI_NoSWD_NoSentiment"
        exp_config['excluded_metrics'] = sentiment_excluded_metrics
        results = analysis.run_experiment(**exp_config)
        if results:
            all_results.append(results)

    # --- Run New Experiments (Excluding Length-Dependent Features) ---
    LOG.info("\n" + "="*40 + " RUNNING NEW EXPERIMENTS (NO LENGTH REQUIREMENTS) " + "="*40)
    length_metrics_to_exclude = ["swd", "mtld"]
    LOG.info(f"Globally excluding metrics that require minimum text length: {length_metrics_to_exclude}")
    
    # Combine permanent and length-based exclusions
    final_excluded_metrics = permanently_excluded_metrics + length_metrics_to_exclude

    for experiment in tqdm(experiments, desc="Running No-Length-Reqs Experiments"):
        # Create a new dictionary for this run to avoid modifying the original
        exp_config = experiment.copy()
        # Append a suffix for a unique output directory
        exp_config['task_name'] = f"{exp_config['task_name']}_NoLengthReqs_NoPMI_NoSWD"
        
        results = analysis.run_experiment(
            excluded_metrics=final_excluded_metrics,
            **exp_config
        )
        if results:
            all_results.append(results)
    
    # --- Run New "Literary vs Non-Literary" Experiment ---
    LOG.info("\n" + "="*40 + " RUNNING LITERARY VS NON-LITERARY EXPERIMENT (NO SENTIMENT) " + "="*40)
    
    literary_prefixes = ['sample', 'popular', 't50']
    non_literary_prefixes = ['t50freelength', 'snippet', 'nonlit']
    sentiment_metrics = ['sentiment_polarity', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'sentiment_label']
    
    # Combine permanent (from user's edit) and sentiment exclusions
    no_sentiment_exclusions = list(set(permanently_excluded_metrics + sentiment_metrics))

    lit_vs_nonlit_config = {
        "task_name": "Literary_vs_NonLiterary_NoSentiment",
        "pos_prefixes": literary_prefixes,
        "neg_prefixes": non_literary_prefixes,
        "excluded_metrics": no_sentiment_exclusions
    }
    
    new_results = analysis.run_experiment(**lit_vs_nonlit_config)
    if new_results:
        all_results.append(new_results)

    # --- Aggregate and Save All Results ---
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Sort by AUC score for better readability
        summary_df = summary_df.sort_values(by="AUC Score", ascending=False).reset_index(drop=True)
        
        summary_path = analysis.output_dir / "all_experiments_summary.csv"
        
        # Create a copy for saving with formatted strings, keep original for printing with full precision
        summary_to_save = summary_df.copy()
        summary_to_save["AUC Score"] = summary_to_save["AUC Score"].map('{:.4f}'.format)
        summary_to_save["Accuracy"] = summary_to_save["Accuracy"].map('{:.2%}'.format)
        summary_to_save["Macro F1-Score"] = summary_to_save["Macro F1-Score"].map('{:.4f}'.format)
        
        summary_to_save.to_csv(summary_path, index=False)
        LOG.info(f"\nSaved aggregated results for all experiments to {summary_path}")
        
        # Also print the full-precision summary to the console
        LOG.info("\n" + "="*80)
        LOG.info("--- AGGREGATE RESULTS SUMMARY ---")
        LOG.info("="*80 + "\n")
        LOG.info(summary_df.to_string(index=False))
        LOG.info("\n" + "="*80)
    else:
        LOG.info("\nNo new experiments were run, so no summary file was generated.")


if __name__ == "__main__":
    main() 