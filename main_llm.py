"""
Data Explorer & Cleaner GUI
A comprehensive desktop tool for data import, cleaning, exploration, and 
reproducible script generation, featuring single-level Undo and a Visualization tab.

Includes fixes for NameError in script generation and comprehensive evaluation metrics.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
import textwrap
import threading

# Try importing Groq for AI column descriptions (free tier available at console.groq.com)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try importing Google Gemini (free tier available at aistudio.google.com)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try importing Cohere (free tier available at dashboard.cohere.com)
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# Try importing plotting libraries for interactive visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    # Use dummy objects if plotting libraries aren't installed to prevent crashes
    plt = None
    sns = None
    FigureCanvasTkAgg = None
    Figure = None


# ==============================================================================
# 1. Data Analysis and Utility Class (Simulating data_utils.py)
# ==============================================================================

class DataAnalyzer:
    """Encapsulates data analysis and transformation logic."""

    @staticmethod
    def _iqr_outliers(series):
        """Computes IQR-based outliers for a single numeric series."""
        col = series.dropna()
        if len(col) < 4:
            return 0, 0, 0, 0, 0
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = col[(col < lower) | (col > upper)]
        pct = len(outliers) / len(col) * 100 if len(col) > 0 else 0
        return len(outliers), pct, lower, upper, iqr

    @classmethod
    def generate_quality_report(cls, df):
        """Generates the comprehensive human-readable data quality report."""
        if df is None or df.empty:
            return "No data loaded or data is empty.", None

        out = io.StringIO()
        report_data = [] 

        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("DATA QUALITY ANALYSIS", file=out)
        print("="*80, file=out)
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}", file=out)
        print("="*80, file=out)
        
        # Empty columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        print(f"\nEmpty columns (100% NA): {len(empty_cols)}", file=out)
        if empty_cols:
            print(f"List: {', '.join(empty_cols)}", file=out)
        
        # Empty rows
        empty_rows = df.isna().all(axis=1).sum()
        pct_empty_rows = empty_rows / len(df) * 100
        print(f"\nEmpty rows (all NaN): {empty_rows} ({pct_empty_rows:.2f}%)", file=out)

        # Duplicate rows
        duplicate_rows = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicate_rows} ({duplicate_rows / len(df) * 100:.2f}%)", file=out)
        if duplicate_rows > 0:
            sample = df[df.duplicated(keep='first')].head(5)
            print("\nSample of 5 duplicate rows (indices may vary):", file=out)
            print(sample.to_string(), file=out)
        
        # Missing per column
        print('\n' + "="*80, file=out)
        print('Missing values per column (Top 20 by percentage):', file=out)
        na_counts = df.isna().sum()
        na_pct = (na_counts / len(df) * 100).sort_values(ascending=False)
        
        print(f"{'Column':<30} | {'Missing Count':>15} | {'Missing %':>10}", file=out)
        print("-" * 59, file=out)
        
        for col, count in na_counts.items():
            pct = na_pct[col]
            if pct > 0 and na_pct.index.get_loc(col) < 20: 
                 print(f"{col:<30} | {count:>15} | {pct:>10.2f}%", file=out)
            
            report_data.append({'column_name': col, 'dtype': str(df[col].dtype),
                                'missing_count': count, 'missing_pct': pct})

        # Boolean-like columns
        bool_cols = [c for c in df.columns if df[c].dropna().nunique() <= 2 and df[c].dropna().isin([True, False, 0, 1]).all()]
        print('\n' + "="*80, file=out)
        print('Boolean-like columns:', file=out)
        print(textwrap.fill(', '.join(bool_cols) or 'None', width=80), file=out)

        # Text columns summary
        obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print('\n' + "="*80, file=out)
        print('Text columns (Unique counts top 20):', file=out)
        
        for c in obj_cols[:20]:
            unique = df[c].nunique(dropna=False)
            pct = unique / len(df) * 100
            print(f"{c:<30}: Unique={unique} ({pct:.2f}%)", file=out)
            for item in report_data:
                if item['column_name'] == c:
                    item['unique_count'] = unique

        # Numeric columns descriptive statistics
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        print('\n' + "="*80, file=out)
        print('Numeric columns descriptive stats:', file=out)
        if num_cols:
            print(df[num_cols].describe().transpose().to_string(), file=out)
        else:
            print('None', file=out)

        # Outlier detection (IQR rule)
        print('\n' + "="*80, file=out)
        print('Outlier Detection (IQR method):', file=out)
        for c in num_cols:
            outlier_count, outlier_pct, lower, upper, _ = cls._iqr_outliers(df[c])
            
            for item in report_data:
                if item['column_name'] == c:
                    item['outlier_count'] = outlier_count
                    item['outlier_pct'] = outlier_pct
            
            if outlier_count > 0:
                print(f"{c:<30}: Outliers={outlier_count} ({outlier_pct:.2f}%) | Bounds=[{lower:.2f},{upper:.2f}]", file=out)
            
        # Footer
        print('\n' + "="*80, file=out)
        print(f"Analysis Timestamp: {timestamp}", file=out)
        print("Note: Outliers computed using IQR method: lower = Q1 − 1.5×IQR, upper = Q3 + 1.5×IQR", file=out)

        csv_df = pd.DataFrame(report_data).fillna('')
        return out.getvalue(), csv_df
    
# ==============================================================================
# 2. Main GUI Class
# ==============================================================================

class DataExplorerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Explorer & Cleaner")
        self.df = None
        self.undo_stack = [] # Stack for single-level undo
        self.filepath = None
        self.operations_log = []
        
        # State variables for Import/XLSX options
        self.skip_rows = tk.IntVar(value=0)
        self.remove_bottom = tk.IntVar(value=0)
        self.ai_provider = tk.StringVar(value="Groq")
        self.groq_api_key = tk.StringVar(value="")
        self.groq_model = tk.StringVar(value="llama-3.3-70b-versatile")
        self.gemini_api_key = tk.StringVar(value="")
        self.gemini_model = tk.StringVar(value="gemini-1.5-flash")
        self.cohere_api_key = tk.StringVar(value="")
        self.cohere_model = tk.StringVar(value="command-r-plus")
        
        # State variables for Script Generator
        self.opt_standardize_cols = tk.BooleanVar(value=True)
        self.opt_trim_whitespace = tk.BooleanVar(value=True)
        self.opt_delete_empty_cols = tk.BooleanVar(value=False)
        self.opt_delete_empty_rows = tk.BooleanVar(value=False)
        self.opt_delete_duplicates = tk.BooleanVar(value=True)
        self.opt_handle_missing = tk.StringVar(value='drop') 
        self.opt_target_column = tk.StringVar(value='target')

        self.opt_eda_plots = tk.BooleanVar(value=True)
        self.opt_classification = tk.BooleanVar(value=False)
        self.opt_regression = tk.BooleanVar(value=False)
        self.opt_clustering = tk.BooleanVar(value=False)
        self.opt_automl_autogluon = tk.BooleanVar(value=False)
        self.opt_automl_flaml = tk.BooleanVar(value=False)

        self._build_ui()
        self._show_splash()

    def _show_splash(self):
        """Shows a splash screen on startup crediting Claude."""
        splash = tk.Toplevel(self.root)
        splash.title("Welcome")
        splash.resizable(False, False)
        splash.grab_set()

        # Center the splash
        splash.update_idletasks()
        w, h = 420, 320
        x = (splash.winfo_screenwidth() // 2) - (w // 2)
        y = (splash.winfo_screenheight() // 2) - (h // 2)
        splash.geometry(f"{w}x{h}+{x}+{y}")

        frame = ttk.Frame(splash, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="🔬 Data Explorer & Cleaner",
                  font=('TkDefaultFont', 16, 'bold')).pack(pady=(10, 2))
        ttk.Label(frame, text="Your all-in-one data analysis and cleaning tool",
                  font=('TkDefaultFont', 9), foreground='gray').pack(pady=(0, 20))

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(frame, text="✨ Built with Claude",
                  font=('TkDefaultFont', 12, 'bold')).pack(pady=(10, 4))
        ttk.Label(frame,
                  text="This application was designed and developed\nwith the assistance of Claude,\nAnthropic's AI assistant.",
                  font=('TkDefaultFont', 10), justify=tk.CENTER).pack(pady=4)
        ttk.Label(frame, text="claude.ai  •  anthropic.com",
                  font=('TkDefaultFont', 9), foreground='gray').pack(pady=(2, 10))

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Button(frame, text="Get Started →", command=splash.destroy).pack(pady=(10, 0))

    def _show_about(self):
        """Shows the About dialog crediting Claude."""
        about = tk.Toplevel(self.root)
        about.title("About")
        about.resizable(False, False)
        about.grab_set()

        w, h = 380, 280
        x = (about.winfo_screenwidth() // 2) - (w // 2)
        y = (about.winfo_screenheight() // 2) - (h // 2)
        about.geometry(f"{w}x{h}+{x}+{y}")

        frame = ttk.Frame(about, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="🔬 Data Explorer & Cleaner",
                  font=('TkDefaultFont', 14, 'bold')).pack(pady=(5, 2))
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(frame, text="✨ Built with Claude",
                  font=('TkDefaultFont', 11, 'bold')).pack(pady=(5, 4))
        ttk.Label(frame,
                  text="This application was designed and developed\nwith the assistance of Claude,\nAnthropic's AI assistant.",
                  font=('TkDefaultFont', 10), justify=tk.CENTER).pack(pady=4)
        ttk.Label(frame, text="claude.ai  •  anthropic.com",
                  font=('TkDefaultFont', 9), foreground='gray').pack(pady=(4, 10))

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Close", command=about.destroy).pack(pady=(8, 0))

    def _log_operation(self, description):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.operations_log.append(f"[{timestamp}] {description}")

    def _save_snapshot(self):
        """Saves a snapshot of the current dataframe for undo."""
        if self.df is not None:
            if len(self.undo_stack) > 0:
                self.undo_stack.pop() 
            self.undo_stack.append(self.df.copy())

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8))

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Info Label and Notebook
        self.info_label = ttk.Label(right, text="No file loaded", anchor=tk.W)
        self.info_label.pack(fill=tk.X)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab Frames ---
        preview_frame = ttk.Frame(self.notebook); self.notebook.add(preview_frame, text="Data Preview")
        analysis_frame = ttk.Frame(self.notebook); self.notebook.add(analysis_frame, text="Analysis Results")
        
        vis_frame = ttk.Frame(self.notebook)
        self.notebook.add(vis_frame, text="Visualizations")
        self._build_visualization_tab(vis_frame)
        
        script_frame = ttk.Frame(self.notebook); self.notebook.add(script_frame, text="Python Script Generator")

        # --- Data Preview Content ---
        self.preview_text = tk.Text(preview_frame, wrap=tk.NONE)
        vsb_preview = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
        hsb_preview = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_text.xview)
        self.preview_text.config(yscrollcommand=vsb_preview.set, xscrollcommand=hsb_preview.set)
        vsb_preview.pack(side="right", fill="y")
        hsb_preview.pack(side="bottom", fill="x")
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        # --- Analysis Results Content ---
        self.analysis_text = tk.Text(analysis_frame, wrap=tk.NONE)
        vsb_analysis = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.analysis_text.yview)
        hsb_analysis = ttk.Scrollbar(analysis_frame, orient="horizontal", command=self.analysis_text.xview)
        self.analysis_text.config(yscrollcommand=vsb_analysis.set, xscrollcommand=hsb_analysis.set)
        vsb_analysis.pack(side="right", fill="y")
        hsb_analysis.pack(side="bottom", fill="x")
        self.analysis_text.pack(fill=tk.BOTH, expand=True)

        # --- Script Generator Content ---
        self._build_script_generator_tab(script_frame)

        # --- Left Column Controls (Toolbar) ---
        self._build_import_frame(left)
        self._build_exploration_frame(left)
        self._build_cleaning_frame(left)
        self._build_export_frame(left)

    def _build_import_frame(self, parent):
        imp_frame = ttk.LabelFrame(parent, text="Import")
        imp_frame.pack(fill=tk.X, pady=4)
        ttk.Button(imp_frame, text="Import CSV/TXT", command=self.import_csv).pack(fill=tk.X, pady=2)
        ttk.Button(imp_frame, text="Import XLSX", command=self.import_xlsx).pack(fill=tk.X, pady=2)

        # XLSX options
        opt_frame = ttk.Frame(imp_frame)
        opt_frame.pack(fill=tk.X)
        
        ttk.Label(opt_frame, text="Skip Top Rows:").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(opt_frame, from_=0, to=100, width=5, textvariable=self.skip_rows).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(opt_frame, text="Remove Bottom Rows:").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(opt_frame, from_=0, to=100, width=5, textvariable=self.remove_bottom).pack(side=tk.LEFT, padx=2)


    def _build_exploration_frame(self, parent):
        exp_frame = ttk.LabelFrame(parent, text="Exploration")
        exp_frame.pack(fill=tk.X, pady=4)
        ttk.Button(exp_frame, text="Top 5 Rows", command=self.show_head).pack(fill=tk.X, pady=2)
        ttk.Button(exp_frame, text="Bottom 5 Rows", command=self.show_tail).pack(fill=tk.X, pady=2)
        ttk.Button(exp_frame, text="Analyze Data Quality", command=self.analyze_data).pack(fill=tk.X, pady=2)
        ttk.Button(exp_frame, text="Export Quality Report", command=self.export_quality_report).pack(fill=tk.X, pady=2)
        ttk.Separator(exp_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(exp_frame, text="AI Column Descriptions:", font=('TkDefaultFont', 8, 'bold')).pack(anchor=tk.W, padx=2)

        # Provider selector
        prov_frame = ttk.Frame(exp_frame)
        prov_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prov_frame, text="Provider:").pack(side=tk.LEFT, padx=2)
        provider_cb = ttk.Combobox(prov_frame, textvariable=self.ai_provider, width=10,
                                   values=["Groq", "Gemini", "Cohere"], state="readonly")
        provider_cb.pack(side=tk.LEFT, padx=2)
        provider_cb.bind("<<ComboboxSelected>>", self._on_provider_change)

        # Container that swaps content based on provider
        self.ai_config_frame = ttk.Frame(exp_frame)
        self.ai_config_frame.pack(fill=tk.X)
        self._build_groq_config(self.ai_config_frame)   # default shown

        self.ai_describe_btn = ttk.Button(exp_frame, text="✨ Describe Columns with AI",
                                          command=self.describe_columns_with_ai)
        self.ai_describe_btn.pack(fill=tk.X, pady=2)

    def _clear_config_frame(self):
        for w in self.ai_config_frame.winfo_children():
            w.destroy()

    def _on_provider_change(self, event=None):
        self._clear_config_frame()
        p = self.ai_provider.get()
        if p == "Groq":
            self._build_groq_config(self.ai_config_frame)
        elif p == "Gemini":
            self._build_gemini_config(self.ai_config_frame)
        elif p == "Cohere":
            self._build_cohere_config(self.ai_config_frame)

    def _build_groq_config(self, parent):
        ttk.Label(parent, text="API Key (console.groq.com):", wraplength=180).pack(anchor=tk.W, padx=2)
        ttk.Entry(parent, textvariable=self.groq_api_key, show="*").pack(fill=tk.X, padx=2, pady=2)
        mf = ttk.Frame(parent)
        mf.pack(fill=tk.X, padx=2)
        ttk.Label(mf, text="Model:").pack(side=tk.LEFT)
        ttk.Combobox(mf, textvariable=self.groq_model, width=18,
                     values=["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                             "mixtral-8x7b-32768", "gemma2-9b-it"]
                     ).pack(side=tk.LEFT, padx=2)

    def _build_gemini_config(self, parent):
        ttk.Label(parent, text="API Key (aistudio.google.com):", wraplength=180).pack(anchor=tk.W, padx=2)
        ttk.Entry(parent, textvariable=self.gemini_api_key, show="*").pack(fill=tk.X, padx=2, pady=2)
        mf = ttk.Frame(parent)
        mf.pack(fill=tk.X, padx=2)
        ttk.Label(mf, text="Model:").pack(side=tk.LEFT)
        ttk.Combobox(mf, textvariable=self.gemini_model, width=18,
                     values=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
                     ).pack(side=tk.LEFT, padx=2)

    def _build_cohere_config(self, parent):
        ttk.Label(parent, text="API Key (dashboard.cohere.com):", wraplength=180).pack(anchor=tk.W, padx=2)
        ttk.Entry(parent, textvariable=self.cohere_api_key, show="*").pack(fill=tk.X, padx=2, pady=2)
        mf = ttk.Frame(parent)
        mf.pack(fill=tk.X, padx=2)
        ttk.Label(mf, text="Model:").pack(side=tk.LEFT)
        ttk.Combobox(mf, textvariable=self.cohere_model, width=18,
                     values=["command-r-plus", "command-r", "command"]
                     ).pack(side=tk.LEFT, padx=2)

    def _build_cleaning_frame(self, parent):
        clean_frame = ttk.LabelFrame(parent, text="Cleaning")
        clean_frame.pack(fill=tk.X, pady=4)
        ttk.Button(clean_frame, text="Delete Selected Columns", command=self.delete_selected_columns).pack(fill=tk.X, pady=2)
        ttk.Button(clean_frame, text="Delete Selected Rows", command=self.delete_selected_rows).pack(fill=tk.X, pady=2)
        ttk.Button(clean_frame, text="Delete Empty Columns", command=self.delete_empty_columns).pack(fill=tk.X, pady=2)
        ttk.Button(clean_frame, text="Delete Empty Rows", command=self.delete_empty_rows).pack(fill=tk.X, pady=2)
        ttk.Button(clean_frame, text="Delete N Rows from Top", command=self.delete_n_top).pack(fill=tk.X, pady=2)
        ttk.Button(clean_frame, text="Delete N Rows from Bottom", command=self.delete_n_bottom).pack(fill=tk.X, pady=2)

    def _build_export_frame(self, parent):
        exp2_frame = ttk.LabelFrame(parent, text="Export")
        exp2_frame.pack(fill=tk.X, pady=4)
        ttk.Button(exp2_frame, text="Export to CSV", command=self.export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(exp2_frame, text="Export to XLSX", command=self.export_xlsx).pack(fill=tk.X, pady=2)
        ttk.Button(exp2_frame, text="Undo", command=self.undo_action).pack(fill=tk.X, pady=2)
        ttk.Separator(exp2_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Button(exp2_frame, text="ℹ️ About", command=self._show_about).pack(fill=tk.X, pady=2)
        
    def _build_visualization_tab(self, parent):
        if FigureCanvasTkAgg is None:
            ttk.Label(parent, text="Matplotlib/TkAgg not installed. Cannot display interactive plots.").pack(pady=20)
            return

        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Generate Histograms", command=lambda: self._draw_plot('hist')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Correlation Heatmap", command=lambda: self._draw_plot('corr')).pack(side=tk.LEFT, padx=5)
        
        self.vis_container = ttk.Frame(parent)
        self.vis_container.pack(fill=tk.BOTH, expand=True)
        self.vis_canvas = None 

    def _draw_plot(self, plot_type):
        if self.df is None or plt is None:
            messagebox.showwarning("Warning", "Load data first or install Matplotlib/Seaborn.")
            return
            
        for widget in self.vis_container.winfo_children():
            widget.destroy()

        if plot_type == 'hist':
            num_cols = self.df.select_dtypes(include=np.number).columns
            if len(num_cols) == 0:
                messagebox.showinfo("Info", "No numeric columns for histograms.")
                return
                
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            self.df[num_cols].hist(ax=ax, bins=15) 
            ax.set_title(f"Histograms of Numeric Features")
            fig.tight_layout()
            
        elif plot_type == 'corr':
            num_data = self.df.select_dtypes(include=np.number)
            if num_data.shape[1] < 2:
                messagebox.showinfo("Info", "Need at least two numeric columns for correlation heatmap.")
                return

            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            sns.heatmap(num_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
        else:
            return

        self.vis_canvas = FigureCanvasTkAgg(fig, master=self.vis_container)
        self.vis_canvas_widget = self.vis_canvas.get_tk_widget()
        self.vis_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.vis_canvas.draw()
        self.notebook.select(self.notebook.tabs()[2])
        self._log_operation(f"Generated {plot_type} plot in Visualization tab.")

    def _build_script_generator_tab(self, parent):
        gen_main = ttk.Frame(parent)
        gen_main.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(gen_main, width=300)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        script_preview_frame = ttk.Frame(gen_main)
        script_preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(controls_frame)
        canvas.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(controls_frame, orient="vertical", command=canvas.yview)
        vsb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=vsb.set)

        controls_content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=controls_content, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        controls_content.bind("<Configure>", _on_frame_configure)
        
        # --- Script Generator Controls ---
        
        clean_opt_frame = ttk.LabelFrame(controls_content, text="Cleaning Options")
        clean_opt_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Checkbutton(clean_opt_frame, text="Standardize column names (snake_case)", variable=self.opt_standardize_cols).pack(anchor=tk.W)
        ttk.Checkbutton(clean_opt_frame, text="Trim whitespace for text columns", variable=self.opt_trim_whitespace).pack(anchor=tk.W)
        ttk.Checkbutton(clean_opt_frame, text="Delete 100% empty columns", variable=self.opt_delete_empty_cols).pack(anchor=tk.W)
        ttk.Checkbutton(clean_opt_frame, text="Delete 100% empty rows", variable=self.opt_delete_empty_rows).pack(anchor=tk.W)
        ttk.Checkbutton(clean_opt_frame, text="Delete duplicate rows", variable=self.opt_delete_duplicates).pack(anchor=tk.W)
        
        missing_frame = ttk.Frame(clean_opt_frame)
        missing_frame.pack(fill=tk.X)
        ttk.Label(missing_frame, text="Handle Missing Values:").pack(side=tk.LEFT)
        ttk.Radiobutton(missing_frame, text="Drop Rows", variable=self.opt_handle_missing, value='drop').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(missing_frame, text="Fill Mean/Unknown", variable=self.opt_handle_missing, value='fill').pack(side=tk.LEFT)
        
        eda_opt_frame = ttk.LabelFrame(controls_content, text="EDA Options")
        eda_opt_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Checkbutton(eda_opt_frame, text="Include basic EDA plots", variable=self.opt_eda_plots).pack(anchor=tk.W)
        
        model_opt_frame = ttk.LabelFrame(controls_content, text="Modeling Options")
        model_opt_frame.pack(fill=tk.X, pady=5, padx=5)
        
        target_frame = ttk.Frame(model_opt_frame)
        target_frame.pack(fill=tk.X)
        ttk.Label(target_frame, text="Target Column:").pack(side=tk.LEFT)
        ttk.Entry(target_frame, textvariable=self.opt_target_column, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(model_opt_frame, text="Classification (RandomForest) + Evaluation", variable=self.opt_classification).pack(anchor=tk.W)
        ttk.Checkbutton(model_opt_frame, text="Regression (LinearRegression)", variable=self.opt_regression).pack(anchor=tk.W)
        ttk.Checkbutton(model_opt_frame, text="Clustering (KMeans)", variable=self.opt_clustering).pack(anchor=tk.W)
        
        automl_opt_frame = ttk.LabelFrame(controls_content, text="AutoML Options")
        automl_opt_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Checkbutton(automl_opt_frame, text="AutoGluon (Robust Ensembling)", variable=self.opt_automl_autogluon).pack(anchor=tk.W)
        ttk.Checkbutton(automl_opt_frame, text="FLAML (Fast & Optimized XGBoost/LightGBM)", variable=self.opt_automl_flaml).pack(anchor=tk.W)
        
        # --- Script Preview and Buttons ---
        self.script_preview_text = tk.Text(script_preview_frame, wrap=tk.NONE)
        self.script_preview_text.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(script_preview_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Generate Full Python Script", command=self.generate_script).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(button_frame, text="Save Script to File (.py)", command=self.save_script).pack(side=tk.LEFT, expand=True, padx=5)


    # ---------------- Import/Data Functions ----------------
    def import_data(self, path, reader_func, **kwargs):
        try:
            df = reader_func(path, **kwargs)
            
            rb = self.remove_bottom.get()
            if rb > 0 and len(df) > rb:
                df = df.iloc[:-rb]
            elif rb > len(df):
                messagebox.showwarning('Warning', f'Cannot remove {rb} rows. Dataset only has {len(df)} rows.')
                return

            self._save_snapshot() 
            self.df = df
            self.filepath = path
            self.update_display()
            self._log_operation(f"Loaded file: {os.path.basename(path)} ({len(self.df)} rows, {len(self.df.columns)} columns)")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load data: {e}")

    def import_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/TXT","*.csv *.txt")])
        if not path: return
        self.import_data(path, pd.read_csv, low_memory=False, skiprows=self.skip_rows.get())

    def import_xlsx(self):
        path = filedialog.askopenfilename(filetypes=[("Excel","*.xlsx")])
        if not path: return
        self.import_data(path, pd.read_excel, skiprows=self.skip_rows.get())

    def update_display(self):
        if self.df is not None:
            info = f"Loaded {os.path.basename(self.filepath)} | {len(self.df)} rows, {len(self.df.columns)} columns"
            self.info_label.config(text=info)
            self.show_head()
        else:
             self.info_label.config(text="No file loaded")
             self.preview_text.delete(1.0, tk.END)

    def show_head(self):
        if self.df is not None:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, self.df.head().to_string())

    def show_tail(self):
        if self.df is not None:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, self.df.tail().to_string())

    # ---------------- Cleaning Functions ----------------
    def delete_selected_columns(self):
        if self.df is None: return
        win = tk.Toplevel(self.root)
        win.title("Select Columns to Delete")
        listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, width=50)
        for col in self.df.columns:
            listbox.insert(tk.END, col)
        listbox.pack(fill=tk.BOTH, expand=True)
        def confirm():
            selected = [listbox.get(i) for i in listbox.curselection()]
            if not selected:
                messagebox.showinfo('Info', 'No columns selected')
                win.destroy(); return
            
            if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {len(selected)} columns?"):
                self._save_snapshot()
                self.df.drop(columns=selected, inplace=True)
                self.update_display()
                self._log_operation(f"Deleted {len(selected)} columns: {', '.join(selected)}")
                messagebox.showinfo('Success', f'Deleted {len(selected)} columns')
            win.destroy()
        ttk.Button(win, text="Delete Selected", command=confirm).pack(fill=tk.X)

    def delete_selected_rows(self):
        if self.df is None: return
        win = tk.Toplevel(self.root)
        win.title("Select Rows to Delete")
        
        list_frame = ttk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(list_frame, orient="vertical")
        listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, width=100, height=20, yscrollcommand=vsb.set)
        vsb.config(command=listbox.yview)
        vsb.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        preview_limit = 200 
        for i, row in self.df.head(preview_limit).iterrows():
            preview = ' | '.join(map(str, row.values.tolist()[:3])) 
            listbox.insert(tk.END, f"{i}: {preview}")
            
        if len(self.df) > preview_limit:
            listbox.insert(tk.END, f"... (Showing first {preview_limit} of {len(self.df)} rows) ...")

        def confirm():
            indices = [int(listbox.get(i).split(':')[0]) for i in listbox.curselection()]
            if not indices:
                messagebox.showinfo('Info', 'No rows selected'); win.destroy(); return
            
            if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {len(indices)} rows?"):
                self._save_snapshot()
                self.df.drop(index=indices, inplace=True)
                self.update_display()
                self._log_operation(f"Deleted {len(indices)} selected rows.")
                messagebox.showinfo('Success', f'Deleted {len(indices)} rows')
            win.destroy()
        ttk.Button(win, text="Delete Selected", command=confirm).pack(fill=tk.X)

    def delete_empty_columns(self):
        if self.df is None: return
        cols = [c for c in self.df.columns if self.df[c].isna().all()]
        if not cols:
            messagebox.showinfo('Info', 'No empty columns found'); return
        
        if messagebox.askyesno("Confirm Deletion", f"Delete {len(cols)} columns that are 100% missing?"):
            self._save_snapshot()
            self.df.drop(columns=cols, inplace=True)
            self.update_display()
            self._log_operation(f"Deleted {len(cols)} columns (100% NA).")
            messagebox.showinfo('Success', f'Deleted {len(cols)} empty columns')

    def delete_empty_rows(self):
        if self.df is None: return
        before = len(self.df)
        rows_to_drop = self.df.isna().all(axis=1)
        removed_count = rows_to_drop.sum()
        
        if removed_count == 0:
            messagebox.showinfo('Info', 'No empty rows found'); return

        if messagebox.askyesno("Confirm Deletion", f"Delete {removed_count} rows that are 100% missing?"):
            self._save_snapshot()
            self.df.dropna(how='all', inplace=True)
            self.update_display()
            self._log_operation(f"Deleted {removed_count} rows (100% NA).")
            messagebox.showinfo('Success', f'Deleted {removed_count} empty rows')

    def delete_n_top(self):
        if self.df is None: return
        n = simpledialog.askinteger('Delete Top Rows', 'Enter number of top rows to delete:', minvalue=1, maxvalue=len(self.df)-1)
        if n and n > 0:
            if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the top {n} rows?"):
                self._save_snapshot()
                self.df = self.df.iloc[n:]
                self.update_display()
                self._log_operation(f"Deleted top {n} rows.")
                messagebox.showinfo('Success', f'Deleted top {n} rows')

    def delete_n_bottom(self):
        if self.df is None: return
        n = simpledialog.askinteger('Delete Bottom Rows', 'Enter number of bottom rows to delete:', minvalue=1, maxvalue=len(self.df)-1)
        if n and n > 0:
            if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the bottom {n} rows?"):
                self._save_snapshot()
                self.df = self.df.iloc[:-n]
                self.update_display()
                self._log_operation(f"Deleted bottom {n} rows.")
                messagebox.showinfo('Success', f'Deleted bottom {n} rows')
                
    def undo_action(self):
        if not self.undo_stack:
            messagebox.showinfo("Undo", "No action to undo.")
            return

        self.df = self.undo_stack.pop() 
        self.update_display()
        self._log_operation("Performed Undo operation.")
        messagebox.showinfo("Undo", "Last operation reversed.")

    # ---------------- Exploration/Analysis Functions ----------------
    def analyze_data(self):
        if self.df is None: return
        self.analysis_text.delete(1.0, tk.END)
        report_text, report_csv_df = DataAnalyzer.generate_quality_report(self.df)
        
        self.analysis_text.insert(tk.END, report_text)
        self.last_quality_report_text = report_text
        self.last_quality_report_csv = report_csv_df
        self.notebook.select(self.notebook.tabs()[1])
        self._log_operation("Generated Data Quality Analysis.")

    def describe_columns_with_ai(self):
        """Routes to the selected AI provider to describe each column."""
        if self.df is None:
            messagebox.showinfo("Info", "Please load a dataset first.")
            return

        provider = self.ai_provider.get()

        # Build column summary prompt (shared across all providers)
        col_summaries = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing_pct = self.df[col].isna().mean() * 100
            unique = self.df[col].nunique(dropna=True)
            if self.df[col].dtype == object or str(self.df[col].dtype) == 'category':
                samples = self.df[col].dropna().unique()[:5].tolist()
                stats = f"unique={unique}, sample values={samples}"
            else:
                desc = self.df[col].describe()
                stats = (f"min={desc.get('min', 0):.2f}, max={desc.get('max', 0):.2f}, "
                         f"mean={desc.get('mean', 0):.2f}, unique={unique}")
            col_summaries.append(f"- {col} (dtype={dtype}, missing={missing_pct:.1f}%): {stats}")

        prompt = (
            f"I have a dataset with {len(self.df)} rows and {len(self.df.columns)} columns.\n\n"
            f"Column summaries:\n" + "\n".join(col_summaries) + "\n\n"
            "For each column, write a clear 1-2 sentence description of what it likely represents, "
            "its data type, and any notable characteristics (e.g. high missingness, likely ID column, etc.). "
            "Format as a list with the column name followed by the description."
        )

        self.ai_describe_btn.config(state=tk.DISABLED, text="⏳ Describing columns...")
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, f"Calling {provider}... please wait.\n")
        self.notebook.select(self.notebook.tabs()[1])

        if provider == "Groq":
            threading.Thread(target=self._run_groq, args=(prompt,), daemon=True).start()
        elif provider == "Gemini":
            threading.Thread(target=self._run_gemini, args=(prompt,), daemon=True).start()
        elif provider == "Cohere":
            threading.Thread(target=self._run_cohere, args=(prompt,), daemon=True).start()

    def _run_groq(self, prompt):
        if not GROQ_AVAILABLE:
            self.root.after(0, lambda: self._display_ai_description(
                "Groq package not installed.\n\nRun:  pip install groq"))
            return
        api_key = self.groq_api_key.get().strip()
        if not api_key:
            self.root.after(0, lambda: self._display_ai_description(
                "No Groq API key entered.\nGet one free at: https://console.groq.com"))
            return
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=self.groq_model.get().strip(),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            result = response.choices[0].message.content
            self.root.after(0, lambda: self._display_ai_description(result))
        except Exception as e:
            err = str(e)
            msg = ("Invalid API key." if "401" in err or "invalid_api_key" in err.lower()
                   else "Rate limit reached — wait a moment." if "429" in err
                   else f"Groq error:\n{err}")
            self.root.after(0, lambda: self._display_ai_description(msg))

    def _run_gemini(self, prompt):
        if not GEMINI_AVAILABLE:
            self.root.after(0, lambda: self._display_ai_description(
                "Gemini package not installed.\n\nRun:  pip install google-generativeai"))
            return
        api_key = self.gemini_api_key.get().strip()
        if not api_key:
            self.root.after(0, lambda: self._display_ai_description(
                "No Gemini API key entered.\nGet one free at: https://aistudio.google.com"))
            return
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.gemini_model.get().strip())
            response = model.generate_content(prompt)
            self.root.after(0, lambda: self._display_ai_description(response.text))
        except Exception as e:
            err = str(e)
            msg = ("Invalid API key." if "401" in err or "api_key" in err.lower()
                   else "Rate limit reached — wait a moment." if "429" in err
                   else f"Gemini error:\n{err}")
            self.root.after(0, lambda: self._display_ai_description(msg))

    def _run_cohere(self, prompt):
        if not COHERE_AVAILABLE:
            self.root.after(0, lambda: self._display_ai_description(
                "Cohere package not installed.\n\nRun:  pip install cohere"))
            return
        api_key = self.cohere_api_key.get().strip()
        if not api_key:
            self.root.after(0, lambda: self._display_ai_description(
                "No Cohere API key entered.\nGet one free at: https://dashboard.cohere.com"))
            return
        try:
            client = cohere.ClientV2(api_key=api_key)
            response = client.chat(
                model=self.cohere_model.get().strip(),
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.message.content[0].text
            self.root.after(0, lambda: self._display_ai_description(result))
        except Exception as e:
            err = str(e)
            msg = ("Invalid API key." if "401" in err or "unauthorized" in err.lower()
                   else "Rate limit reached — wait a moment." if "429" in err
                   else f"Cohere error:\n{err}")
            self.root.after(0, lambda: self._display_ai_description(msg))

    def _display_ai_description(self, text):
        """Displays AI description result and re-enables the button."""
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "AI COLUMN DESCRIPTIONS\n")
        self.analysis_text.insert(tk.END, "=" * 80 + "\n\n")
        self.analysis_text.insert(tk.END, text)
        self.ai_describe_btn.config(state=tk.NORMAL, text="✨ Describe Columns with AI")
        self._log_operation("Generated AI column descriptions.")

    def export_quality_report(self):
        if not hasattr(self, 'last_quality_report_text'):
            messagebox.showinfo('Info','Run a Data Quality analysis first')
            return
        
        report_type = simpledialog.askstring("Export Report", "Export format (TXT or CSV)?", initialvalue="TXT")
        if not report_type: return

        if report_type.upper() == 'TXT':
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text','*.txt')])
            if not path: return
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.last_quality_report_text)
            self._log_operation(f"Exported quality report as TXT to {path}.")
            messagebox.showinfo('Success', f'Exported text report to {path}')
        
        elif report_type.upper() == 'CSV':
            path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
            if not path: return
            self.last_quality_report_csv.to_csv(path, index=False)
            self._log_operation(f"Exported quality report as CSV to {path}.")
            messagebox.showinfo('Success', f'Exported CSV report to {path}')
        
        else:
            messagebox.showwarning('Invalid Format', 'Please enter TXT or CSV.')

    # ---------------- Export Functions ----------------
    def export_csv(self):
        if self.df is None: return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if path: 
            self.df.to_csv(path, index=False)
            self._log_operation(f"Exported current DF to CSV: {path}.")

    def export_xlsx(self):
        if self.df is None: return
        path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel','*.xlsx')])
        if path: 
            self.df.to_excel(path, index=False)
            self._log_operation(f"Exported current DF to XLSX: {path}.")

    # ---------------- Script Generator Functions (Corrected and Optimized) ----------------
    def generate_script(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        script = []
        is_automl_selected = self.opt_automl_autogluon.get() or self.opt_automl_flaml.get()

        # --- Operations Log Header ---
        script.append("# Generated by Data Explorer & Cleaner GUI")
        script.append(f"# Generation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script.append("# Operations Log from GUI Session:")
        for log in self.operations_log:
            script.append(f"# {log}")
        script.append("\n" + "="*80 + "\n")

        # --- 1. Import Libraries ---
        script.append("# --- 1. Import Libraries ---")
        script.append("import pandas as pd")
        script.append("import numpy as np")
        if self.opt_eda_plots.get() or self.opt_classification.get() or self.opt_regression.get():
             script.append("import matplotlib.pyplot as plt")
             script.append("import seaborn as sns")
        
        # General modeling utilities
        if self.opt_classification.get() or self.opt_regression.get() or self.opt_clustering.get() or is_automl_selected:
             script.append("from sklearn.model_selection import train_test_split")

        if self.opt_classification.get():
            script.append("from sklearn.ensemble import RandomForestClassifier")
            script.append("from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score")
        if self.opt_regression.get():
            script.append("from sklearn.linear_model import LinearRegression")
            script.append("from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error")
        if self.opt_clustering.get():
            script.append("from sklearn.cluster import KMeans")
            script.append("from sklearn.metrics import silhouette_score")
            
        if self.opt_automl_autogluon.get():
            script.append("# For AutoGluon: pip install autogluon.tabular")
            script.append("from autogluon.tabular import TabularPredictor")
            script.append("from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score") # Needed for explicit AG evaluation
        if self.opt_automl_flaml.get():
            script.append("# For FLAML: pip install flaml")
            script.append("from flaml import AutoML")
        
        # --- 2. Data Loading ---
        script.append("\n# --- 2. Data Loading ---")
        
        file_ext = os.path.splitext(self.filepath)[1].lower()
        if file_ext in ['.csv', '.txt']:
            reader_code = f"df = pd.read_csv(r'{self.filepath}', low_memory=False)"
        elif file_ext in ['.xlsx', '.xls']:
            skip_rows_val = self.skip_rows.get()
            reader_code = f"df = pd.read_excel(r'{self.filepath}', skiprows={skip_rows_val})"
        else:
            reader_code = f"# WARNING: Could not determine reader function for {file_ext}. Using pd.read_csv by default."
            reader_code += f"\ndf = pd.read_csv(r'{self.filepath}', low_memory=False)"

        script.append(reader_code)
        
        if self.remove_bottom.get() > 0:
            script.append(f"# Remove bottom {self.remove_bottom.get()} rows (as configured in GUI)")
            script.append(f"df = df.iloc[:-{self.remove_bottom.get()}]")
            
        script.append("print(f\"Initial data shape: {df.shape}\")")
        script.append("print(df.info())")
        
        # --- 3. Initial Data Cleaning & Types Verification ---
        script.append("\n# --- 3. Initial Data Cleaning & Types Verification ---")
        script.append("\n# Convert columns to best possible dtypes (e.g., int64 to Int64)")
        script.append("df = df.convert_dtypes()")
        script.append("print(\"\\n--- Dtypes after conversion ---\")")
        script.append("print(df.dtypes)")
        
        if self.opt_standardize_cols.get():
            script.append("\n# Standardize column names (from GUI setting)")
            script.append("def standardize_cols(df):")
            script.append("    cols = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True).str.strip('_')")
            script.append("    df.columns = cols")
            script.append("    return df")
            script.append("df = standardize_cols(df)")
            
        if self.opt_trim_whitespace.get():
            script.append("\n# Trim whitespace for string columns (from GUI setting)")
            script.append("for col in df.select_dtypes(include='object').columns:")
            script.append("    df[col] = df[col].str.strip()")
        
        if self.opt_delete_empty_cols.get():
            script.append("\n# Delete 100% empty columns (from GUI setting)")
            script.append("cols_before = len(df.columns)")
            script.append("df = df.loc[:, ~df.isna().all()]")
            script.append("print(f\"Removed {cols_before - len(df.columns)} empty columns.\")")

        if self.opt_delete_empty_rows.get():
            script.append("\n# Delete 100% empty rows (from GUI setting)")
            script.append("rows_before = len(df)")
            script.append("df.dropna(how='all', inplace=True)")
            script.append("print(f\"Removed {rows_before - len(df)} empty rows.\")")
            
        if self.opt_delete_duplicates.get():
            script.append("\n# Delete duplicate rows (from GUI setting)")
            script.append("rows_before = len(df)")
            script.append("df.drop_duplicates(inplace=True)")
            script.append("print(f\"Removed {rows_before - len(df)} duplicate rows.\")")

        # Handle Missing Values & Encoding - SKIP if AutoGluon is selected
        if self.opt_automl_autogluon.get():
            script.append("\n# Skipping manual Missing Value Handling and Encoding: AutoGluon performs these steps automatically.")
        else:
            script.append("\n# Handle Missing Values (Strategy: " + self.opt_handle_missing.get() + ")")
            if self.opt_handle_missing.get() == 'drop':
                script.append("df.dropna(axis=0, inplace=True, how='any') # Dropping rows with any NaN")
            elif self.opt_handle_missing.get() == 'fill':
                script.append("# Numeric imputation: fill with mean/median")
                script.append("for col in df.select_dtypes(include=np.number).columns:")
                script.append("    df[col].fillna(df[col].mean(), inplace=True)")
                script.append("# Text imputation: fill with 'Unknown'")
                script.append("for col in df.select_dtypes(include=['object', 'category']).columns:")
                script.append("    df[col].fillna('Unknown', inplace=True)")
            
            script.append("\n# Outlier Handling (IQR-based capping - Commented out by default)")
            script.append("# Note: This should be done carefully to avoid data contamination.")
            script.append("# for col in df.select_dtypes(include=np.number).columns:")
            script.append("#     Q1 = df[col].quantile(0.25)")
            script.append("#     Q3 = df[col].quantile(0.75)")
            script.append("#     IQR = Q3 - Q1")
            script.append("#     Lower_Bound = Q1 - 1.5 * IQR")
            script.append("#     Upper_Bound = Q3 + 1.5 * IQR")
            script.append("#     df[col] = np.where(df[col] < Lower_Bound, Lower_Bound, df[col])")
            script.append("#     df[col] = np.where(df[col] > Upper_Bound, Upper_Bound, df[col])")
            
            # Data Split and Encoding for SKLearn/FLAML
            target = self.opt_target_column.get()
            script.append("\n# Manual Encoding for standard SKLearn/FLAML models")
            script.append(f"if '{target}' in df.columns:")
            script.append(f"    X = df.drop(columns='{target}')")
            script.append(f"    y = df['{target}']")
            script.append("    X = pd.get_dummies(X, drop_first=True)")
            script.append("    X = X.select_dtypes(include=np.number)")
            script.append("    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)")
            script.append("    print(f\"\\nTrain/Test Split Shapes: {X_train.shape} / {X_test.shape}\")")
            script.append("else:")
            script.append(f"    print(f\"Skipping train/test split: Target column '{target}' not found.\")")
            
        # --- 4. Exploratory Data Analysis (EDA) and Visuals ---
        script.append("\n# --- 4. Exploratory Data Analysis (EDA) and Visuals ---")

        if self.opt_eda_plots.get():
            script.append("\n# Basic EDA Plots (from GUI setting)")
            script.append("sns.set_style('whitegrid')")
            script.append("plt.figure(figsize=(10, 6))")
            
            script.append("df.select_dtypes(include=np.number).hist(figsize=(12, 12))")
            script.append("plt.suptitle('Histograms of Numeric Features')")
            script.append("plt.show()")
            
            script.append("plt.figure(figsize=(10, 8))")
            script.append("sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt='.2f', cmap='coolwarm')")
            script.append("plt.title('Correlation Heatmap')")
            script.append("plt.show()")

        
        # --- 5. Modeling Examples and Evaluation ---
        script.append("\n# --- 5. Modeling Examples and Evaluation ---")
        
        target = self.opt_target_column.get()
        
        if self.opt_classification.get() or self.opt_regression.get():
            script.append(f"\nif '{target}' in df.columns and not {is_automl_selected}:")

            if self.opt_classification.get():
                script.append("\n    # Classification Example (RandomForestClassifier - from GUI setting)")
                script.append("    try:")
                script.append("        clf = RandomForestClassifier(n_estimators=100, random_state=42)")
                script.append("        clf.fit(X_train, y_train)")
                script.append("        y_pred = clf.predict(X_test)")
                script.append("        y_prob = clf.predict_proba(X_test)[:, 1]")
                script.append("        print(\"\\n--- RandomForestClassifier Evaluation ---\")")
                script.append("        print(f\"Accuracy: {accuracy_score(y_test, y_pred):.4f}\")")
                script.append("        print(f\"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\")")
                script.append("        print(f\"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\")")
                script.append("        print(f\"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\")")
                script.append("        print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, y_pred))")
                script.append("        print(\"Classification Report:\\n\", classification_report(y_test, y_pred))")
                script.append("        print(f\"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}\")")
                
                script.append("\n        # Plot Feature Importance")
                script.append("        plt.figure(figsize=(10, 6))")
                script.append("        importances = pd.Series(clf.feature_importances_, index=X_train.columns)")
                script.append("        importances.nlargest(10).plot(kind='barh')")
                script.append("        plt.title('Top 10 Feature Importances')")
                script.append("        plt.show()")
                
                script.append("\n        # Plot ROC Curve")
                script.append("        fpr, tpr, _ = roc_curve(y_test, y_prob)")
                script.append("        plt.figure(figsize=(8, 8))")
                script.append("        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')")
                script.append("        plt.plot([0, 1], [0, 1], 'k--')")
                script.append("        plt.title('ROC Curve')")
                script.append("        plt.legend()")
                script.append("        plt.show()")
                
                script.append("    except Exception as e:")
                script.append("        print(f\"Classification failed: {e}\")")
            
            if self.opt_regression.get():
                script.append("\n    # Regression Example (LinearRegression - from GUI setting)")
                script.append("    try:")
                script.append("        reg = LinearRegression()")
                script.append("        reg.fit(X_train, y_train)")
                script.append("        y_pred = reg.predict(X_test)")
                
                # Metrics
                script.append("        r2 = r2_score(y_test, y_pred)")
                script.append("        mae = mean_absolute_error(y_test, y_pred)")
                script.append("        mse = mean_squared_error(y_test, y_pred)")
                
                script.append("        print(\"\\n--- LinearRegression Evaluation ---\")")
                script.append("        print(f\"R-squared (R2) Score: {r2:.4f}\")")
                script.append("        print(f\"Mean Absolute Error (MAE): {mae:.4f}\")")
                script.append("        print(f\"Mean Squared Error (MSE): {mse:.4f}\")")
                
                script.append("    except Exception as e:")
                script.append("        print(f\"Regression failed: {e}\")")

        if self.opt_clustering.get():
            script.append("\n# Clustering Example (KMeans - from GUI setting)")
            script.append("num_data = df.select_dtypes(include=np.number).dropna()")
            script.append("try:")
            script.append("    # Note: K=3 is an arbitrary default. Use Elbow Method or Silhouette analysis to optimize.")
            script.append("    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')")
            script.append("    cluster_labels = kmeans.fit_predict(num_data)")
            script.append("    df['Cluster'] = cluster_labels")
            
            # Metrics
            script.append("    if len(num_data) > 1 and len(np.unique(cluster_labels)) > 1:")
            script.append("        silhouette_avg = silhouette_score(num_data, cluster_labels)")
            script.append("        print(f\"\\nSilhouette Score: {silhouette_avg:.4f} (Closer to 1 is better)\")")
            script.append("    else:")
            script.append("        print(\"\\nSilhouette Score requires at least 2 samples and 2 clusters.\")")

            script.append("    print(f\"Cluster counts:\\n{df['Cluster'].value_counts()}\")")
            script.append("except Exception as e:")
            script.append("    print(f\"Clustering failed: {e}\")")

        if self.opt_automl_autogluon.get():
            script.append("\n# --- AutoML: AutoGluon (Leveraging automatic feature engineering) ---")
            script.append(f"if '{target}' in df.columns:")
            
            # 1. Prepare data for AutoGluon by splitting it explicitly for evaluation
            script.append("    # AutoGluon handles cleaning/encoding, but we split manually for external evaluation.")
            script.append("    df.dropna(subset=['" + target + "'], inplace=True) # Drop rows with missing target")
            script.append("    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)")

            script.append("    # AutoGluon handles missing values, encoding, and model ensembling automatically.")
            script.append(f"    predictor = TabularPredictor(label='{target}', problem_type='binary' if len(df['{target}'].unique()) == 2 else 'multiclass').fit(train_data=train_data, time_limit=3600, presets='best_quality')")
            
            # 2. Evaluation using the test set
            script.append("\n    # --- AutoGluon Model Evaluation ---")
            script.append("    y_test = test_data['" + target + "']")
            
            # FIX 1: Change outer f-string to literal string
            script.append("    y_pred = predictor.predict(test_data.drop(columns=['" + target + "']))")
            
            # Explicitly print required metrics using sklearn
            script.append("    print(f\"\\nAutoGluon Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}\")")
            script.append("    print(f\"AutoGluon Test Set Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\")")
            script.append("    print(f\"AutoGluon Test Set Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\")")
            script.append("    print(f\"AutoGluon Test Set F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\")")

            # 3. Leaderboard and Feature Importance
            # FIX 2: Corrected to output literal string (was line 916 in local copy)
            script.append("    print(f\"\\nAutoGluon Leaderboard (Test Data):\\n{predictor.leaderboard(test_data, silent=True)}\")")
            # FIX 3: Corrected to output literal string (was line 917 in local copy)
            script.append("    print(f\"\\nFeature Importance:\\n{predictor.feature_importance(df)}\")") 


        if self.opt_automl_flaml.get():
            script.append("\n# --- AutoML: FLAML (Fast, Lightweight AutoML - Optimized for speed) ---")
            script.append(f"if '{target}' in df.columns:")
            script.append("    # Requires pre-processed X_train, y_train from Section 3.")
            script.append("    X = df.drop(columns=['" + self.opt_target_column.get() + "'])")
            script.append("    y = df['" + self.opt_target_column.get() + "']")
            script.append("    X = pd.get_dummies(X, drop_first=True)")
            script.append("    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)")
            script.append("    automl = AutoML()")
            script.append("    automl.fit(X_train=X_train, y_train=y_train, task='classification', time_budget=120)")
            
            script.append("    print(f\"\\nFLAML Best Model: {automl.model.estimator_name} with Score: {automl.best_score}\")")


        # --- Main block ---
        final_script = ["if __name__ == '__main__':"]
        for line in script:
            final_script.append(textwrap.indent(line, '    '))
        
        full_script = "\n".join(script)
        self.script_preview_text.delete(1.0, tk.END)
        self.script_preview_text.insert(tk.END, full_script)
        self.last_generated_script = full_script
        self.notebook.select(self.notebook.tabs()[3])

    def save_script(self):
        if not hasattr(self, 'last_generated_script'):
            messagebox.showinfo("Info", "Generate a script first.")
            return

        path = filedialog.asksaveasfilename(defaultextension='.py', filetypes=[('Python Script','*.py')])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.last_generated_script)
            self._log_operation(f"Saved generated script to {path}.")
            messagebox.showinfo('Success', f'Script saved to {path}')


if __name__ == '__main__':
    root = tk.Tk()
    app = DataExplorerGUI(root)
    root.geometry('1100x750')
    root.mainloop()