# 🔬 Data Explorer & Cleaner

A desktop GUI application for importing, cleaning, exploring, and analysing tabular data — with AI-powered column descriptions, interactive visualisations, and a full end-to-end machine learning script generator.

---

## 💡 Origin & Transparency

**The idea behind this app is entirely mine.** I wanted a single desktop tool that could handle the repetitive early stages of any data project — loading messy files, cleaning them up, getting a quick quality report, and generating reusable code — without having to write the same boilerplate every time.

**The code and this README were written by [Claude](https://claude.ai), Anthropic's AI assistant**, based on my requirements and direction. Every feature, workflow, and design decision was specified by me; Claude translated those ideas into working Python.

I'm sharing this project openly because I think it demonstrates something worth talking about: **you don't need to be a developer to build useful tools**. The creativity, the problem definition, and the domain knowledge came from me. The implementation came from an LLM. That combination is powerful, and I want to be honest about how it works rather than pretend otherwise.

---

## ✨ Features

### 📥 Import
- Load **CSV / TXT** and **Excel (.xlsx)** files
- Options to skip header rows or trim trailing rows on import

### 🔍 Exploration
- Preview the first or last 5 rows instantly
- Full **Data Quality Report** covering:
  - Missing values per column (count and percentage)
  - Duplicate rows
  - Empty columns and rows
  - Boolean-like columns
  - Text column uniqueness
  - Numeric descriptive statistics
  - Outlier detection using the IQR method
- Export the quality report as **TXT or CSV**

### 🤖 AI Column Descriptions
Describe every column in plain English using one of three free LLM providers — no paid subscription, no local installation needed:

| Provider | Free Key | pip package |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | `pip install groq` |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com) | `pip install google-generativeai` |
| **Cohere** | [dashboard.cohere.com](https://dashboard.cohere.com) | `pip install cohere` |

Switch between providers from a dropdown — each has its own API key field and model selector. The AI call runs in a background thread so the app never freezes while waiting for a response.

### 🧹 Cleaning
- Delete selected columns or rows (with confirmation dialogs)
- Delete all 100% empty columns or rows
- Trim N rows from the top or bottom
- **Single-level Undo** for all cleaning operations

### 📊 Visualisations
Interactive in-app charts powered by Matplotlib, rendered directly inside the app:
- Histograms for all numeric columns
- Correlation heatmap

### ⚙️ Python Script Generator
The most powerful feature. Configure options and generate a **standalone, reproducible `.py` script** you can run outside the app entirely. The generated script is fully self-contained and covers the complete data science workflow:

**Cleaning options:**
- Snake_case column name standardisation
- Whitespace trimming for text columns
- Deduplication
- Missing value handling (drop rows or fill with mean/unknown)

**Modelling — top 3 models per method, ranked by performance:**
- **Classification**: RandomForest, GradientBoosting, LogisticRegression — ranked by F1 score
- **Regression**: LinearRegression, Ridge, GradientBoosting — ranked by R²
- **Clustering**: KMeans with silhouette scoring
- **AutoML via AutoGluon**: configurable preset and time limit, full leaderboard output
- **AutoML via FLAML**: fast lightweight AutoML with top config summary

**Every modelling run automatically saves:**
- A ranked model comparison table (CSV)
- The test set used for accuracy evaluation (CSV)
- The best model as a `.pkl` file (reusable without retraining)
- Predictions on the held-out test split with actual vs predicted columns (CSV)
- Final predictions on a new unseen data file (CSV) — if provided
- A summary banner listing every output file and its size

**Script options:**
- Configurable train/test split size (e.g. 70/30, 80/20)
- Output folder picker — all files go to one place
- Browse and point to a new data file for final real-world predictions
- AutoGluon preset selector and time limit spinner

### 💾 Export
- Export the cleaned dataframe to **CSV** or **XLSX** at any point
- Save generated scripts as `.py` files with scrollable preview

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/data-explorer-cleaner.git
cd data-explorer-cleaner
```

### 2. Install core dependencies
```bash
pip install pandas numpy matplotlib seaborn openpyxl joblib
```

### 3. Install AI provider(s) — pick at least one
```bash
pip install groq                  # Groq (recommended — fastest)
pip install google-generativeai   # Google Gemini
pip install cohere                # Cohere
pip install scikit-learn       # Classification, Regression, Clustering
pip install autogluon          # AutoGluon AutoML
pip install flaml              # FLAML AutoML
```

### 4. Run the app
```bash
python main.py
```

---

## 🖥️ Requirements

- Python 3.7+
- Windows, macOS, or Linux (tkinter is included with standard Python)

---

## 📁 Project Structure

```
data-explorer-cleaner/
│
├── main.py          # Full application — single file, no external config needed
└── README.md        # This file
```

---

## 🆕 Recent Additions

These features were added after the initial release:

**AI Column Descriptions** — Button-activated LLM analysis of your dataset's columns. Supports three free providers (Groq, Gemini, Cohere) with a provider switcher, model dropdown per provider, and background threading so the UI stays responsive.

**Top 3 model comparison** — The script generator no longer trains a single model. For classification and regression it now trains three algorithms, evaluates each on the held-out test set, ranks them in a printed and saved table, and automatically selects the best one for predictions.

**Model saving** — The best model from each method is saved as a `.pkl` file using `joblib`, so you can reload and use it later without retraining.

**Test set export** — The exact rows used for accuracy evaluation are saved to a CSV, giving you full transparency over how performance was measured.

**Predict on new data** — A separate file browser in the Script Generator lets you point to a new, unlabelled dataset. After training and evaluating, the generated script loads this file, applies the same encoding pipeline, and saves final predictions to CSV. This is the complete real-world inference step.

**Configurable train/test split** — A spinbox lets you set the split ratio (default 0.30) instead of it being hardcoded.

**AutoGluon speed controls** — Preset dropdown (best_quality → medium_quality) and time limit spinner so you can trade off speed vs accuracy. Default is `medium_quality` + 120 seconds for fast results.

**Output folder picker** — All generated files (models, predictions, rankings, test sets) go to a single configurable folder with a folder browser button.

**Output summary banner** — The generated script prints a formatted table of every file it saved, with file sizes, so you know exactly what was produced when you run it.

**Script preview scrollbars** — The script preview pane now has both horizontal and vertical scrollbars for navigating long generated scripts.

**Splash screen & About dialog** — The app opens with a credits screen acknowledging Claude (Anthropic) as the code author, with an About button accessible at any time from the sidebar.

---

## 🤝 A Note on AI-Assisted Development

This project is an honest example of human-AI collaboration:

- **My contribution**: the concept, the feature requirements, the workflow design, and all decisions about what the tool should and shouldn't do
- **Claude's contribution**: writing the Python code and this README based on my specifications, iterating on the implementation as requirements evolved

I believe this kind of transparency matters. LLMs are tools — powerful ones — but they work best when a person with domain knowledge and a clear vision is driving. This project is proof that you can build something genuinely useful by combining your own creativity with AI assistance, and there's no reason to hide that.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with [Claude](https://claude.ai) by Anthropic — at the direction and creative vision of its author.*
<img width="686" height="486" alt="image" src="https://github.com/user-attachments/assets/2e0737aa-4a8c-4add-89ff-9a3ee093ca51" />
