# Automation-in-Data-Science

# 🔬 Data Explorer & Cleaner

A desktop GUI application for importing, cleaning, exploring, and analysing tabular data — with AI-powered column descriptions and reproducible Python script generation.

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
Describe every column in plain English using one of three free LLM providers — no paid subscription needed:

| Provider | Free Key | pip package |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | `pip install groq` |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com) | `pip install google-generativeai` |
| **Cohere** | [dashboard.cohere.com](https://dashboard.cohere.com) | `pip install cohere` |

Switch between providers from a dropdown — each has its own API key field and model selector. The AI runs in a background thread so the app never freezes.

### 🧹 Cleaning
- Delete selected columns or rows (with confirmation)
- Delete all 100% empty columns or rows
- Trim N rows from the top or bottom
- **Single-level Undo** for all cleaning operations

### 📊 Visualisations
Interactive in-app charts powered by Matplotlib:
- Histograms for all numeric columns
- Correlation heatmap

### ⚙️ Python Script Generator
Configure and generate a **standalone, reproducible `.py` script** with options for:
- Cleaning steps (snake_case column names, whitespace trimming, deduplication, missing value handling)
- Basic EDA plots
- Machine learning models: **RandomForest**, **LinearRegression**, **KMeans**
- AutoML via **AutoGluon** or **FLAML**

### 💾 Export
- Export cleaned data to **CSV** or **XLSX**
- Save generated scripts as `.py` files

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Bongiwe-Tshabalala-DS/data-explorer-cleaner.git
cd data-explorer-cleaner
```

### 2. Install core dependencies
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

### 3. Install AI provider(s) — pick at least one
```bash
pip install groq                  # Groq
pip install google-generativeai   # Google Gemini
pip install cohere                # Cohere
```

### 4. Run the app
```bash
python main_llm.py
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

## 🛠️ Optional: ML & AutoML Dependencies

These are only needed if you use the Script Generator's modelling options:

```bash
pip install scikit-learn       # RandomForest, LinearRegression, KMeans
pip install autogluon          # AutoGluon AutoML
pip install flaml              # FLAML AutoML
```

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

<img width="686" height="486" alt="image" src="https://github.com/user-attachments/assets/060def29-89b6-445e-96d4-9135c8e1dae2" />
