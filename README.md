# 🏏 Snowflake + LLM for IPL Data Analysis

A comprehensive data analysis platform that integrates **Snowflake** data warehouse with **local Large Language Models (LLMs)** and **Natural Language Processing (NLP)** to analyze Indian Premier League (IPL) cricket statistics and fan sentiment.

## 📋 Overview

This project demonstrates the power of combining Snowflake's cloud data warehouse with state-of-the-art local LLM models to extract meaningful insights from IPL cricket data. It includes sentiment analysis, emotion detection, and sophisticated statistical analysis of player and team performance across IPL 2024 and 2025 seasons.

### Key Features
- 🤖 **Local LLM Integration** - Uses HuggingFace transformers for sentiment analysis and emotion detection
- 📊 **Multi-dimensional Analysis** - Bowler stats, team performance, venue analytics, and top scorers
- ☁️ **Snowflake Integration** - Seamless data querying and analysis from Snowflake warehouse
- 📈 **IPL 2024 & 2025 Coverage** - Comprehensive statistics for both seasons
- 💬 **Fan Sentiment Analysis** - Analyzes IPL fan comments for sentiment and emotions
- ⚡ **PyTorch & Transformers** - GPU-optimized NLP models for faster processing

## 🎯 Project Structure

```
Snowflake-LLM-For-DataAnalysis/
├── README.md
├── requirements.txt
├── llm_demo.py                      # LLM QA demo with IPL context
├── nlp_demo.py                      # NLP demonstration with local models
├── snowflake_llm_integration.py     # Snowflake + LLM integration demo
└── scripts/
    ├── ipl_2025_bowler_analysis.py      # Bowler performance analysis for IPL 2025
    ├── ipl_2025_team_analysis.py        # Team performance analysis for IPL 2025
    ├── ipl_2025_venue_analysis.py       # Venue statistics for IPL 2025
    └── ipl_top_scorers_analysis.py      # Top run scorers analysis for IPL 2024
```

## 📊 Analysis Modules

### 1. **Bowler Performance Analysis** (`ipl_2025_bowler_analysis.py`)
Comprehensive analysis of bowler statistics including:
- Top wicket-takers
- Economy rates
- Bowling averages
- Strike rates
- LLM-powered performance insights

### 2. **Team Performance Analysis** (`ipl_2025_team_analysis.py`)
Team-level analytics covering:
- Total match wins
- Win rates
- Home vs away performance
- LLM-generated team performance summaries

### 3. **Venue Statistics** (`ipl_2025_venue_analysis.py`)
Venue-specific insights including:
- Match counts per venue
- City-wise venue popularity
- Home team advantage analysis
- Venue-specific team performance

### 4. **Top Scorers Analysis** (`ipl_top_scorers_analysis.py`)
Batting performance metrics:
- Top 10 run scorers for IPL 2024
- Individual player statistics
- Consistent vs breakthrough performances
- LLM-powered player insights

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (optional but recommended)
- Snowflake account with IPL data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kagank01/Snowflake-LLM-For-DataAnalysis.git
   cd Snowflake-LLM-For-DataAnalysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   source .venv/bin/activate      # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

**For Snowflake Integration:**

Update connection details in the scripts:
```python
conn = snowflake.connector.connect(
    account='your_account',
    user='your_username',
    password='your_password',
    database='IPL_ANALYSIS',
    schema='YEAR_2024_2025'
)
```

### Running the Demos

#### 1. **NLP Demo** - Local model demonstrations
```bash
python nlp_demo.py
```
Demonstrates sentiment analysis and emotion detection on cricket-related text.

#### 2. **LLM QA Demo** - Question answering with IPL context
```bash
python llm_demo.py
```
Shows question-answering capabilities using LLMs with IPL cricket context.

#### 3. **Snowflake Integration Demo** - Complete integration example
```bash
python snowflake_llm_integration.py
```
Analyzes fan comments using local NLP models.

#### 4. **IPL Analysis Scripts** - Specialized analysis
```bash
# IPL 2025 Bowler Analysis
python scripts/ipl_2025_bowler_analysis.py

# IPL 2025 Team Analysis
python scripts/ipl_2025_team_analysis.py

# IPL 2025 Venue Analysis
python scripts/ipl_2025_venue_analysis.py

# IPL 2024 Top Scorers
python scripts/ipl_top_scorers_analysis.py
```

## 📦 Dependencies

Core libraries used in this project:

- **torch** - PyTorch for deep learning and GPU acceleration
- **transformers** - HuggingFace transformers for NLP models
- **snowflake-connector-python** - Snowflake database connector
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis

See `requirements.txt` for complete list.

## 🔧 Key Technologies

| Technology | Purpose |
|-----------|---------|
| **Snowflake** | Cloud data warehouse for IPL data storage & querying |
| **PyTorch** | Deep learning framework with GPU support |
| **Transformers** | Pre-trained NLP models (CardiffNLP, DistilRoBERTa, etc.) |
| **Python 3.10+** | Primary programming language |

## 📈 Supported Analysis

### IPL Seasons
- **IPL 2024** - Complete season data with player statistics
- **IPL 2025** - Current season bowler, team, venue analytics

### Analysis Types
- **Sentiment Analysis** - Positive/Negative/Neutral sentiment detection
- **Emotion Detection** - Joy, Anger, Surprise, Sadness, etc.
- **Performance Metrics** - Statistics extraction and aggregation
- **Comparative Analysis** - Player and team performance comparisons
- **LLM Insights** - AI-powered narrative generation from data

## 🎓 Use Cases

1. **Cricket Analytics** - In-depth IPL player and team performance analysis
2. **Sentiment Analysis** - Monitor fan reactions and sentiment trends
3. **Data Warehouse Integration** - Learn Snowflake + Python integration
4. **NLP Pipeline** - Understand local LLM deployment and usage
5. **Sports Analytics** - Build analytics dashboards for cricket
6. **AI-Powered Insights** - Generate natural language insights from raw data

## 📝 Example Output

### Bowler Analysis
```
🏏 IPL 2025 Bowler Performance Analysis
🤖 Analyzing IPL 2025 Bowler Performance with Local LLM

🏆 Top Wicket-Takers in IPL 2025:
Jasprit Bumrah took 28 wickets in IPL 2025, conceding 450 runs in 720 balls, 
showing exceptional bowling performance.
```

### Sentiment Analysis
```
📝 Sample Comment: "What an amazing innings by Virat Kohli!"
   Sentiment: POSITIVE (0.998)
   Emotion: JOY (0.945)
```

## 🔄 Snowflake Schema Reference

### Tables Used
- `IPL_24_25_BALL_BY_BALL` - Ball-by-ball match data for 2024-2025
- `IPL_MATCH_SUMMARY_2024_2025` - Match summaries for both seasons

### Schema Location
- **Database**: `IPL_ANALYSIS`
- **Schema**: `YEAR_2024_2025`

## ⚙️ Performance Tips

1. **GPU Acceleration** - Ensure PyTorch is using GPU if available
2. **Model Caching** - Models are cached after first download
3. **Batch Processing** - Process fan comments in batches for efficiency
4. **Snowflake Query Optimization** - Use specific WHERE clauses to filter data

## 🐛 Troubleshooting

### Virtual Environment Issues
If you encounter `pip.exe` launcher errors after renaming folders:
```bash
python -m venv --upgrade .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Snowflake Connection Errors
- Verify account, user, and password credentials
- Check network connectivity to Snowflake
- Ensure database and schema names are correct

### Model Download Issues
- First run downloads models (~500MB+) - ensure sufficient disk space
- Models are cached in `~/.cache/huggingface/hub/`
- Check internet connectivity

## 📚 Resources

- [Snowflake Python Connector Docs](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [IPL Cricket Data](https://www.iplt20.com/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest new analysis features
- Optimize existing code
- Add support for more IPL seasons

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Ankit** - [GitHub Profile](https://github.com/kagank01)

## 📧 Support

For questions or support, please open an issue in the GitHub repository.

---

**Happy Analyzing! 🏏⚡**
