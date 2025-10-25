# 🚀 Quick Start Guide

Get up and running with the Anomaly Detection & Explanation system in 5 minutes!

## Prerequisites Check

✅ Python 3.10 or higher installed  
✅ pip package manager available  
✅ (Optional) Groq API key for LLM explanations

## Step-by-Step Setup

### 1️⃣ Install Dependencies (2 minutes)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
```

### 2️⃣ Configure API Key (1 minute - Optional)

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# Get a free key at: https://console.groq.com
```

**Without API Key?** No problem! The system will use template-based explanations.

### 3️⃣ Run the Application (30 seconds)

```bash
# Start the Streamlit app
streamlit run app/app.py
```

Your browser will open automatically at `http://localhost:8501` 🎉

### 4️⃣ Try the Sample Data (2 minutes)

1. **Upload & Preview** page:
   - ✅ Check "Use sample data"
   - Click "Build Features"
2. **Detect Anomalies** page:
   - Leave default settings (k=2.5)
   - Click "🚀 Run Detection"
3. **Explain & Review** page:

   - Click "🧠 Generate Explanations"
   - Select an anomaly
   - Review the explanation
   - Provide feedback

4. **User Study Results** page:
   - View performance metrics
   - Analyze feedback

## 🎯 What to Expect

With the sample data, you should see:

- **~120 total records** loaded
- **8-12 anomalies** detected (includes 8 intentionally injected)
- **Detection time**: ~5 seconds
- **Explanation generation**: ~2-5 seconds per row (with LLM)

## 🔍 Sample Anomalies

The sample data includes these injected anomalies:

1. **Late Night Transfer** - Transaction at 02:30 AM
2. **Duplicate Entry** - Same amount, vendor, date
3. **Weekend Transaction** - Posted on Saturday night
4. **Unbalanced Voucher** - Missing credit entry
5. **Large Round Amount** - $25,000 suspicious transaction

## ⚙️ Quick Settings

### Adjust Sensitivity

In the **Detect Anomalies** page:

- **k = 2.0**: More anomalies (more sensitive)
- **k = 2.5**: Balanced (default)
- **k = 3.0**: Fewer anomalies (more conservative)

### Feature Flags

In `core/config.py`:

```python
ENABLE_XGBOOST = True   # Use advanced ML
ENABLE_SHAP = True      # Show feature importance
ENABLE_LLM = True       # Natural language explanations
```

## 📊 Your Own Data

### CSV Format Required

```csv
date,voucher_id,account,debit,credit,amount,vendor,poster,description
2024-10-01 09:00:00,V001,1000-CASH,1000.00,0.00,1000.00,Vendor,User,Description
```

### Minimum Required Columns

- `date` - Transaction timestamp
- `voucher_id` - Unique identifier
- `account` - Account code
- `amount` - Transaction amount

### Optional Columns

- `debit`, `credit` - For double-entry validation
- `vendor`, `poster` - For frequency analysis
- `description` - For context

## 🧪 Verify Installation

Run tests to ensure everything works:

```bash
# Run all tests (should take ~10 seconds)
pytest tests/ -v

# Expected output:
# ✅ test_features.py .............. [ 40%]
# ✅ test_detectors.py ............ [ 75%]
# ✅ test_llm_schema.py ........... [100%]
# ===================== XX passed in X.XXs =====================
```

## 🆘 Troubleshooting

### Issue: "Module not found"

**Solution**: Activate virtual environment and reinstall

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"

**Solution**: Kill existing process or use different port

```bash
streamlit run app/app.py --server.port 8502
```

### Issue: "Groq API error"

**Solution**: Check API key in `.env` file, or disable LLM

```python
# In core/config.py
ENABLE_LLM = False
```

### Issue: Tests failing

**Solution**: Ensure all dependencies installed

```bash
pip install pytest
pytest tests/ -v
```

## 🎓 Next Steps

1. **Explore the UI**: Try all 4 pages
2. **Adjust Settings**: Experiment with threshold sensitivity
3. **Upload Your Data**: Test with real ledger files
4. **Review Explanations**: Understand the reasoning
5. **Provide Feedback**: Help improve the model
6. **Check Analytics**: View performance metrics

## 📚 Learn More

- **Full Documentation**: See [README.md](README.md)
- **Data Format**: See [data/README.md](data/README.md)
- **Architecture**: Check README Architecture section
- **Contributing**: Open issues or PRs on GitHub

## 💡 Tips for Best Results

1. **Start Small**: Test with sample data first
2. **Iterate**: Adjust k based on your false positive tolerance
3. **Review High-Risk First**: Focus on high anomaly scores
4. **Provide Feedback**: User feedback improves metrics
5. **Export Results**: Download CSV for reporting

## ⚡ Using Makefile (macOS/Linux)

```bash
make setup   # One-command setup
make run     # Start the app
make test    # Run tests
make clean   # Clean cache files
```

## 🎉 You're Ready!

The system is now running and ready to detect anomalies in your accounting data.

**Need Help?**

- 📖 Read the [full README](README.md)
- 🐛 Open an issue on GitHub
- 💬 Check inline tooltips in the UI

---

**Happy anomaly hunting! 🔍✨**
