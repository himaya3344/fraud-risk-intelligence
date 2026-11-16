# Fraud Risk Intelligence — Internship Project (Day 01 Starter)

Source: uploaded `DS 02(1).pdf` assignment and `transaction_fraud_risk_advanced_dataset.csv`.

## Day 01 — Windows PowerShell / VS Code
1. Extract this ZIP and open the extracted `fraud-risk-intelligence-day01` folder in VS Code.
2. Open Terminal > New Terminal (PowerShell).
3. Confirm Python: `py --version` (if unavailable, try `python --version`).
4. Create the virtual environment: `py -m venv .venv`.
5. Install packages without needing environment activation:
   `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
6. In VS Code, select the interpreter `.venv\Scripts\python.exe` using Python: Select Interpreter.
7. Run `01_fraud_risk_analysis.ipynb` using its notebook kernel, OR verify with:
   `.\.venv\Scripts\python.exe src\day01_check.py`

## Expected initial results
- 3,000 transactions and 25 columns
- Label `fraud`: 364 fraud / 2,636 legitimate (12.13% fraud)
- 0 fully duplicated rows
- Missing values: transaction_amount=53, distance_from_usual_location_km=75, merchant_risk_score=60, ip_risk_score=45, login_to_transaction_minutes=60

## Important rules
- Keep `data/raw/transaction_fraud_risk_advanced_dataset.csv` unchanged.
- Do not fit a model or fill missing values on Day 01.
- A true chronological split cannot be inferred from `hour` and `day_of_week` alone.
- `transaction_id` is an identifier, not a sensible predictive feature.
- `reports/` will contain figures/reports, and `models/` will contain trained models in later days.
- The final assignment requires 8 named artifacts; this starter notebook will be extended over later days.

Note: `authentication_level` contains a valid string `None` category (289 records). Use `pd.read_csv(..., keep_default_na=False, na_values=[""])` to avoid falsely counting these as missing.
