# Sample Ledger Data

This directory contains sample accounting ledger data for testing the anomaly detection system.

## File: `sample_ledger.csv`

### Structure

The CSV file contains 105 accounting entries with the following columns:

- **date**: Transaction timestamp (YYYY-MM-DD HH:MM:SS)
- **voucher_id**: Unique voucher identifier (e.g., V1001)
- **account**: Account code (e.g., 1000-CASH, 5000-EXPENSE)
- **debit**: Debit amount (positive or 0)
- **credit**: Credit amount (positive or 0)
- **amount**: Net transaction amount (debit - credit)
- **vendor**: Vendor/supplier name
- **poster**: User who posted the transaction
- **description**: Transaction description

### Injected Anomalies

The sample data includes intentionally injected anomalies** for testing, for example:

1. **Late Night Transfer (Rows 20-21)**: Transaction at 02:30 AM with rare vendor and unknown user
2. **Duplicate Entry**: Identical transaction from same vendor on same day at same time
3. **Late Night Transaction**: Transaction at 23:15 on Monday (unusual timing)
4. **Large Round Amount (Row 54)**: Unusual $25,000 round amount with suspicious vendor (V1028)
5. **Rare Vendor Late Hour**: Multiple transactions from infrequent vendors at unusual times
6. **Unusual Account Pair**: Rare combinations of account codes within vouchers
7. **High Z-Score Amounts**: Transactions with amounts significantly higher than account averages
8. **Unbalanced Voucher**: Voucher V1028 has only debit entry, missing credit (unbalanced)

### Normal Patterns

The majority of entries follow normal patterns:

- **Business Hours**: Most transactions between 8:00-18:00
- **Balanced Vouchers**: Each voucher has matching debit and credit entries
- **Regular Vendors**: ABC Corp, Tech Vendor, Office Depot, XYZ Ltd appear frequently
- **Common Accounts**: 1000-CASH, 5000-EXPENSE, 4000-REVENUE are standard
- **Typical Amounts**: Range from $950 to $7,800 for normal operations

### Usage

```python
import pandas as pd
df = pd.read_csv('data/sample_ledger.csv')
print(f"Loaded {len(df)} entries")
```

### Expected Detection Results

With default settings (k=2.5), the system should flag approximately:

- **8-12 anomalies** (includes the 8 injected + some borderline cases)
- **High-risk items**: Large round amount ($25,000), late night transactions (02:30 AM, 23:15)
- **Medium-risk items**: Duplicates, unusual vendor-hour combinations, unbalanced vouchers
- **Low-risk items**: Borderline statistical outliers

### Data Generation

This data was carefully crafted to simulate realistic accounting scenarios with:

- Temporal patterns (business hours, weekdays, month-end)
- Vendor relationships (frequency distributions)
- Account hierarchies (cash, expense, revenue categories)
- Voucher balance constraints (double-entry bookkeeping)

All entries are fictional and created for demonstration purposes only.
