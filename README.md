# 🛡️ VendorSense: Strategic Risk Management Dashboard

VendorSense is a powerful, data-driven dashboard designed to provide strategic and operational visibility into an organization's vendor contract portfolio. It translates complex performance metrics into clear, actionable risk and compliance scores, enabling rapid prioritization of critical vendor relationships and delayed projects.

This project demonstrates how low-code/no-code principles (like relational logic, calculated columns, and conditional scoring) can be rapidly prototyped using Python's Streamlit and Pandas library.

### 🚀 Key Features

This dashboard is divided into two primary views:

1. Vendor Health (Strategic View)
This view provides an aggregate, high-level analysis of vendor risk based on overall contract performance.

Custom Risk Scoring: Automatically calculates a Risk Score (High, Medium, Low) for every contract based on defined thresholds (Cost Overrun, Reported Issues, Days from Deadline).
Vendor Rollup: Aggregates contract data to show the total number of contracts and the percentage of contracts in High Risk for each vendor, enabling strategic prioritization.
Risk Visualization: Bar charts compare total contract volume against the high-risk count per vendor.

2. Contracts Queue (Operational View)
This view focuses on immediate action, displaying only contracts flagged for attention.

Compliance Flagging: Automatically flags contracts that violate defined business rules (e.g., T&M contracts used for IT Infrastructure projects or mandatory clause waivers).
Filterable Queue: Allows users to filter contracts by Risk Score and Compliance Flag to quickly generate worklists for corrective action.
Project Delay Tracking (New): Features a clear line chart highlighting all contracts that are currently behind schedule (Days from Deadline > 0), identifying the most significant delays instantly.

### 🛠️ Technology Stack
Language: Python
Dashboard Framework: Streamlit

Data Handling: Pandas & NumPy
