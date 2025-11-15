import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Data Generation and Initial Setup ---

@st.cache_data
def load_data():
    """
    Creates dummy dataframes for Contracts and Vendors, mimicking the
    Google Sheets structure provided in the initial project files.
    """
    # Vendor Master Data (Static)
    vendor_data = {
        'Vendor ID': ['V001', 'V002', 'V003', 'V004', 'V005'],
        'Vendor Name': ['EcoFusion Technologies', 'Beacon Health Solutions', 'Dynamized Learning', 'Synergy Research Partners', 'Apex Precision Systems (APS)']
    }
    df_vendors = pd.DataFrame(vendor_data)

    # Contracts Details Data (Operational Records)
    contract_data = {
        'Contract ID': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010', 'C011', 'C012', 'C013', 'C014', 'C015'],
        'Contract Name': [
            'ERP System Implementation - 1', 'Economic Impact Study', 'Electronic Records Upgrade', 'Water Treatment Plant Upgrade',
            'Staff Augmentation (Audit)', 'Sustainability Training', 'ERP System Implementation - 2', 'Digital Transformation Strategy',
            'Community Development Impact Study', 'Health Benefit Portal Dev', 'Smart Solar Grid Pilot', 'Facility Management Software',
            'Cybersecurity Audit', 'HR Policy Consulting', 'Office Supply Contract'
        ],
        'Vendor ID': ['V005', 'V004', 'V002', 'V001', 'V004', 'V003', 'V005', 'V005', 'V004', 'V002', 'V001', 'V003', 'V005', 'V002', 'V003'],
        'Contract Value': [50000, 250000, 1000000, 8000000, 150000, 40000, 50000, 500000, 190000, 340000, 1200000, 75000, 300000, 120000, 15000],
        'Cost Overrun %': [0.15, 0.05, 0.01, 0.20, 0.02, 0.05, 0.00, 0.08, 0.01, 0.04, 0.00, 0.10, 0.05, 0.03, 0.00], # Decimal
        'Reported Issues Count': [8, 3, 5, 4, 20, 17, 0, 6, 3, 9, 1, 12, 4, 2, 0],
        'Days from Deadline': [45, 30, -8, -15, -1, -25, 110, -2, -5, 25, 60, -10, 5, 15, 30], # Positive is late, Negative is early
        'Project Type': ['IT Infrastructure', 'Policy Making', 'IT Infrastructure', 'Engineering', 'Audit', 'HR L&D', 'IT Infrastructure', 'IT Infrastructure', 'Policy Making', 'IT Infrastructure', 'Engineering', 'IT Infrastructure', 'IT Infrastructure', 'Consulting', 'Administrative'],
        'Payment Schedule Type': ['Milestone', 'T&M', 'T&M', 'Milestone', 'T&M', 'Fixed Price', 'Milestone', 'T&M', 'Milestone', 'Milestone', 'Fixed Price', 'T&M', 'Fixed Price', 'Fixed Price', 'Fixed Price'],
        'Mandatory Clause Waiver?': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No']
    }
    df_contracts = pd.DataFrame(contract_data)
    
    # Merge for display purposes
    df_contracts = df_contracts.merge(df_vendors, on='Vendor ID', how='left')
    return df_contracts, df_vendors

# --- 2. Low-Code Logic Engine (Computed Columns) ---

def apply_risk_logic(df_contracts):
    """
    Applies the custom risk scoring logic to the contracts DataFrame.
    IMPORTANT: Creates a deep copy to prevent Streamlit/Pandas cache issues.
    """
    # Create a deep copy to ensure modifications don't interfere with cached objects
    df = df_contracts.copy(deep=True) 
    
    # Initialize the Risk Score column
    df['Risk Score'] = 'LOW RISK'

    # Rule 1: HIGH RISK (Cost Overrun > 10% OR > 10 Issues OR > 30 Days Late)
    high_risk_condition = (
        (df['Cost Overrun %'] > 0.10) |
        (df['Reported Issues Count'] > 10) |
        (df['Days from Deadline'] > 30)
    )
    df.loc[high_risk_condition, 'Risk Score'] = 'HIGH RISK'

    # Rule 2: MEDIUM RISK (Cost Overrun > 5% OR 5-10 Issues OR 10-30 Days Late)
    # We only apply this to contracts that aren't already HIGH RISK
    medium_risk_condition = (
        (df['Risk Score'] == 'LOW RISK') & (
            (df['Cost Overrun %'] > 0.05) |
            ((df['Reported Issues Count'] >= 5) & (df['Reported Issues Count'] <= 10)) |
            ((df['Days from Deadline'] >= 10) & (df['Days from Deadline'] <= 30))
        )
    )
    df.loc[medium_risk_condition, 'Risk Score'] = 'MEDIUM RISK'

    return df # Return the modified copy

def determine_compliance_flag(row):
    """
    Helper function to determine compliance status for a single row.
    """
    is_tm_violation = (row['Payment Schedule Type'] == 'T&M') and (row['Project Type'] == 'IT Infrastructure')
    is_waiver_violation = (row['Mandatory Clause Waiver?'] == 'Yes')

    flags = []
    if is_tm_violation:
        flags.append('T&M Policy Violation')
    if is_waiver_violation:
        flags.append('Waiver Required')

    if flags:
        # Join multiple flags with the pipe separator
        return 'FLAGGED: ' + ' | '.join(flags)
    return 'PASS'

def apply_compliance_logic(df_contracts):
    """
    Applies the proxy compliance flagging logic using the robust helper function.
    IMPORTANT: Creates a deep copy to prevent Streamlit/Pandas cache issues.
    """
    # Create a deep copy to ensure modifications don't interfere with cached objects
    df = df_contracts.copy(deep=True) 

    # Apply the helper function row-by-row to calculate the Compliance Flag
    df['Compliance Flag'] = df.apply(determine_compliance_flag, axis=1)
    
    return df # Return the modified copy

def aggregate_vendor_health(df_contracts, df_vendors):
    """
    Aggregates contract-level data up to the vendor level.
    This mimics the Glide 'Relation & Rollup' columns.
    """
    # Count total contracts per vendor
    df_vendor_counts = df_contracts.groupby('Vendor ID').size().reset_index(name='Total Contracts')

    # Count high-risk contracts per vendor (Rollup with Filter)
    df_high_risk = df_contracts[df_contracts['Risk Score'] == 'HIGH RISK'].groupby('Vendor ID').size().reset_index(name='High Risk Count')

    # Merge aggregations into the Vendor Master table
    df_health = df_vendors.merge(df_vendor_counts, on='Vendor ID', how='left').fillna(0)
    df_health = df_health.merge(df_high_risk, on='Vendor ID', how='left').fillna(0)

    # Calculate High Risk Percentage
    df_health['High Risk %'] = (df_health['High Risk Count'] / df_health['Total Contracts']) * 100
    df_health['High Risk %'] = df_health['High Risk %'].replace([np.inf, -np.inf], 0).round(1)

    # Sort by the most problematic vendors
    df_health = df_health.sort_values(by=['High Risk Count', 'Total Contracts'], ascending=[False, False])

    return df_health

# --- 3. Streamlit Application Layout ---

def format_metric(value, prefix='Dh', suffix=''):
    """Helper for displaying metric changes with clean formatting."""
    if value >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.1f}M{suffix}"
    return f"{prefix}{value:,}{suffix}"


def run_vendorsense_app():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="VendorSense Dashboard")

    st.title("🛡️ VendorSense: strategic vendor risk management")
    st.markdown("A demonstration of Low-Code/No-Code logic translated to a Python Streamlit dashboard.")
    st.divider()

    # Load and process data
    # The functions below ensure a deep copy, preventing the KeyError
    df_contracts, df_vendors = load_data()
    df_contracts = apply_risk_logic(df_contracts)
    df_contracts = apply_compliance_logic(df_contracts)
    df_health = aggregate_vendor_health(df_contracts, df_vendors)

    # --- Global Metrics Bar ---
    total_contracts = len(df_contracts)
    total_value = df_contracts['Contract Value'].sum()
    high_risk_contracts = df_contracts[df_contracts['Risk Score'] == 'HIGH RISK'].shape[0]
    flagged_compliance = df_contracts[df_contracts['Compliance Flag'].str.contains('FLAGGED')].shape[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Active Contracts", f"{total_contracts:,}")
    with col2:
        # Adjusting the format_metric call for clarity on Portfolio Value
        st.metric("Portfolio Value", format_metric(total_value))
    with col3:
        st.metric("High Risk Contracts (Apex Metric)", f"{high_risk_contracts:,}", delta=f"{high_risk_contracts / total_contracts * 100:.1f}% of total", delta_color="inverse")
    with col4:
        st.metric("Compliance Flags Raised", f"{flagged_compliance:,}")

    st.divider()

    # --- Tabs for Dual View ---
    tab_strategic, tab_operational = st.tabs(["📊 Vendor Health (Strategic View)", "📋 Contracts Queue (Operational View)"])

    # --- Strategic View: Vendor Health ---
    with tab_strategic:
        st.header("Strategic Vendor Health Assessment")
        st.markdown("Prioritize vendors based on the quantity and percentage of high-risk contracts they hold.")

        # 1. Vendor Risk Summary Table
        st.subheader("Top Riskiest Vendors")
        st.dataframe(
            df_health[['Vendor Name', 'Total Contracts', 'High Risk Count', 'High Risk %']]
            .rename(columns={'High Risk Count': 'High Risk Contracts', 'High Risk %': 'High Risk % of Portfolio'}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "High Risk % of Portfolio": st.column_config.ProgressColumn(
                    "High Risk % of Portfolio",
                    format="%f %%",
                    min_value=0,
                    max_value=100,
                    width='large'
                ),
            }
        )

        # 2. Vendor Risk vs. Volume Chart
        st.subheader("Risk Count vs. Total Contract Volume")
        st.bar_chart(
            df_health.set_index('Vendor Name')[['Total Contracts', 'High Risk Count']],
            color=["#2563eb", "#ef4444"], # Blue for Total, Red for High Risk
            use_container_width=True,
            height=300
        )

    # --- Operational View: Contracts Queue ---
    with tab_operational:
        st.header("Operational Contracts Queue")
        st.markdown("Filter to view contracts requiring immediate attention due to Risk or Compliance issues.")

        # Sidebar Filters
        st.sidebar.header("Operational Filters")
        risk_filter = st.sidebar.multiselect(
            "Filter by Risk Score",
            options=df_contracts['Risk Score'].unique(),
            default=['HIGH RISK', 'MEDIUM RISK']
        )
        # Ensure we only show flags, not PASS
        compliance_options = [flag for flag in df_contracts['Compliance Flag'].unique() if flag != 'PASS']
        compliance_filter = st.sidebar.multiselect(
            "Filter by Compliance Flag",
            options=compliance_options,
            default=compliance_options
        )

        # Apply Filters
        df_filtered = df_contracts[df_contracts['Risk Score'].isin(risk_filter)]
        # This line was where the first error occurred, now it should be safe
        df_filtered = df_filtered[df_filtered['Compliance Flag'].isin(compliance_filter)]

        st.info(f"Showing **{len(df_filtered)}** contracts matching the selected filters.")

        # Display Filtered Contracts
        st.dataframe(
            df_filtered[[
                'Contract ID', 'Contract Name', 'Vendor Name', 'Contract Value',
                'Risk Score', 'Compliance Flag', 'Cost Overrun %', 'Days from Deadline'
            ]].style.map(
                lambda x: 'background-color: #f84e4e' if x == 'HIGH RISK' else 'background-color: ##fdaf78' if x == 'MEDIUM RISK' else '',
                subset=['Risk Score']
            ).format({
                'Contract Value': '${:,.0f}',
                'Cost Overrun %': '{:.1%}',
                'Days from Deadline': lambda x: f'{int(x)} days late' if x > 0 else f'{int(abs(x))} days early',
            }),
            use_container_width=True,
            hide_index=True
        )

        # --- LINE CHART FOR LATE CONTRACTS ---
        st.subheader("Contracts Behind Schedule (Days Late)")

        # 1. Filter for contracts that are behind schedule (Days from Deadline > 0)
        df_late_contracts = df_contracts[df_contracts['Days from Deadline'] < 0].sort_values(
            by='Days from Deadline', ascending=False
        )

        if not df_late_contracts.empty:
            st.line_chart(
                # Use Contract Name as the X-axis index
                df_late_contracts.set_index('Contract Name')['Days from Deadline'],
                color="#f97316", # Orange/Amber color for warning
                height=300
            )
        else:
            st.success("🎉 All contracts are currently on time or ahead of schedule!")
        # --- END LINE CHART ---


# Run the application
if __name__ == "__main__":
    run_vendorsense_app()

