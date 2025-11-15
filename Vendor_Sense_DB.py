import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Data Loading from CSV Files ---

@st.cache_data
def load_data():
    """
    Loads dataframes for Contracts and Vendors directly from the uploaded
    CSV files that represent the Excel sheets.
    """
    # File names derived from the uploaded "VendorSense_Dataset.xlsx" content
    file = "VendorSense Dataset.xlsx"

    try:
        # Read Contracts Details
        df_contracts = pd.read_excel(file, sheet_name="Contracts Details")
        
        # Read Vendor Master (dropping duplicates for clean lookup)
        df_vendors = pd.read_excel(file, sheet_name="Vendor Master").drop_duplicates(subset=['Vendor ID'])
        
        # Data Cleaning and Type Conversion (Essential after reading from file)
        # Convert performance metrics to correct numeric types
        df_contracts['Contract Value'] = pd.to_numeric(df_contracts['Contract Value'], errors='coerce')
        df_contracts['Cost Overrun %'] = pd.to_numeric(df_contracts['Cost Overrun %'], errors='coerce')
        df_contracts['Reported Issues Count'] = pd.to_numeric(df_contracts['Reported Issues Count'], errors='coerce').fillna(0).astype(int)
        df_contracts['Days from Deadline'] = pd.to_numeric(df_contracts['Days from Deadline'], errors='coerce').fillna(0).astype(int)

        return df_contracts, df_vendors
    except FileNotFoundError:
        st.error(f"Required data file not found. Please ensure '{file}' is accessible.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs on failure

# --- 2. Low-Code Logic Engine (Computed Columns) ---
# Note: Deep copy is maintained in these functions for stability.

def apply_risk_logic(df_contracts):
    """
    Applies the custom risk scoring logic to the contracts DataFrame.
    Creates a deep copy to prevent Streamlit/Pandas cache issues.
    """
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
    # Apply only to contracts that aren't already HIGH RISK
    medium_risk_condition = (
        (df['Risk Score'] == 'LOW RISK') & (
            (df['Cost Overrun %'] > 0.05) |
            ((df['Reported Issues Count'] >= 5) & (df['Reported Issues Count'] <= 10)) |
            ((df['Days from Deadline'] >= 10) & (df['Days from Deadline'] <= 30))
        )
    )
    df.loc[medium_risk_condition, 'Risk Score'] = 'MEDIUM RISK'

    return df

def determine_compliance_flag(row):
    """
    Helper function to determine compliance status for a single row.
    """
    is_tm_violation = (row['Payment Schedule Type'] == 'T&M') and (row['Project Type'] == 'IT infrastructure')
    is_waiver_violation = (row['Mandatory Clause Waiver?'] == 'Yes')

    flags = []
    if is_tm_violation:
        flags.append('T&M Policy Violation')
    if is_waiver_violation:
        flags.append('Waiver Required')

    if flags:
        return 'FLAGGED: ' + ' | '.join(flags)
    return 'PASS'

def apply_compliance_logic(df_contracts):
    """
    Applies the proxy compliance flagging logic.
    Creates a deep copy to prevent Streamlit/Pandas cache issues.
    """
    df = df_contracts.copy(deep=True) 

    # Apply the helper function row-by-row to calculate the Compliance Flag
    df['Compliance Flag'] = df.apply(determine_compliance_flag, axis=1)
    
    return df

def aggregate_vendor_health(df_contracts, df_vendors):
    """
    Aggregates contract-level data up to the vendor level.
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

def format_metric(value, prefix='$', suffix=''):
    """Helper for displaying metric changes with clean formatting."""
    if value >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.1f}M{suffix}"
    return f"{prefix}{value:,}{suffix}"


def run_vendorsense_app():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="VendorSense Dashboard")

    st.title("🛡️ VendorSense: Proactive Risk & Compliance Monitoring")
    st.markdown("A demonstration of Low-Code/No-Code logic translated to a Python Streamlit dashboard.")
    st.divider()

    # Load and process data
    df_contracts, df_vendors = load_data()
    
    # Check if data loaded successfully before proceeding
    if df_contracts.empty or df_vendors.empty:
        return

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
        st.metric("Portfolio Value", format_metric(total_value))
    with col3:
        st.metric("High Risk Contracts", f"{high_risk_contracts:,}", delta=f"{high_risk_contracts / total_contracts * 100:.1f}% of total", delta_color="inverse")
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
        
        # --- NEW CHART: Delays (Line Graph) ---
        st.subheader("Project Time Delays (Days Behind Schedule)")
        st.markdown("This line chart visualizes the delay magnitude for all contracts that are currently running behind schedule.")

        # Filter for contracts that are behind schedule (Days from Deadline > 0)
        df_delays = df_contracts[df_contracts['Days from Deadline'] > 0].copy()

        if not df_delays.empty:
            # Sort by delay magnitude for better visual flow
            df_delays = df_delays.sort_values(by='Days from Deadline', ascending=False)
            
            # Use Contract Name for the index and plot the delay
            df_chart = df_delays[['Contract Name', 'Days from Deadline']]

            st.line_chart(
                df_chart.set_index('Contract Name')['Days from Deadline'],
                color="#f97316", # Orange color for warning
                use_container_width=True,
                height=350
            )
            st.caption(f"Showing {len(df_delays)} contracts currently behind schedule.")
        else:
            st.success("🎉 All projects are currently on schedule or ahead of schedule!")
        
        st.divider()
        st.markdown("Filter to view contracts requiring immediate attention due to Risk or Compliance issues.")


        # Sidebar Filters
        st.sidebar.header("Operational Filters")
        risk_filter = st.sidebar.multiselect(
            "Filter by Risk Score",
            options=df_contracts['Risk Score'].unique(),
            default=['HIGH RISK', 'MEDIUM RISK']
        )
        # Ensure we only show flags, not PASS
        compliance_options = [flag for flag in df_contracts['Compliance Flag'].unique() if flag.startswith('FLAGGED')]
        compliance_filter = st.sidebar.multiselect(
            "Filter by Compliance Flag",
            options=compliance_options,
            default=compliance_options
        )

        # Apply Filters
        df_filtered = df_contracts[df_contracts['Risk Score'].isin(risk_filter)]
        df_filtered = df_filtered[df_filtered['Compliance Flag'].isin(compliance_filter)]

        st.info(f"Showing **{len(df_filtered)}** contracts matching the selected filters.")

        # Display Filtered Contracts
        st.dataframe(
            df_filtered[[
                'Contract ID', 'Contract Name', 'Vendor Name', 'Contract Value',
                'Risk Score', 'Compliance Flag', 'Cost Overrun %', 'Days from Deadline'
            ]].style.map(
                lambda x: 'background-color: #f84e4e' if x == 'HIGH RISK' else 'background-color: #fdaf78' if x == 'MEDIUM RISK' else '',
                subset=['Risk Score']
            ).format({
                'Contract Value': '${:,.0f}',
                'Cost Overrun %': '{:.1%}',
                'Days from Deadline': lambda x: f'{int(x)} days late' if x > 0 else f'{int(abs(x))} days early',
            }),
            use_container_width=True,
            hide_index=True
        )


# Run the application
if __name__ == "__main__":
    run_vendorsense_app()