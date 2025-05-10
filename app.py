# -- Streamlit App --

import streamlit as st
import pulp as p
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io # Needed for downloading dataframe as excel
import re # Import regex module for parsing variable names

# Suppress warnings for cleaner output in the app
warnings.filterwarnings("ignore")

# -- App Title and Description --
st.title("Material Requirement Optimization")
st.write("""
This application runs a Linear Programming model to optimize additional material ordering due to change in the requirement considering Bill of Materials (BOM), Demand, Inventory, Transit Times, and Penalties.
Upload your data files below and adjust the optimization coefficients using the sliders in the sidebar.
""")

# -- Sidebar for File Uploads and Coefficients --
st.sidebar.header("Upload Data Files")

# File Uploaders
bom_file = st.sidebar.file_uploader("Upload Bill Of Material (BOM) File (CSV)", type=['csv'], key="bom_uploader")
demand_file = st.sidebar.file_uploader("Upload Demand File (CSV)", type=['csv'], key="demand_uploader")
inventory_file = st.sidebar.file_uploader("Upload Incoming Inventory File (CSV)", type=['csv'], key="inventory_uploader")
onhand_file = st.sidebar.file_uploader("Upload On Hand/Capacity File (CSV)", type=['csv'], key="onhand_uploader")
transit_file = st.sidebar.file_uploader("Upload Transit Time File (CSV)", type=['csv'], key="transit_uploader")
penalty_file = st.sidebar.file_uploader("Upload Penalty File (CSV)", type=['csv'], key="penalty_file")

st.sidebar.header("Coefficient Inputs")

# Sliders for Coefficients
capacity_theta = st.sidebar.slider("Capacity Shortfall Penalty (Theta)", min_value=0, max_value=200, value=80, step=1)
demand_theta = st.sidebar.slider("Demand Shortfall Penalty (Theta)", min_value=0, max_value=200, value=100, step=1)
inventory_theta = st.sidebar.slider("Inventory Shortfall Penalty (Theta)", min_value=0, max_value=200, value=20, step=1)
DEM_FLUC_LOW_PTHETA = st.sidebar.slider("Demand Fluctuation Low Penalty (Theta)", min_value=0, max_value=200, value=50, step=1)
DEM_FLUC_HIG_PTHETA = st.sidebar.slider("Demand Fluctuation High Penalty (Theta)", min_value=0, max_value=200, value=50, step=1)

# Using number_input for bounds
DEM_FLUC_LOW_LB = st.sidebar.number_input("Demand Fluctuation Lower Bound (Ratio)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
DEM_FLUC_HIG_UB = st.sidebar.number_input("Demand Fluctuation Upper Bound (Ratio)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)


# -- Optimization and Results Section --

# Button to trigger the optimization
if st.button("Run Optimization"):
    # Check if all files are uploaded before proceeding
    if not all([bom_file, demand_file, inventory_file, onhand_file, transit_file, penalty_file]):
        st.warning("Please upload all required CSV files to run the optimization.")
        st.stop() # Stop execution if files are missing

    st.write("Running optimization...") # Keep minimal status message

    # --- Define coefficients *inside* the button block ---
    # This ensures they are in the local scope when the LP model is built
    capacity_theta_local = capacity_theta
    demand_theta_local = demand_theta
    inventory_theta_local = inventory_theta
    DEM_FLUC_LOW_PTHETA_local = DEM_FLUC_LOW_PTHETA
    DEM_FLUC_HIG_PTHETA_local = DEM_FLUC_HIG_PTHETA

    DEM_FLUC_LOW_LB_local = DEM_FLUC_LOW_LB
    DEM_FLUC_HIG_UB_local = DEM_FLUC_HIG_UB
    # --- End of local coefficients ---


    try:
        # -- Data Loading and Preprocessing --
        # Read in the Bill Of Material (BOM) File
        bom_data_raw = pd.read_csv(bom_file)
        try:
            bom_data = pd.melt(bom_data_raw, id_vars=['Product','Material_number'])
            bom_data['variable'] = bom_data['variable'].astype(str).str.extract(r'(\d+)').astype(int)
            bom_data = bom_data.rename(columns={"variable": "TimePeriod", "value": "bom_qty"})
            bom_data['bom_qty'] = bom_data['bom_qty'].fillna(0)
            bom_data.set_index(['Product','Material_number','TimePeriod'], inplace=True)
            bom_data.sort_index(inplace=True)
        except Exception as e:
            st.error(f"Error processing BOM data: {e}")
            st.stop()

        # Read in the demand file (Revised Demand)
        demand_data_raw = pd.read_csv(demand_file)
        try:
            demand_data = pd.melt(demand_data_raw,id_vars='Product')
            demand_data['variable'] = demand_data['variable'].astype(str).str.extract(r'(\d+)').astype(int)
            demand_data = demand_data.rename(columns={"variable": "TimePeriod", "value": "demand_qty"})
            demand_data['demand_qty'] = demand_data['demand_qty'].fillna(0)
            demand_data.set_index(['Product','TimePeriod'], inplace=True)
            demand_data.sort_index(inplace=True)
            demand_set = demand_data.index.unique()
            product_set = demand_data.index.get_level_values(0).unique()
        except Exception as e:
             st.error(f"Error processing Demand data: {e}")
             st.stop()

        # Read in the incoming inventory file (Material Arriving based on Original Demand)
        inventory_data_raw = pd.read_csv(inventory_file)
        try:
            inventory_data = pd.melt(inventory_data_raw,id_vars='Material_number')
            inventory_data['variable'] = inventory_data['variable'].astype(str).str.extract(r'(\d+)').astype(int)
            inventory_data = inventory_data.rename(columns={"variable": "TimePeriod", "value": "inventory_qty"})
            inventory_data['inventory_qty'] = inventory_data['inventory_qty'].fillna(0)
            inventory_data.set_index(['Material_number','TimePeriod'], inplace=True)
            inventory_data.sort_index(inplace=True)
        except Exception as e:
            st.error(f"Error processing Incoming Inventory data: {e}")
            st.stop()

        # Read in the onhand file
        onhand_data = pd.read_csv(onhand_file)
        try:
            onhand_data['On_hand'] = onhand_data['On_hand'].fillna(0)
            onhand_data['Supplier_capacity'] = onhand_data['Supplier_capacity'].fillna(0)
            onhand_data['Inventory_cap'] = onhand_data['Inventory_cap'].fillna(0)
            onhand_data.set_index(['Material_number'], inplace=True)
            onhand_data.sort_index(inplace=True)
            onhand_set = onhand_data.index.unique()
        except Exception as e:
            st.error(f"Error processing On Hand data: {e}")
            st.stop()

        # Read in the transit time file
        transit_data = pd.read_csv(transit_file)
        try:
            transit_data['Transit_time'] = transit_data['Transit_time'].fillna(0)
            transit_data.set_index(['Material_number','Mode'], inplace=True)
            transit_data.sort_index(inplace=True)
            transit_set = transit_data.index.unique()
        except Exception as e:
            st.error(f"Error processing Transit Time data: {e}")
            st.stop()

        # Read in the penalty file
        penalty_data = pd.read_csv(penalty_file)
        try:
            penalty_data['Weightage'] = penalty_data['Weightage'].fillna(0)
            penalty_data.set_index(['Mode'], inplace=True)
            penalty_data.sort_index(inplace=True)
            penalty_set = penalty_data.index.unique()
        except Exception as e:
            st.error(f"Error processing Penalty data: {e}")
            st.stop()


        # Determine the time periods and sets
        all_time_periods_list = []
        if 'TimePeriod' in bom_data.index.names and not bom_data.empty:
            all_time_periods_list.extend(bom_data.index.get_level_values('TimePeriod').tolist())
        if 'TimePeriod' in demand_data.index.names and not demand_data.empty:
            all_time_periods_list.extend(demand_data.index.get_level_values('TimePeriod').tolist())
        if 'TimePeriod' in inventory_data.index.names and not inventory_data.empty:
            all_time_periods_list.extend(inventory_data.index.get_level_values('TimePeriod').tolist())

        if not all_time_periods_list:
            st.error("Could not extract any time periods from the uploaded data.")
            st.stop()
        unique_all_time_periods = sorted(list(set(all_time_periods_list)))
        if not unique_all_time_periods:
            st.error("No unique time periods found in the data.")
            st.stop()
        max_time = max(unique_all_time_periods)
        time_periods = list(range(1, max_time + 1))
        if not time_periods:
             st.error("Failed to generate a list of time periods (1 to max_time).")
             st.stop()

        material_set = inventory_data.index.get_level_values('Material_number').unique().tolist() if 'Material_number' in inventory_data.index.names and not inventory_data.empty else []
        mode_set = penalty_data.index.unique().tolist() if 'Mode' in penalty_data.index.names and not penalty_data.empty else []

        mat_time_set = [(j, t) for j in material_set for t in time_periods]
        mat_time_mode_set = [(j, t, m) for j in material_set for t in time_periods for m in mode_set]

        if not mat_time_set or not mat_time_mode_set:
             st.error("Could not determine material, time period, or mode sets from uploaded data.")
             st.stop()

        st.write("Data preprocessing complete. Defining and solving LP model...")

        # --- Define Helper Functions *inside* the button block ---
        # This ensures they are in the local scope when called by the LP constraints
        def get_demand_qty(prod, mat, time):
             try:
                 total_demand = 0
                 # Check if index levels exist and product_set is not empty
                 if 'Product' in demand_data.index.names and 'Material_number' in bom_data.index.names and not product_set.empty:
                      for p_item in product_set:
                           # Check if keys exist before accessing .loc
                           if (p_item, time) in demand_data.index and (p_item, mat, time) in bom_data.index:
                                total_demand += demand_data.loc[(p_item, time),'demand_qty'] * bom_data.loc[(p_item, mat, time),'bom_qty']
                 return total_demand
             except KeyError:
                 return 0

        def get_inventory_qty(mat, time):
            try:
                # Check if index levels exist before accessing .loc
                if 'Material_number' in inventory_data.index.names and 'TimePeriod' in inventory_data.index.names:
                    return inventory_data.loc[(mat, time), 'inventory_qty']
                return 0 # Return 0 if index levels are missing
            except KeyError:
                return 0

        def get_onhand(mat):
             try:
                  # Check if index level exists before accessing .loc
                  if 'Material_number' in onhand_data.index.names:
                       return onhand_data.loc[mat, 'On_hand']
                  return 0 # Return 0 if index level is missing
             except KeyError:
                  return 0

        def get_inventory_cap(mat):
             try:
                  # Check if index level exists before accessing .loc
                  if 'Material_number' in onhand_data.index.names:
                       return onhand_data.loc[mat, 'Inventory_cap']
                  return 1e9 # Default large capacity if index level is missing
             except KeyError:
                  return 1e9

        def get_supplier_capacity(mat):
             try:
                  # Check if index level exists before accessing .loc
                  if 'Material_number' in onhand_data.index.names:
                       return onhand_data.loc[mat, 'Supplier_capacity']
                  return 1e9 # Default large capacity if index level is missing
             except KeyError:
                  return 1e9

        def get_transit_time(mat, mode):
             try:
                  # Check if index levels exist before accessing .loc
                  if 'Material_number' in transit_data.index.names and 'Mode' in transit_data.index.names:
                       return transit_data.loc[(mat, mode), 'Transit_time']
                  return 0 # Default 0 if index levels are missing
             except KeyError:
                  return 0
        # --- End of Helper Functions ---


        # -- Initialize Linear Programming Model --
        model = p.LpProblem("Material_Stability", p.LpMinimize)

        # -- Declare the Variables --
        INVENTORY = p.LpVariable.dicts("INVENTORY_", mat_time_set, lowBound=0, upBound=None)
        AAQ_QTY = p.LpVariable.dicts("AAQ_QTY_", mat_time_mode_set, lowBound=0, upBound=None)
        ORDER_QTY = p.LpVariable.dicts("ORDER_QTY_", mat_time_mode_set, lowBound=0, upBound=None)
        SUP_CAP_SHORTFALL = p.LpVariable.dicts("SUP_CAP_SHORTFALL_", mat_time_set, lowBound=0, upBound=None)
        SUP_CAP_EXCESS = p.LpVariable.dicts("SUP_CAP_EXCESS_", mat_time_set, lowBound=0, upBound=None)
        DEM_SHORTFALL = p.LpVariable.dicts("DEM_SHORTFALL_", mat_time_set, lowBound=0, upBound=None)
        REL_FLUC_LB_BELOW = p.LpVariable.dicts("REL_FLUC_LB_BELOW_", mat_time_set, lowBound=0, upBound=None)
        REL_FLUC_LB_ABOVE = p.LpVariable.dicts("REL_FLUC_LB_ABOVE_", mat_time_set, lowBound=0, upBound=None)
        REL_FLUC_UB_ABOVE = p.LpVariable.dicts("REL_FLUC_UB_ABOVE_", mat_time_set, lowBound=0, upBound=None)
        REL_FLUC_UB_BELOW = p.LpVariable.dicts("REL_FLUC_UB_BELOW_", mat_time_set, lowBound=0, upBound=None)
        INV_CAP_SHORTFALL = p.LpVariable.dicts("INV_CAP_SHORTFALL_", mat_time_set, lowBound=0, upBound=None)
        INV_CAP_EXCESS = p.LpVariable.dicts("INV_CAP_EXCESS_", mat_time_set, lowBound=0, upBound=None)

        # -- Create objective - Use the local coefficients defined inside the button block --
        model += p.lpSum(
            ((SUP_CAP_SHORTFALL[j,t] * capacity_theta_local) +
             (DEM_SHORTFALL[j,t] * demand_theta_local) +
             (REL_FLUC_LB_BELOW[j,t] * DEM_FLUC_LOW_PTHETA_local) +
             (REL_FLUC_UB_ABOVE[j,t] * DEM_FLUC_HIG_PTHETA_local) +
             (INV_CAP_SHORTFALL[j,t] * inventory_theta_local))
            for (j,t) in mat_time_set
        ) + p.lpSum(ORDER_QTY.get((j,t,m), 0) * penalty_data.loc[m,'Weightage'] for (j,t,m) in mat_time_mode_set if m in penalty_data.index)


        # -- Constraints --
        # Using the helper functions defined just above

        # Non negative Inventory constraint
        for (j,t) in mat_time_set:
            model  += INVENTORY[j,t] >= 0, f"NonNegative_Inventory_{j}_{t}"

        # Closing Inventory Constraint
        for (j,t) in mat_time_set:
             incoming_arr = get_inventory_qty(j, t) + p.lpSum(AAQ_QTY[(j, t, m1)] for m1 in mode_set if (j, t, m1) in mat_time_mode_set)
             required_qty = get_demand_qty(None, j, t)
             opening_inv = get_onhand(j) if t == 1 else INVENTORY.get((j, t-1), 0)

             model += INVENTORY[j,t] >= opening_inv + incoming_arr - required_qty, f"Closing_Inventory_{j}_{t}"

        # Inventory Limit Constraint
        for (j,t) in mat_time_set:
             inv_cap = get_inventory_cap(j)
             model += INVENTORY[j,t] - INV_CAP_EXCESS[j,t] + INV_CAP_SHORTFALL[j,t] == inv_cap, f"Inventory_Limit_{j}_{t}"

        # Demand Fulfillment Constraint
        for (j,t) in mat_time_set:
             incoming_arr = get_inventory_qty(j, t) + p.lpSum(AAQ_QTY[(j, t, m1)] for m1 in mode_set if (j, t, m1) in mat_time_mode_set)
             required_qty = get_demand_qty(None, j, t)
             opening_inv = get_onhand(j) if t == 1 else INVENTORY.get((j, t-1), 0)
             model += opening_inv + incoming_arr - INVENTORY[j,t] == required_qty - DEM_SHORTFALL[j,t], f"Demand_Fulfillment_{j}_{t}"

        # Additional Material Order Constraint (AAQ_QTY = ORDER_QTY with Transit Time)
        for (j,t,d) in mat_time_mode_set:
            transit_t = int(get_transit_time(j, d))
            if t - transit_t >= 1:
                source_time = t - transit_t
                if source_time in time_periods:
                     model += AAQ_QTY[j,t,d] == ORDER_QTY.get((j, source_time, d), 0), f"AAQ_Transit_{j}_{t}_{d}"
                else:
                     model += AAQ_QTY[j,t,d] == 0, f"AAQ_NoOrderOutsideHorizon_{j}_{t}_{d}"
            else:
                model += AAQ_QTY[j,t,d] == 0, f"AAQ_NoOrderBeforeT1_{j}_{t}_{d}"

        # Supplier Capacity Constraint
        for (j,t) in mat_time_set:
             sup_cap = get_supplier_capacity(j)
             incoming_inv_t = get_inventory_qty(j, t)
             sum_orders_t = p.lpSum(ORDER_QTY.get((j, t, m1), 0) for m1 in mode_set if (j, t, m1) in mat_time_mode_set)
             model += sum_orders_t == sup_cap + SUP_CAP_SHORTFALL[j,t] - SUP_CAP_EXCESS[j,t] - incoming_inv_t, f"Supplier_Capacity_{j}_{t}"

        # Material Upper Bound Fluctuation Constraint
        for (j,t) in mat_time_set:
             original_supply_t = get_inventory_qty(j, t)
             sum_orders_t = p.lpSum(ORDER_QTY.get((j, t, m1), 0) for m1 in mode_set if (j, t, m1) in mat_time_mode_set)
             model += sum_orders_t == DEM_FLUC_HIG_UB_local * original_supply_t + REL_FLUC_UB_ABOVE[j,t] - REL_FLUC_UB_BELOW[j,t], f"Fluctuation_UB_{j}_{t}"

        # Material Lower Bound Fluctuation Constraint
        for (j,t) in mat_time_set:
             original_supply_t = get_inventory_qty(j, t)
             sum_orders_t = p.lpSum(ORDER_QTY.get((j, t, m1), 0) for m1 in mode_set if (j, t, m1) in mat_time_mode_set)
             model += sum_orders_t == DEM_FLUC_LOW_LB_local * original_supply_t - REL_FLUC_LB_BELOW[j,t] + REL_FLUC_LB_ABOVE[j,t], f"Fluctuation_LB_{j}_{t}"


        # -- Solve Model --
        st.write("Solving LP model...")
        solver = p.PULP_CBC_CMD(msg=0)
        model.solve(solver)

        # -- Display Solve Status --
        st.subheader("Optimization Status")
        status = p.LpStatus[model.status]
        st.write(f"Solver Status: **{status}**")

        if status != "Optimal":
            st.warning("The optimization did not find an optimal solution. Please check your data and constraints.")


        st.write(f"Optimal Objective Value: **{model.objective.value():,.2f}**")

        # -- Filtering Non-Zero Variables and Creating DataFrame --
        non_zero_vars = {var.name: var.varValue for var in model.variables()}

        df1 = pd.DataFrame.from_dict(non_zero_vars, orient='index', columns=['Value'])
        df1.index.name = 'Variable Name'
        df1 = df1.reset_index()

        # --- START OF VARIABLE PARSING (Using original regex structure) ---
        # Use the user's original regex pattern
        pattern_original_structure = r"(\w+)__\((.*?),_(\d+)(?:,_(.*))?\)"

        parsed_data = df1['Variable Name'].str.extract(pattern_original_structure)

        if parsed_data is None or parsed_data.shape[1] != 4:
             st.warning("Variable name parsing failed using original pattern. Cannot generate detailed results or plots.")
             # Fallback structure for Excel download if parsing fails
             df = df1.rename(columns={'Variable Name': 'Variable', 'Value': 'Value'})
             df['Mat_no'] = 'N/A'
             df['Week'] = 0
             df['Mode'] = 'N/A'
             # Set flag to skip plotting
             parsing_successful = False
        else:
             parsed_data.columns = ['Variable', 'Mat_no_Raw', 'Week_Str', 'Mode_Raw']
             df = pd.concat([df1[['Variable Name', 'Value']], parsed_data], axis=1)
             df['Mat_no'] = df['Mat_no_Raw'].astype(str).str.strip().str.strip("'\"")
             df['Week'] = pd.to_numeric(df['Week_Str'], errors='coerce').fillna(0).astype(int)
             df['Mode'] = df['Mode_Raw'].astype(str).str.strip().str.strip("'\"").replace('nan', 'N/A').fillna('N/A')
             df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0)
             df = df.drop(columns=['Variable Name', 'Mat_no_Raw', 'Week_Str', 'Mode_Raw'])
             df = df.loc[:, ['Variable', 'Mat_no', 'Week', 'Mode', 'Value']]
             parsing_successful = True # Set flag to allow plotting

        # --- END OF VARIABLE PARSING ---

        # Sort the dataframe
        df = df.sort_values(by=['Variable', 'Mat_no', 'Mode', 'Week'])

        # Create the pivot table for Excel output and display
        st.subheader("Optimization Results Table")
        sorted_pivot_table = pd.DataFrame() # Initialize empty
        if parsing_successful:
             pivot_index_cols = ['Variable', 'Mat_no', 'Mode']
             pivot_index_cols = [col for col in pivot_index_cols if col in df.columns]

             if 'Week' in df.columns and not df.empty and pivot_index_cols and 'Value' in df.columns:
                  pivotable_vars = ['AAQ_QTY', 'ORDER_QTY', 'SUP_CAP_SHORTFALL', 'SUP_CAP_EXCESS',
                               'DEM_SHORTFALL', 'REL_FLUC_LB_BELOW', 'REL_FLUC_LB_ABOVE',
                               'REL_FLUC_UB_ABOVE', 'REL_FLUC_UB_BELOW', 'INV_CAP_SHORTFALL',
                               'INV_CAP_EXCESS', 'INVENTORY']
                  pivotable_df = df[df['Variable'].isin(pivotable_vars)].copy()
                  pivot_index_cols_filtered = ['Variable', 'Mat_no', 'Mode']
                  pivot_index_cols_filtered = [col for col in pivot_index_cols_filtered if col in pivotable_df.columns]

                  if pivot_index_cols_filtered and 'Week' in pivotable_df.columns and 'Value' in pivotable_df.columns:
                       try:
                           sorted_pivot_table = pivotable_df.pivot_table(columns='Week', index=pivot_index_cols_filtered, values='Value', fill_value=0)
                           if all(isinstance(col, (int, np.integer)) or (isinstance(col, str) and col.isdigit()) for col in sorted_pivot_table.columns):
                                sorted_weeks = sorted(sorted_pivot_table.columns, key=lambda x: int(x))
                                sorted_pivot_table = sorted_pivot_table.reindex(columns=sorted_weeks)
                           else:
                                st.warning("Could not sort week columns numerically in pivot table. Displaying as is.")

                           # Display the pivot table
                           st.dataframe(sorted_pivot_table)

                       except Exception as e:
                           st.error(f"Error creating or displaying pivot table: {e}")
                           st.exception(e)
                           sorted_pivot_table = pd.DataFrame() # Empty on error
                  else:
                       st.warning("Could not create pivot table from filtered data. Check data columns.")
             else:
                  st.warning("Could not create pivot table. Check data columns or if DataFrame is empty.")


        # -- Provide Excel Download Button (always show if sorted_pivot_table created) --
        if not sorted_pivot_table.empty:
             excel_buffer = io.BytesIO()
             try:
                 sorted_pivot_table.to_excel(excel_buffer, index=True)
                 st.download_button(
                     label="Download Results as Excel",
                     data=excel_buffer.getvalue(),
                     file_name="LP_output.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
             except Exception as e:
                  st.error(f"Error generating Excel file: {e}")
                  st.exception(e)


        # -- Plotting Section --
        st.subheader("Visualization of Results")

        # Only attempt plotting if parsing was successful and df has the necessary columns
        if parsing_successful and all(col in df.columns for col in ['Variable', 'Mat_no', 'Week', 'Value']):

            # Data for Original Requirement Plot
            demand_org = pd.DataFrame() # Initialize empty
            if not inventory_data.empty and 'Material_number' in inventory_data.index.names and 'TimePeriod' in inventory_data.index.names:
                try:
                    demand_org = inventory_data.reset_index()
                    demand_org.columns = ['Mat_no','Week','Value']
                    demand_org['Week'] = demand_org['Week'].astype(int)
                except Exception as e:
                     st.warning(f"Could not prepare Original Demand data for plotting: {e}")
                     st.exception(e)


            # Data for Revised Requirement Plot
            demand_revised = pd.DataFrame() # Initialize empty
            if not demand_data.empty and not bom_data.empty:
                 # Use .empty check for Index/MultiIndex in boolean context
                 if not demand_set.empty and not product_set.empty and 'Material_number' in bom_data.index.names and 'TimePeriod' in demand_data.index.names:
                      demand_revised_list = []
                      try:
                           # Ensure product_set is iterable and contains valid keys for demand_data/bom_data
                           if isinstance(product_set, pd.Index) and not product_set.empty:
                                for (p, t) in demand_set:
                                     try:
                                          if (p, t) in demand_data.index and p in bom_data.index and 'Material_number' in bom_data.loc[p].index.names:
                                               demand_qty = demand_data.loc[(p, t), 'demand_qty']
                                               materials_in_bom = bom_data.loc[p].index.get_level_values('Material_number').unique()
                                               for j in materials_in_bom:
                                                    if (p, j, t) in bom_data.index:
                                                         bom_qty = bom_data.loc[(p, j, t), 'bom_qty']
                                                         demand_revised_list.append({'Mat_no': j, 'Week': t, 'Value': demand_qty * bom_qty})
                                     except KeyError as e:
                                          pass # Skip if key is missing during iteration

                           demand_revised = pd.DataFrame(demand_revised_list)

                           if not demand_revised.empty:
                                try:
                                    demand_revised = demand_revised.groupby(['Mat_no','Week'])['Value'].sum().reset_index()
                                    demand_revised['Week'] = demand_revised['Week'].astype(int)
                                except Exception as e:
                                    st.warning(f"Error grouping revised demand data for plotting: {e}")
                                    st.exception(e)
                                    demand_revised = pd.DataFrame() # Empty if grouping fails
                      except Exception as e:
                           st.warning(f"Error during revised demand list creation or processing: {e}")
                           st.exception(e)
                           demand_revised = pd.DataFrame() # Ensure it's empty on error
                 # Removed st.info if empty revised demand
            # Removed st.info if skipping revised demand calculation because Demand or BOM empty


            # 1. Plot Original vs Revised Material Requirement
            st.write("#### Material Requirements: Original vs. Revised")
            if not demand_org.empty and not demand_revised.empty:
                try:
                    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

                    sns.lineplot(data=demand_org, x='Week', y='Value', hue='Mat_no', marker='o', ax=axes1[0])
                    axes1[0].set_title('Original Material Requirement (Incoming Inventory)')
                    axes1[0].set_ylabel('Quantity')
                    axes1[0].set_xlabel('Week')

                    sns.lineplot(data=demand_revised, x='Week', y='Value', hue='Mat_no', marker='o', ax=axes1[1])
                    axes1[1].set_title('Revised Material Requirement (Calculated)')
                    axes1[1].set_ylabel('Quantity')
                    axes1[1].set_xlabel('Week')

                    all_req_weeks = sorted(list(set(demand_org['Week'].unique()) | set(demand_revised['Week'].unique())))
                    if all_req_weeks:
                         for ax in axes1:
                             ax.set_xticks(all_req_weeks)
                             if len(all_req_weeks) > 10:
                                  ax.tick_params(axis='x', rotation=45)
                             ax.grid(axis='y', linestyle='--', alpha=0.7)

                    handles1, labels1 = [], []
                    for ax in axes1:
                         if ax.get_legend():
                              h, l = ax.get_legend_handles_labels()
                              handles1.extend(h)
                              labels1.extend(l)
                    unique_labels = {}
                    for handle, label in zip(handles1, labels1):
                         unique_labels[label] = handle
                    handles1 = list(unique_labels.values())
                    labels1 = list(unique_labels.keys())

                    for ax in axes1:
                         if ax.get_legend():
                              ax.get_legend().remove()

                    if handles1 and labels1:
                         fig1.legend(handles=handles1, labels=labels1, title='Mat_no', bbox_to_anchor=(1.02, 1), loc='upper left')

                    fig1.suptitle('Material Requirements: Original vs. Revised', y=1.05, fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 0.95, 0.98])
                    st.pyplot(fig1)
                    plt.close(fig1)
                except Exception as e:
                     st.error(f"Error generating Material Requirements plot: {e}")
                     st.exception(e)
                     plt.close('all')
            else:
                pass


            # 2. Plot ORDER_QTY
            st.write("#### ORDER_QTY over Weeks by Material Number and Mode")
            df_ORDER_QTY = df[df['Variable'] == 'ORDER_QTY'].copy()
            if not df_ORDER_QTY.empty:
                try:
                    g1 = sns.relplot(
                        data=df_ORDER_QTY, x='Week', y='Value', hue='Mat_no', col='Mode',
                        kind='line', marker='o', height=5, aspect=1.2, facet_kws={'sharey': False, 'sharex': True}
                    )
                    g1.fig.suptitle('ORDER_QTY over Weeks by Material Number and Mode', y=1.02)
                    g1.set_titles("Mode: {col_name}")
                    order_weeks = sorted(df_ORDER_QTY['Week'].unique())
                    if order_weeks:
                         for ax in g1.axes.flat:
                             ax.set_xticks(order_weeks)
                             if len(order_weeks) > 10:
                                  ax.tick_params(axis='x', rotation=45)
                             ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout(rect=[0, 0.03, 0.93, 0.95])
                    st.pyplot(g1.fig)
                    plt.close(g1.fig)
                except Exception as e:
                     st.error(f"Error generating ORDER_QTY plot: {e}")
                     st.exception(e)
                     plt.close('all')


            # 3. Plot AAQ_QTY
            st.write("#### AAQ_QTY over Weeks by Material Number and Mode")
            df_AAQ_QTY = df[df['Variable'] == 'AAQ_QTY'].copy()
            if not df_AAQ_QTY.empty:
                try:
                    g2 = sns.relplot(
                        data=df_AAQ_QTY, x='Week', y='Value', hue='Mat_no', col='Mode',
                        kind='line', marker='o', height=5, aspect=1.2, facet_kws={'sharey': False, 'sharex': True}
                    )
                    g2.fig.suptitle('AAQ_QTY over Weeks by Material Number and Mode', y=1.02)
                    g2.set_titles("Mode: {col_name}")
                    aaq_weeks = sorted(df_AAQ_QTY['Week'].unique())
                    if aaq_weeks:
                         for ax in g2.axes.flat:
                             ax.set_xticks(aaq_weeks)
                             if len(aaq_weeks) > 10:
                                  ax.tick_params(axis='x', rotation=45)
                             ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout(rect=[0, 0.03, 0.93, 0.95])
                    st.pyplot(g2.fig)
                    plt.close(g2.fig)
                except Exception as e:
                     st.error(f"Error generating AAQ_QTY plot: {e}")
                     st.exception(e)
                     plt.close('all')


            # 4. Plot DEM_SHORTFALL
            st.write("#### Demand Shortfall over Weeks by Material Number")
            df_DEM_SHORTFALL = df[df['Variable'] == 'DEM_SHORTFALL'].copy()
            if not df_DEM_SHORTFALL.empty:
                try:
                    g3 = sns.relplot(
                        data=df_DEM_SHORTFALL, x='Week', y='Value', hue='Mat_no',
                        kind='line', marker='o', height=5, aspect=1.5
                    )
                    g3.fig.suptitle('Demand Shortfall over Weeks by Material Number', y=1.02)
                    shortfall_weeks = sorted(df_DEM_SHORTFALL['Week'].unique())
                    if shortfall_weeks:
                         for ax in g3.axes.flat:
                             ax.set_xticks(shortfall_weeks)
                             if len(shortfall_weeks) > 10:
                                  ax.tick_params(axis='x', rotation=45)
                             ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
                    st.pyplot(g3.fig)
                    plt.close(g3.fig)
                except Exception as e:
                     st.error(f"Error generating Demand Shortfall plot: {e}")
                     st.exception(e)
                     plt.close('all')


            # 5. Plot INVENTORY vs INV_CAP_EXCESS (Inventory Limits)
            st.write("#### Material Inventory vs Inventory Limits")
            df_INVENTORY = df[df['Variable'] == 'INVENTORY'].copy()
            df_INV_CAP_EXCESS = df[df['Variable'] == 'INV_CAP_EXCESS'].copy()
            if not df_INVENTORY.empty or not df_INV_CAP_EXCESS.empty:
                try:
                    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                    if not df_INVENTORY.empty:
                         sns.lineplot(data=df_INVENTORY, x='Week', y='Value', hue='Mat_no', marker='o', ax=axes4[0])
                         axes4[0].set_title('Inventory Level')
                         axes4[0].set_ylabel('Quantity')
                         axes4[0].set_xlabel('Week')
                         inv_weeks_inv = df_INVENTORY['Week'].unique()
                         h_inv, l_inv = axes4[0].get_legend_handles_labels()
                    else:
                         axes4[0].set_title('Inventory Level (No Data)')
                         inv_weeks_inv = []
                         h_inv, l_inv = [], []
                         axes4[0].legend([],[], frameon=False)

                    if not df_INV_CAP_EXCESS.empty:
                         sns.lineplot(data=df_INV_CAP_EXCESS, x='Week', y='Value', hue='Mat_no', marker='o', ax=axes4[1])
                         axes4[1].set_title('Inventory Limit Excess')
                         axes4[1].set_ylabel('Quantity')
                         axes4[1].set_xlabel('Week')
                         inv_weeks_excess = df_INV_CAP_EXCESS['Week'].unique()
                         h_excess, l_excess = axes4[1].get_legend_handles_labels()
                    else:
                         axes4[1].set_title('Inventory Limit Excess (No Data)')
                         inv_weeks_excess = []
                         h_excess, l_excess = [], []
                         axes4[1].legend([],[], frameon=False)

                    inv_weeks = sorted(list(set(inv_weeks_inv) | set(inv_weeks_excess)))
                    if inv_weeks:
                         for ax in axes4:
                             ax.set_xticks(inv_weeks)
                             if len(inv_weeks) > 10:
                                  ax.tick_params(axis='x', rotation=45)
                             ax.grid(axis='y', linestyle='--', alpha=0.7)

                    handles4 = h_inv + h_excess
                    labels4 = l_inv + l_excess
                    unique_labels = {}
                    for handle, label in zip(handles4, labels4):
                         unique_labels[label] = handle
                    handles4 = list(unique_labels.values())
                    labels4 = list(unique_labels.keys())

                    if handles4 and labels4:
                         fig4.legend(handles=handles4, labels=labels4, title='Mat_no', bbox_to_anchor=(1.02, 1), loc='upper left')

                    fig4.suptitle('Material Inventory vs Inventory Limit Excess', y=1.05, fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 0.95, 0.98])
                    st.pyplot(fig4)
                    plt.close(fig4)
                except Exception as e:
                     st.error(f"Error generating Inventory plots: {e}")
                     st.exception(e)
                     plt.close('all')


            # 6. Plot SUP_CAP_EXCESS vs SUP_CAP_SHORTFALL (Supplier Capacity)
            st.write("#### Supplier Capacity: Available vs Shortfall")
            df_SUP_CAP_EXCESS = df[df['Variable'] == 'SUP_CAP_EXCESS'].copy()
            df_SUP_CAP_SHORTFALL = df[df['Variable'] == 'SUP_CAP_SHORTFALL'].copy()
            if not df_SUP_CAP_EXCESS.empty or not df_SUP_CAP_SHORTFALL.empty:
                try:
                    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                    if not df_SUP_CAP_EXCESS.empty:
                         sns.lineplot(data=df_SUP_CAP_EXCESS, x='Week', y='Value', hue='Mat_no', marker='o', ax=axes5[0])
                         axes5[0].set_title('Supplier Capacity Excess')
                         axes5[0].set_ylabel('Quantity')
                         axes5[0].set_xlabel('Week')
                         sup_weeks_excess = df_SUP_CAP_EXCESS['Week'].unique()
                         h_excess, l_excess = axes5[0].get_legend_handles_labels()
                    else:
                         axes5[0].set_title('Supplier Capacity Excess (No Data)')
                         sup_weeks_excess = []
                         h_excess, l_excess = [], []
                         axes5[0].legend([],[], frameon=False)

                    if not df_SUP_CAP_SHORTFALL.empty:
                         sns.lineplot(data=df_SUP_CAP_SHORTFALL, x='Week', y='Value', hue='Mat_no', marker='o', ax=axes5[1])
                         axes5[1].set_title('Supplier Capacity Shortfall')
                         axes5[1].set_ylabel('Quantity')
                         axes5[1].set_xlabel('Week')
                         sup_weeks_shortfall = df_SUP_CAP_SHORTFALL['Week'].unique()
                         h_shortfall, l_shortfall = axes5[1].get_legend_handles_labels()
                    else:
                         axes5[1].set_title('Supplier Capacity Shortfall (No Data)')
                         sup_weeks_shortfall = []
                         h_shortfall, l_shortfall = [], []
                         axes5[1].legend([],[], frameon=False)

                    sup_weeks = sorted(list(set(sup_weeks_excess) | set(sup_weeks_shortfall)))
                    if sup_weeks:
                         for ax in axes5:
                             ax.set_xticks(sup_weeks)
                             if len(sup_weeks) > 10:
                                  ax.tick_params(axis='x', rotation=45)
                             ax.grid(axis='y', linestyle='--', alpha=0.7)

                    handles5 = h_excess + h_shortfall
                    labels5 = l_excess + l_shortfall
                    unique_labels = {}
                    for handle, label in zip(handles5, labels5):
                         unique_labels[label] = handle
                    handles5 = list(unique_labels.values())
                    labels5 = list(unique_labels.keys())

                    if handles5 and labels5:
                         fig5.legend(handles=handles5, labels=labels5, title='Mat_no', bbox_to_anchor=(1.02, 1), loc='upper left')

                    fig5.suptitle('Supplier Capacity: Excess vs Shortfall', y=1.05, fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 0.95, 0.98])
                    st.pyplot(fig5)
                    plt.close(fig5)
                except Exception as e:
                     st.error(f"Error generating Supplier Capacity plots: {e}")
                     st.exception(e)
                     plt.close('all')

        # Removed else block for main plotting section - errors will be caught individually


    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        st.exception(e)


# Helper functions definition (moved inside button block)

# -- Optional: Instructions or example data info --
st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Upload all six required CSV files using the browse buttons.
2. Adjust the penalty and fluctuation coefficients using the sliders/inputs.
3. Click the "Run Optimization" button.
4. The results table download button and plots will appear below.
5. You can download the results table as an Excel file.

**File Format Notes:**
Ensure your CSV files have the exact column names expected by the script (e.g., 'Product', 'Material_number', 'On_hand', 'TimePeriod', 'Mode', 'Weightage', and columns representing time periods like '1', '2', '3', etc. for melted data).
""")
