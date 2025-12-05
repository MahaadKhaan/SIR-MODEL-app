import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Project Patient Zero", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Dark Theme Background */
    .stApp { background-color: #050505; }

    /* Neon Red Main Title */
    .neon-red { 
        color: #ff3333; 
        text-shadow: 0px 0px 15px #ff0000; 
        font-family: 'Courier New', Courier, monospace; 
        text-transform: uppercase;
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        line-height: 1.2;
    }

    /* Neon Green Sub-Title */
    .neon-green {
        color: #39ff14;
        text-shadow: 0px 0px 15px #39ff14;
        font-family: 'Courier New', Courier, monospace;
        font-size: 25px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
    }

    /* Analysis Section Glow */
    .analysis-box {
        border: 2px solid #ff3333;
        box-shadow: 0px 0px 20px rgba(255, 0, 0, 0.4);
        padding: 20px;
        border-radius: 15px;
        background-color: #0a0a0a;
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Credits Text */
    .credits {
        font-family: 'Courier New';
        color: #888;
        text-align: right;
        font-size: 14px;
        margin-top: 50px;
    }

    /* Day Counter Style */
    .day-counter {
        font-family: 'Courier New';
        font-size: 30px;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 0px 0px 10px #ffffff;
    }

    /* Button Styling */
    .stButton>button {
        color: white;
        background-color: #ff3333;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #cc0000;
        box-shadow: 0px 0px 15px #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'started' not in st.session_state:
    st.session_state.started = False

# --- COORDINATE DATABASE ---
LOCATIONS = {
    "Pakistan": [30.3753, 69.3451, 2.5],
    "United States": [37.0902, -95.7129, 6.0],
    "China": [35.8617, 104.1954, 5.0],
    "United Kingdom": [55.3781, -3.4360, 1.0],
    "Brazil": [-14.2350, -51.9253, 5.0],
    "Russia": [61.5240, 105.3188, 8.0],
    "India": [20.5937, 78.9629, 4.0]
}


# --- MATH ENGINE (RK4) ---
def get_sir_derivatives(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - (gamma * I)
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def run_rk4_simulation(N, I0, beta, gamma, days):
    y = np.array([N - I0, I0, 0.0])
    dt = 1.0
    t_points = np.linspace(0, days, days + 1)

    S_list, I_list, R_list = [], [], []

    for t in t_points:
        S_list.append(y[0])
        I_list.append(y[1])
        R_list.append(y[2])

        k1 = np.array(get_sir_derivatives(y, t, N, beta, gamma))
        k2 = np.array(get_sir_derivatives(y + 0.5 * dt * k1, t + 0.5 * dt, N, beta, gamma))
        k3 = np.array(get_sir_derivatives(y + 0.5 * dt * k2, t + 0.5 * dt, N, beta, gamma))
        k4 = np.array(get_sir_derivatives(y + dt * k3, t + dt, N, beta, gamma))
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_points, S_list, I_list, R_list


# --- VISUAL ENGINE ---
def get_dots_for_frame(lat_c, lon_c, scale, S, I, R, N):
    MAX_VISUAL_DOTS = 1200

    count_S = int((S / N) * MAX_VISUAL_DOTS)
    count_I = int((I / N) * MAX_VISUAL_DOTS)
    count_R = int((R / N) * MAX_VISUAL_DOTS)

    lat_s = np.random.normal(lat_c, scale / 3, count_S)
    lon_s = np.random.normal(lon_c, scale / 3, count_S)

    spread_factor_I = (scale / 4) + (I / N) * (scale / 3)
    lat_i = np.random.normal(lat_c, spread_factor_I, count_I)
    lon_i = np.random.normal(lon_c, spread_factor_I, count_I)

    lat_r = np.random.normal(lat_c, scale / 3, count_R)
    lon_r = np.random.normal(lon_c, scale / 3, count_R)

    df_s = pd.DataFrame({'lat': lat_s, 'lon': lon_s, 'Type': 'Susceptible', 'Color': '#00ccff'})
    df_i = pd.DataFrame({'lat': lat_i, 'lon': lon_i, 'Type': 'Infected', 'Color': '#ff0000'})
    df_r = pd.DataFrame({'lat': lat_r, 'lon': lon_r, 'Type': 'Recovered', 'Color': '#00ff00'})

    return pd.concat([df_s, df_i, df_r])


# ============================
# PAGE 1: WELCOME SCREEN
# ============================
if not st.session_state.started:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Title Split into two lines as requested
        st.markdown('<div class="neon-red">☣️ S I R  M O D E L<br>SIMULATION</div>', unsafe_allow_html=True)
        st.markdown('<div class="neon-green">🦠 PROJECT: PATIENT ZERO</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 18px; color: #ccc; text-align: justify;">
        Welcome to the advanced biostatistical engine. This tool simulates the spread of infectious diseases across global populations using the 
        <b>Runge-Kutta 4th Order (RK4)</b> numerical method.
        <br><br>
        Scientists use this model to predict pandemics, calculate R0 scores, and visualize herd immunity thresholds.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧬 ENTER LABORATORY"):
            st.session_state.started = True
            st.rerun()

        st.markdown("""
        <div class="credits">
        Developed by <b>M. Mahad Khan</b><br>
        Numerical Computing Semester Project
        </div>
        """, unsafe_allow_html=True)

# ============================
# PAGE 2: MAIN SIMULATION APP
# ============================
else:
    # --- HEADERS ---
    # Same split title for consistency
    st.markdown('<div class="neon-red">☣️ S I R  M O D E L<br>SIMULATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-green">🦠 PROJECT: PATIENT ZERO</div>', unsafe_allow_html=True)

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("⚙️ SIMULATION PROTOCOLS")
    target = st.sidebar.selectbox("Target Region", list(LOCATIONS.keys()))

    st.sidebar.subheader("Initial Conditions")
    N = st.sidebar.number_input("Population Size (N)", value=1_000_000, step=100_000)
    I0 = st.sidebar.number_input("Initial Infected (I0)", value=50, min_value=1)

    st.sidebar.subheader("Pathogen DNA")
    beta = st.sidebar.slider("Infectivity Rate (Beta)", 0.0, 1.0, 0.45)
    gamma = st.sidebar.slider("Recovery Rate (Gamma)", 0.0, 1.0, 0.1)
    days = st.sidebar.slider("Time Horizon (Days)", 20, 100, 60)

    # --- INITIATE BUTTON ---
    if st.sidebar.button("🟥 INITIATE PATHOGEN"):

        # 1. Run Math
        t_vals, S_vals, I_vals, R_vals = run_rk4_simulation(N, I0, beta, gamma, days)

        # 2. Setup Map
        center_lat, center_lon, scale = LOCATIONS[target]

        # --- NEW: DAYS COUNTER PLACEHOLDER ---
        days_placeholder = st.empty()

        map_placeholder = st.empty()
        stats_placeholder = st.empty()

        # 3. Animation Loop
        for day in range(len(t_vals)):
            curr_S, curr_I, curr_R = S_vals[day], I_vals[day], R_vals[day]

            # UPDATE DAY COUNTER
            days_placeholder.markdown(f'<div class="day-counter">DAY: {day}</div>', unsafe_allow_html=True)

            # Generate Dots
            df_dots = get_dots_for_frame(center_lat, center_lon, scale, curr_S, curr_I, curr_R, N)

            # Draw Map
            fig = px.scatter_geo(
                df_dots, lat='lat', lon='lon', color='Type',
                projection="orthographic",
                color_discrete_map={'Susceptible': '#00ccff', 'Infected': '#ff0000', 'Recovered': '#00ff00'},
                opacity=0.7, template="plotly_dark"
            )

            fig.update_geos(
                bgcolor="black", showocean=True, oceancolor="#111111",
                showland=True, landcolor="#222222",
                showcountries=True, countrycolor="#555555",
                projection_rotation=dict(lon=center_lon, lat=center_lat)
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500, showlegend=False)

            map_placeholder.plotly_chart(fig, use_container_width=True)

            # Live Metrics
            stats_placeholder.markdown(f"""
                <div style="display: flex; justify-content: space-around; background-color: #111; padding: 10px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px;">
                    <div style="color: #00ccff; text-align: center;"><h3>🟦 SUSCEPTIBLE</h3><h2>{int(curr_S):,}</h2></div>
                    <div style="color: #ff3333; text-align: center;"><h3>🟥 INFECTED</h3><h2>{int(curr_I):,}</h2></div>
                    <div style="color: #00ff00; text-align: center;"><h3>🟩 RECOVERED</h3><h2>{int(curr_R):,}</h2></div>
                </div>
            """, unsafe_allow_html=True)

            time.sleep(0.05)

        # --- 4. POST OUTBREAK ANALYSIS (FIXED VISUAL) ---
        st.markdown("<br>", unsafe_allow_html=True)

        # Fixed: Text is now INSIDE the div so it sits inside the red box
        st.markdown("""
        <div class="analysis-box">
            <h2 style='color: white; margin:0;'>📊 POST-OUTBREAK ANALYSIS</h2>
        </div>
        """, unsafe_allow_html=True)

        # Charts
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=t_vals, y=S_vals, name='Susceptible', line=dict(color='#00ccff')))
        line_fig.add_trace(go.Scatter(x=t_vals, y=I_vals, name='Infected', line=dict(color='#ff0000', width=4)))
        line_fig.add_trace(go.Scatter(x=t_vals, y=R_vals, name='Recovered', line=dict(color='#00ff00')))
        line_fig.update_layout(template="plotly_dark", height=400, hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(line_fig, use_container_width=True)

        # SMART REMARKS
        peak_infected = max(I_vals)
        final_S = S_vals[-1]
        total_affected = N - final_S
        affected_percentage = (total_affected / N) * 100
        r0 = beta / gamma if gamma > 0 else 0

        # AI SUMMARY BOX
        st.markdown("""
        <div style="border: 1px solid #333; padding: 20px; border-radius: 10px; background-color: #0e0e0e;">
        <h3 style='color: #888;'>📝 AI SUMMARY REPORT</h3>
        """, unsafe_allow_html=True)

        if affected_percentage > 90:
            st.error(
                f"💀 **TOTAL POPULATION COLLAPSE:** The virus infected {affected_percentage:.1f}% of the population. The outbreak only stopped because there were no healthy hosts left.")
        elif r0 < 1:
            st.success(
                f"✅ **CONTAINMENT SUCCESSFUL:** The R0 score was {r0:.2f}. The virus failed to spread effectively and died out naturally.")
        else:
            st.warning(f"⚠️ **OUTBREAK PERSISTENT:** The R0 score was {r0:.2f}. The virus is spreading exponentially.")

        if peak_infected > (N * 0.4):
            st.error(
                f"🚨 **CRITICAL SYSTEM FAILURE:** Peak infection reached {int(peak_infected):,} people. Hospitals overwhelmed.")
        elif peak_infected < (N * 0.1):
            st.info(
                f"🛡️ **SYSTEM STABLE:** Peak infection was low ({int(peak_infected):,}). Healthcare systems remained within capacity.")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Static Preview
        st.info("👈 Open the Sidebar, Adjust DNA, and Click 'INITIATE PATHOGEN'")
        center_lat, center_lon, _ = LOCATIONS[target]
        fig = px.scatter_geo(lat=[center_lat], lon=[center_lon], projection="orthographic", template="plotly_dark")
        fig.update_geos(
            bgcolor="black", showocean=True, oceancolor="#111111",
            showland=True, landcolor="#222222",
            showcountries=True, countrycolor="#555555",
            projection_rotation=dict(lon=center_lon, lat=center_lat)
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500)
        st.plotly_chart(fig, use_container_width=True)