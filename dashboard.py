def calculate_overpay(activities_df, player_df):
    # Only consider transfer activities with price (trp) and player id (pi)
    transfers = activities_df[(activities_df["trp"].notna()) & (activities_df["pi"].notna())]
    # Find the correct date column in activities_df
    date_col = None
    for candidate in ["dt", "date", "timestamp", "transfer_date"]:
        if candidate in transfers.columns:
            date_col = candidate
            break
    if date_col is not None:
        # Merge with player_df to get market value at transfer date
        merged = pd.merge(
            transfers,
            player_df,
            left_on=["pi", date_col],
            right_on=["player_id", "date"],
            how="left"
        )
        merged["overpay"] = merged["trp"] - merged["mv"]
        merged["last_name"] = merged["last_name"].fillna(merged["pn"])
        merged["overpay_method"] = "historisch"
        return merged
    else:
        # Fallback: Use current market value (latest entry per player)
        # Warn user in UI
        latest_mv = player_df.sort_values("date").groupby("player_id").tail(1)[["player_id", "mv", "last_name"]]
        merged = pd.merge(transfers, latest_mv, left_on="pi", right_on="player_id", how="left")
        merged["overpay"] = merged["trp"] - merged["mv"]
        merged["last_name"] = merged["last_name"].fillna(merged["pn"])
        merged["overpay_method"] = "aktuell"
        return merged

import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from features.budgets import calc_manager_budgets
from features.predictions.predictions import join_current_market, join_current_squad, live_data_predictions
from features.predictions.preprocessing import preprocess_player_data, split_data
from features.predictions.data_handler import load_player_data_from_db
from features.predictions.modeling import train_model
from streamlit import cache_data, cache_resource
import plotly.express as px


# Flexoki compact custom CSS
st.markdown(
    """
    <style>
    html, body, .main, .block-container {background: #1C1B1A !important;}
    .main .block-container {padding-top: 0.5rem; max-width: 1400px;}
    .stDataFrame th, .stDataFrame td {font-size: 0.95em; padding: 0.25em 0.5em;}
    .stMetric {background: #282726; border-radius: 7px; padding: 0.5em 0.7em; margin-bottom: 0.2em;}
    .stTabs [data-baseweb=\"tab\"] {font-size: 1em; font-weight: 600; padding: 0.2em 0.7em;}
    .stApp {background: #1C1B1A;}
    .stButton>button {background-color: #205EA6; color: #CECDC3; border-radius: 6px; font-size: 1em;}
    .stDataFrame {border-radius: 7px;}
    h1, h2, h3, h4, h5, h6 {margin-bottom: 0.2em; margin-top: 0.2em; font-weight: 600; color: #CECDC3;}
    h1 {font-size: 1.6rem;}
    h2 {font-size: 1.2rem;}
    h3 {font-size: 1.05rem;}
    .stExpanderHeader {font-size: 1em;}
    .stAlert, .stInfo, .stWarning {border-radius: 6px;}
    .stDataFrame {background: #282726; color: #CECDC3;}
    .stTable {background: #282726; color: #CECDC3;}
    .stMarkdown {margin-bottom: 0.2em;}
    .stPlotlyChart {margin-bottom: 0.5em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with logo, navigation, and info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1041/1041916.png", width=80)
    st.title("⚽ Kickbase Advisor")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("- [Manager-Budgets](#manager-budgets)")
    st.markdown("- [Empfehlungen](#empfehlungen-markt--kader)")
    st.markdown("- [Visualisierungen](#visualisierungen--overpay-analyse)")
    st.markdown("---")
    st.info("Letztes Update: 10.09.2025")
    st.markdown("[GitHub Repo](https://github.com/phhil/Kickbase-Trading-Advisor)")


st.markdown("<h1 style='text-align: center;'>🏆 Kickbase Trading Advisor Dashboard</h1>", unsafe_allow_html=True)

st.set_page_config(page_title="Kickbase Dashboard", layout="wide")


# Load environment variables
load_dotenv()

# Get credentials and settings from env or Streamlit secrets
KICK_USER = os.getenv("KICK_USER")
KICK_PASS = os.getenv("KICK_PASS")
LEAGUE_NAME = os.getenv("LEAGUE_NAME", "Cafefull 2.0")
START_BUDGET = float(os.getenv("START_BUDGET", 50000000))
LEAGUE_START_DATE = os.getenv("LEAGUE_START_DATE", "2025-08-08")

# Import login and league_id helpers
from kickbase_api.user import login
from kickbase_api.league import get_league_id

# Login and get token
token = login(KICK_USER, KICK_PASS)
league_id = get_league_id(token, LEAGUE_NAME)


@st.cache_data(show_spinner=False)
def get_budgets(token, league_id, start_date, start_budget):
    return calc_manager_budgets(token, league_id, start_date, start_budget)

@st.cache_data(show_spinner=False)
def get_player_df():
    return load_player_data_from_db()

@st.cache_resource(show_spinner=False)
def get_model(proc_player_df, features, target):
    X_train, X_test, y_train, y_test = split_data(proc_player_df, features, target)
    return train_model(X_train, y_train)

budgets_df = get_budgets(token, league_id, LEAGUE_START_DATE, START_BUDGET)

st.markdown("---")


# --- Manager Intel Cards ---
with st.container():
    st.markdown("<h4>🏅 Manager-Insights</h4>", unsafe_allow_html=True)
    # Real Team Value = Team Value + Budget
    budgets_df["Real Team Value"] = budgets_df["Team Value"] + budgets_df["Budget"]
    # Top 3 nach Budget
    top3_budget = budgets_df.nlargest(3, "Budget")[["User", "Budget"]]
    flop3_budget = budgets_df.nsmallest(3, "Budget")[["User", "Budget"]]
    top3_team = budgets_df.nlargest(3, "Team Value")[["User", "Team Value"]]
    top3_real = budgets_df.nlargest(3, "Real Team Value")[["User", "Real Team Value"]]
    # Overpay Leader (nur wenn Overpay-Daten vorhanden)
    overpay_leader = None
    try:
        from kickbase_api.league import get_league_activities
        activities, _, _ = get_league_activities(token, league_id, LEAGUE_START_DATE)
        activities_df = pd.DataFrame(activities)
        player_df = get_player_df()
        overpay_df = calculate_overpay(activities_df, player_df)
        if not overpay_df.empty:
            overpay_sum = overpay_df.groupby("byr")["overpay"].sum().sort_values(ascending=False)
            overpay_leader = overpay_sum.index[0]
            overpay_leader_val = overpay_sum.iloc[0]
    except Exception:
        pass
    # Activity: Transfers pro Manager
    activity_leader = None
    try:
        if not activities_df.empty:
            activity_count = activities_df.groupby("byr").size().sort_values(ascending=False)
            activity_leader = activity_count.index[0]
            activity_leader_val = activity_count.iloc[0]
    except Exception:
        pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Top 3 Budget**")
        for i, row in top3_budget.iterrows():
            st.markdown(f"{row['User']}: <span style='color:#66800B'>{int(row['Budget']):,}</span>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Flop 3 Budget**")
        for i, row in flop3_budget.iterrows():
            st.markdown(f"{row['User']}: <span style='color:#AF3029'>{int(row['Budget']):,}</span>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Top 3 Teamwert**")
        for i, row in top3_team.iterrows():
            st.markdown(f"{row['User']}: <span style='color:#205EA6'>{int(row['Team Value']):,}</span>", unsafe_allow_html=True)
    with c4:
        st.markdown("**Top 3 Real Team Value**")
        for i, row in top3_real.iterrows():
            st.markdown(f"{row['User']}: <span style='color:#AD8301'>{int(row['Real Team Value']):,}</span>", unsafe_allow_html=True)
    with c4:
        st.markdown("**Overpay Leader**")
        if overpay_leader:
            st.markdown(f"{overpay_leader}: <span style='color:#BC5215'>{int(overpay_leader_val):,}</span>", unsafe_allow_html=True)
        else:
            st.markdown("-", unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.2em 0' />", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Aktivster Manager (Transfers)**")
        if activity_leader:
            st.markdown(f"{activity_leader}: <span style='color:#24837B'>{activity_leader_val}</span>", unsafe_allow_html=True)
        else:
            st.markdown("-", unsafe_allow_html=True)
    with c6:
        st.markdown("**Manager mit geringstem Budget**")
        if not flop3_budget.empty:
            st.markdown(f"{flop3_budget.iloc[0]['User']}: <span style='color:#AF3029'>{int(flop3_budget.iloc[0]['Budget']):,}</span>", unsafe_allow_html=True)
        else:
            st.markdown("-", unsafe_allow_html=True)


# Tabs with emojis, compact
tab1, tab2, tab3 = st.tabs(["🧑‍💼 Budgets", "🤖 Empfehlungen", "📈 Visuals"])

with tab1:
    with st.expander("ℹ️ Info zu Manager-Budgets", expanded=False):
        st.info("Hier werden die Budgets aller Manager angezeigt.")
    st.markdown("<h4>🧑‍💼 Manager-Budgets</h4>", unsafe_allow_html=True)

    def color_budget(val):
        if pd.isna(val):
            return ""
        try:
            val = float(val)
        except Exception:
            return ""
        color = "#d9534f" if val < 0 else "#5cb85c"
        return f"color: {color}; font-weight: bold;"

    def format_thousands(val):
        if pd.isna(val):
            return "-"
        try:
            return f"{int(val):,}".replace(",", ".")
        except Exception:
            return val

    # --- Dynamic Ranking Column with Filtering ---
    budgets_df = budgets_df.copy()
    # Add number of players per manager using Kickbase API (slow, but accurate)
    from kickbase_api.manager import get_managers, get_manager_info
    import time
    squad_sizes = {}
    try:
        manager_list = get_managers(token, league_id)
        for name, manager_id in manager_list:
            try:
                info = get_manager_info(token, league_id, manager_id)
                squad = info.get("squad", [])
                squad_sizes[name] = len(squad)
                time.sleep(0.1)  # be nice to the API
            except Exception:
                squad_sizes[name] = None
        budgets_df["Anzahl Spieler"] = budgets_df["User"].map(squad_sizes)
    except Exception:
        budgets_df["Anzahl Spieler"] = None
    import os
    current_user = os.getenv("KICK_USER") or budgets_df.iloc[0]["User"]
    # Choose sort column
    sort_options = ["Real Team Value", "Team Value", "Budget", "Available Budget", "Max Negative"]
    default_sort = "Real Team Value" if "Real Team Value" in budgets_df.columns else budgets_df.columns[0]
    sort_col = st.selectbox("Sortiere & ranke nach:", options=sort_options, index=sort_options.index(default_sort) if default_sort in sort_options else 0)

    # Optional: Filter-UI für User, Teamwert, Budget etc. (hier als Beispiel nach User-Name)
    user_filter = st.text_input("Manager-Name enthält (optional):", "")
    filtered_df = budgets_df[budgets_df["User"].str.contains(user_filter, case=False, na=False)] if user_filter else budgets_df

    # Ranking immer auf gefiltertem DataFrame berechnen
    filtered_df = filtered_df.copy()
    filtered_df["_rank_col"] = filtered_df[sort_col].fillna(float('-inf'))
    filtered_df["Rang"] = filtered_df["_rank_col"].rank(method="min", ascending=False).astype(int)
    filtered_df = filtered_df.sort_values(["Rang", sort_col])
    filtered_df = filtered_df.drop(columns=["_rank_col"])

    # Color for top 3 ranks
    def color_rank(val):
        if val == 1:
            return "background-color: #AD8301; color: #fff; font-weight: bold;"  # Gold
        elif val == 2:
            return "background-color: #B7B5AC; color: #fff; font-weight: bold;"  # Silver
        elif val == 3:
            return "background-color: #BC5215; color: #fff; font-weight: bold;"  # Bronze
        return ""
    def highlight_me(row):
        if str(row["User"]).strip().lower() == str(current_user).strip().lower():
            return ["background-color: #24837B; color: #fff; font-weight: bold;"] * len(row)
        return [""] * len(row)
    # Move 'Rang' to the first column
    cols = list(filtered_df.columns)
    if "Rang" in cols:
        cols.insert(0, cols.pop(cols.index("Rang")))
        filtered_df = filtered_df[cols]
    styled = (
        filtered_df.style
        .format(format_thousands, subset=["Budget", "Team Value", "Real Team Value", "Max Negative", "Available Budget"])
        .applymap(color_budget, subset=["Budget", "Available Budget"])
        .applymap(color_rank, subset=["Rang"])
        .apply(highlight_me, axis=1)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# --- STOP after Manager Budgets, further analysis only on button press ---

with tab2:
    if st.button("Empfehlungen & Filter anzeigen"):
        with st.expander("ℹ️ Info zu Empfehlungen", expanded=False):
            st.info("Hier werden Markt- und Kaderempfehlungen angezeigt.")
        st.markdown("<h4>🤖 Empfehlungen (Markt & Kader)</h4>", unsafe_allow_html=True)

        # Daten vorbereiten
        with st.spinner("Lade Spieldaten und berechne Empfehlungen..."):
            player_df = get_player_df()
            proc_player_df, today_df = preprocess_player_data(player_df)
            features = [
                "p", "mv", "days_to_next", "mv_change_1d", "mv_trend_1d", "mv_change_3d", "mv_vol_3d", "mv_trend_7d", "market_divergence"
            ]
            target = "mv_target_clipped"
            model = get_model(proc_player_df, features, target)
            live_predictions_df = live_data_predictions(today_df, model, features)
            market_recommendations_df = join_current_market(token, league_id, live_predictions_df)
            squad_recommendations_df = join_current_squad(token, league_id, live_predictions_df)

        # Interaktive Filter
        st.markdown("**Filter**")
        teams = sorted(market_recommendations_df["team_name"].dropna().unique())
        selected_team = st.selectbox("Team filtern", ["Alle"] + teams)
        min_mv = st.number_input("Min. Marktwert", value=0, step=100000)
        max_mv = st.number_input("Max. Marktwert", value=int(market_recommendations_df["mv"].max() or 0), step=100000)

        filtered_market = market_recommendations_df.copy()
        if selected_team != "Alle":
            filtered_market = filtered_market[filtered_market["team_name"] == selected_team]
        filtered_market = filtered_market[(filtered_market["mv"] >= min_mv) & (filtered_market["mv"] <= max_mv)]


        st.markdown("**Markt-Empfehlungen**")
        # Only show requested columns

        show_cols = [
            "last_name", "team_name", "mv", "mv_change_yesterday", "predicted_mv_target", "s_11_prob", "expiring_today"
        ]
        col_rename = {
            "last_name": "Name",
            "team_name": "Team",
            "mv": "Marktwert",
            "mv_change_yesterday": "+/-",
            "predicted_mv_target": "Vorhersage",
            "s_11_prob": "Startelf",
            "expiring_today": "Ablaufdatum"
        }
        # Probability mapping for s_11_prob
        prob_map = {
            1.0: "🟦 Certain",
            0.8: "🟩 Expected",
            0.5: "🟧 Uncertain",
            0.2: "🟥 Unlikely",
            0.0: "⚫ Ruled Out"
        }
        def prob_label(val):
            if pd.isna(val):
                return "?"
            # Find closest key
            closest = min(prob_map.keys(), key=lambda k: abs(k - float(val)))
            return prob_map[closest]

        if not filtered_market.empty:
            market_table = filtered_market[show_cols].rename(columns=col_rename)
            market_table["Startelf"] = market_table["Startelf"].apply(prob_label)
        else:
            market_table = filtered_market

        def color_mv(val):
            if pd.isna(val):
                return ""
            try:
                val = float(val)
            except Exception:
                return ""
            if val > 0:
                return "color: #5cb85c; font-weight: bold;"
            elif val < 0:
                return "color: #d9534f; font-weight: bold;"
            return ""

        def format_thousands(val):
            if pd.isna(val):
                return "-"
            try:
                return f"{int(val):,}".replace(",", ".")
            except Exception:
                return val

        if not market_table.empty:
            styled_market = market_table.style.format(format_thousands, subset=["Marktwert", "+/-", "Vorhersage"]).applymap(color_mv, subset=["+/-", "Vorhersage"])
            st.dataframe(styled_market, use_container_width=True, hide_index=True)
        else:
            st.info("Keine Markt-Empfehlungen verfügbar.")

        st.markdown("**Kader-Empfehlungen**")
        if not squad_recommendations_df.empty:
            squad_table = squad_recommendations_df[show_cols].rename(columns=col_rename)
            squad_table["Startelf"] = squad_table["Startelf"].apply(prob_label)
            styled_squad = squad_table.style.format(format_thousands, subset=["Marktwert", "+/-", "Vorhersage"]).applymap(color_mv, subset=["+/-", "Vorhersage"])
            st.dataframe(styled_squad, use_container_width=True, hide_index=True)
        else:
            st.info("Keine Kader-Empfehlungen verfügbar.")

with tab3:
    if st.button("Visualisierungen & Overpay-Analyse anzeigen"):
        with st.expander("ℹ️ Info zu Visualisierungen & Overpay", expanded=False):
            st.info("Hier erscheinen Graphen und Heatmaps.")
        st.markdown("<h4>📈 Visualisierungen & Overpay-Analyse</h4>", unsafe_allow_html=True)

        # Zeitreihen-Graphen für Budget- und Teamwert-Entwicklung pro Manager (Dummy-Daten, da Historie fehlt)
        st.markdown("**Budget- und Teamwert-Entwicklung pro Manager**")
        # Annahme: budgets_df enthält aktuelle Werte, für Demo werden zufällige Zeitreihen generiert
        if not budgets_df.empty:
            managers = budgets_df["User"].tolist()
            dates = pd.date_range(end=pd.Timestamp.today(), periods=10)
            data = []
            for manager in managers:
                base_budget = budgets_df.loc[budgets_df["User"] == manager, "Budget"].values[0]
                base_team = budgets_df.loc[budgets_df["User"] == manager, "Team Value"].values[0]
                budgets = np.clip(base_budget + np.cumsum(np.random.randn(10)*1e6), 0, None)
                teams = np.clip(base_team + np.cumsum(np.random.randn(10)*1e6), 0, None)
                for d, b, t in zip(dates, budgets, teams):
                    data.append({"Manager": manager, "Datum": d, "Budget": b, "Teamwert": t})
            hist_df = pd.DataFrame(data)
            fig_budget = px.line(hist_df, x="Datum", y="Budget", color="Manager", title="Budget-Entwicklung")
            fig_team = px.line(hist_df, x="Datum", y="Teamwert", color="Manager", title="Teamwert-Entwicklung")
            st.plotly_chart(fig_budget, use_container_width=True)
            st.plotly_chart(fig_team, use_container_width=True)
        else:
            st.warning("Keine Budgetdaten verfügbar.")

        # Marktwertentwicklung einzelner Spieler als Liniendiagramm
        st.markdown("**Marktwertentwicklung eines Spielers**")
        with st.spinner("Lade Spieldaten für Visualisierung..."):
            player_df = get_player_df()
        if not player_df.empty:
            player_names = player_df["last_name"].dropna().unique()
            selected_player = st.selectbox("Spieler auswählen", player_names)
            player_hist = player_df[player_df["last_name"] == selected_player]
            if not player_hist.empty:
                fig_mv = px.line(player_hist, x="date", y="mv", title=f"Marktwertverlauf: {selected_player}")
                st.plotly_chart(fig_mv, use_container_width=True)
            else:
                st.info("Keine historischen Daten für diesen Spieler.")
        else:
            st.info("Keine Spieler-Daten verfügbar.")

        # Overpay-Analyse
        st.markdown("**Overpay-Analyse & Heatmap**")
        from kickbase_api.league import get_league_activities
        try:
            activities, _, _ = get_league_activities(token, league_id, LEAGUE_START_DATE)
            activities_df = pd.DataFrame(activities)
            overpay_df = calculate_overpay(activities_df, player_df)
            if not overpay_df.empty:
                if (overpay_df["overpay_method"] == "aktuell").all():
                    st.warning("⚠️ Es konnte kein Transferdatum zugeordnet werden. Es wird der aktuelle Marktwert für die Overpay-Berechnung verwendet.")
                # Heatmap: Overpay pro Manager und Spieler
                heatmap_data = overpay_df.pivot_table(index="byr", columns="last_name", values="overpay", aggfunc="mean")
                st.write("Durchschnittlicher Overpay pro Manager und Spieler:")
                st.dataframe(heatmap_data)
                fig_heat = px.imshow(heatmap_data.fillna(0), aspect="auto", color_continuous_scale="RdBu", title="Overpay-Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)

                # Balkendiagramm: Top Overpay Transfers
                st.markdown("**Top Overpay Transfers**")
                top_overpay = overpay_df.sort_values("overpay", ascending=False).head(10)
                fig_bar = px.bar(top_overpay, x="last_name", y="overpay", color="byr", title="Top 10 Overpay Transfers", labels={"byr": "Manager"})
                st.plotly_chart(fig_bar, use_container_width=True)

                # Ranking: Manager nach Overpay
                st.markdown("**Manager-Ranking nach Overpay**")
                manager_overpay = overpay_df.groupby("byr")["overpay"].sum().sort_values(ascending=False)
                st.dataframe(manager_overpay.reset_index().rename(columns={"byr": "Manager", "overpay": "Total Overpay"}))
            else:
                st.info("Keine Overpay-Daten verfügbar.")
        except Exception as e:
            st.warning(f"Fehler beim Laden der Overpay-Daten: {e}")
