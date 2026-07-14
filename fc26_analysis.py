import os
import re
import unicodedata
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

from PIL import Image


# ==========================================================
# Page Settings
# ==========================================================

st.set_page_config(
    page_title="FC26 Players Analysis",
    page_icon="⚽",
    layout="wide"
)


# ==========================================================
# File Paths
# ==========================================================

data_path = "FC26_20250921.csv"

images_folder = images_folder = "images"


# ==========================================================
# Statistics Columns
# ==========================================================

stat_columns = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",

    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",

    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",

    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",

    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",

    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",

    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",

    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes"
]


# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        data_path,
        dtype={"player_tags": str},
        low_memory=False
    )

    numeric_columns = [
        "overall",
        "potential",
        "age",
        "value_eur",
        "wage_eur"
    ] + stat_columns

    existing_numeric_columns = [
        column
        for column in numeric_columns
        if column in data.columns
    ]

    data[existing_numeric_columns] = (
        data[existing_numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    return data


fc26 = load_data()


existing_stat_columns = [
    column
    for column in stat_columns
    if column in fc26.columns
]


# ==========================================================
# Normalize Player Names
# ==========================================================

def normalize_name(name):

    name = str(name).strip().lower()

    name = unicodedata.normalize(
        "NFKD",
        name
    )

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    name = re.sub(
        r"[^a-z0-9]",
        "",
        name
    )

    return name


# ==========================================================
# Create Local Image Index
# ==========================================================

@st.cache_resource
def create_image_index():

    image_index = {}

    image_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    )

    if not os.path.isdir(images_folder):
        return image_index

    for filename in os.listdir(images_folder):

        if filename.lower().endswith(image_extensions):

            filename_without_extension = os.path.splitext(
                filename
            )[0]

            normalized_filename = normalize_name(
                filename_without_extension
            )

            image_index[normalized_filename] = os.path.join(
                images_folder,
                filename
            )

    return image_index


image_index = create_image_index()


# ==========================================================
# Find Local Player Image
# ==========================================================

def find_local_player_image(player):

    possible_names = []

    possible_columns = [
        "short_name",
        "long_name",
        "player_id",
        "sofifa_id"
    ]

    for column in possible_columns:

        if column in player.index:

            value = player[column]

            if pd.notna(value):
                possible_names.append(value)

    for name in possible_names:

        normalized_player_name = normalize_name(
            name
        )

        if normalized_player_name in image_index:

            return image_index[
                normalized_player_name
            ]

    return None


# ==========================================================
# Download Player Image From URL
# ==========================================================

@st.cache_data(show_spinner=False)
def download_player_image(image_url):

    if pd.isna(image_url):
        return None

    image_url = str(image_url).strip()

    if not image_url.startswith(
        ("http://", "https://")
    ):
        return None

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            image_url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        )

        return image.copy()

    except Exception:

        return None


# ==========================================================
# Get Player Image
# ==========================================================

def get_player_image(player):

    if "player_face_url" in player.index:

        image_url = player[
            "player_face_url"
        ]

        online_image = download_player_image(
            image_url
        )

        if online_image is not None:
            return online_image

    local_image_path = find_local_player_image(
        player
    )

    if local_image_path:

        try:

            local_image = Image.open(
                local_image_path
            )

            return local_image.copy()

        except Exception:

            return None

    return None


# ==========================================================
# Formatting Functions
# ==========================================================

def format_money(value):

    if pd.isna(value):
        return "Unknown"

    value = float(value)

    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"€{value / 1_000:.1f}K"

    return f"€{value:,.0f}"


def format_stat_name(stat_column):

    special_names = {
        "overall": "Overall Rating",
        "physic": "Physical"
    }

    if stat_column in special_names:
        return special_names[stat_column]

    return stat_column.replace(
        "_",
        " "
    ).title()


# ==========================================================
# Display Player Card
# ==========================================================

def display_player_card(
    player,
    main_stat,
    main_stat_label
):

    player_image = get_player_image(
        player
    )

    if player_image is not None:

        st.image(
            player_image,
            use_container_width=True
        )

    else:

        st.warning(
            "No image available"
        )

    st.markdown(
        f"### {player['short_name']}"
    )

    st.metric(
        label=main_stat_label,
        value=int(
            player[main_stat]
        )
    )

    if (
        main_stat != "overall"
        and "overall" in player.index
        and pd.notna(player["overall"])
    ):

        st.write(
            f"**Overall:** "
            f"{int(player['overall'])}"
        )

    if (
        "club_name" in player.index
        and pd.notna(player["club_name"])
    ):

        st.write(
            f"**Club:** "
            f"{player['club_name']}"
        )

    if (
        "player_positions" in player.index
        and pd.notna(player["player_positions"])
    ):

        st.write(
            f"**Position:** "
            f"{player['player_positions']}"
        )

    if (
        "age" in player.index
        and pd.notna(player["age"])
    ):

        st.write(
            f"**Age:** "
            f"{int(player['age'])}"
        )

    if "value_eur" in player.index:

        st.write(
            "**Value:** "
            + format_money(
                player["value_eur"]
            )
        )


# ==========================================================
# Get Top Players
# ==========================================================

def get_top_players(
    dataframe,
    stat_column,
    number_of_players
):

    sorting_columns = [
        stat_column
    ]

    ascending_values = [
        False
    ]

    if (
        stat_column != "overall"
        and "overall" in dataframe.columns
    ):

        sorting_columns.append(
            "overall"
        )

        ascending_values.append(
            False
        )

    top_players = (
        dataframe
        .dropna(
            subset=[
                "short_name",
                stat_column
            ]
        )
        .sort_values(
            by=sorting_columns,
            ascending=ascending_values
        )
        .head(number_of_players)
        .reset_index(drop=True)
    )

    return top_players


# ==========================================================
# Display Players
# ==========================================================

def display_players(
    dataframe,
    stat_column,
    number_of_players
):

    stat_label = format_stat_name(
        stat_column
    )

    top_players = get_top_players(
        dataframe=dataframe,
        stat_column=stat_column,
        number_of_players=number_of_players
    )

    for start_index in range(
        0,
        len(top_players),
        5
    ):

        player_columns = st.columns(
            5
        )

        current_players = top_players.iloc[
            start_index:start_index + 5
        ]

        for column, (_, player) in zip(
            player_columns,
            current_players.iterrows()
        ):

            with column:

                display_player_card(
                    player=player,
                    main_stat=stat_column,
                    main_stat_label=stat_label
                )

    return top_players


# ==========================================================
# Display Bar Chart
# ==========================================================

def display_bar_chart(
    top_players,
    stat_column,
    chart_title
):

    if top_players.empty:
        return

    chart_data = (
        top_players
        .sort_values(
            stat_column,
            ascending=True
        )
    )

    stat_label = format_stat_name(
        stat_column
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars = ax.barh(
        chart_data["short_name"],
        chart_data[stat_column],
        height=0.65
    )

    ax.set_title(
        chart_title,
        fontsize=17,
        fontweight="bold"
    )

    ax.set_xlabel(
        stat_label
    )

    ax.set_ylabel(
        "Player Name"
    )

    minimum_value = chart_data[
        stat_column
    ].min()

    maximum_value = chart_data[
        stat_column
    ].max()

    chart_range = maximum_value - minimum_value

    if chart_range == 0:
        chart_range = 2

    padding = max(
        2,
        chart_range * 0.25
    )

    ax.set_xlim(
        minimum_value - padding,
        maximum_value + padding
    )

    for bar in bars:

        stat_value = bar.get_width()

        ax.text(
            stat_value + 0.1,
            bar.get_y()
            + bar.get_height() / 2,
            str(int(stat_value)),
            va="center",
            fontweight="bold"
        )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    ax.set_axisbelow(
        True
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


# ==========================================================


# ==========================================================
# Extra Dashboard Helpers
# ==========================================================

def safe_text(player, column, default="Unknown"):
    value = player.get(column, default)
    if pd.isna(value):
        return default
    return str(value)


def find_player_by_aliases(dataframe, aliases):
    name_columns = [
        column
        for column in ["short_name", "long_name"]
        if column in dataframe.columns
    ]

    for alias in aliases:
        alias_normalized = normalize_name(alias)

        for name_column in name_columns:
            normalized_names = (
                dataframe[name_column]
                .fillna("")
                .astype(str)
                .map(normalize_name)
            )

            exact_matches = dataframe[
                normalized_names == alias_normalized
            ]

            if not exact_matches.empty:
                return exact_matches.sort_values(
                    "overall",
                    ascending=False
                ).iloc[0]

        for name_column in name_columns:
            contains_matches = dataframe[
                dataframe[name_column]
                .fillna("")
                .astype(str)
                .str.contains(
                    alias,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

            if not contains_matches.empty:
                return contains_matches.sort_values(
                    "overall",
                    ascending=False
                ).iloc[0]

    return None


def display_basic_player_card(player, title=None):
    player_image = get_player_image(player)

    if player_image is not None:
        st.image(player_image, use_container_width=True)
    else:
        st.info("No image available")

    st.markdown(f"### {title or safe_text(player, 'short_name')}")

    if pd.notna(player.get("overall")):
        st.metric("Overall", int(player["overall"]))

    st.write(f"**Club:** {safe_text(player, 'club_name')}")
    st.write(f"**Position:** {safe_text(player, 'player_positions')}")

    if pd.notna(player.get("age")):
        st.write(f"**Age:** {int(player['age'])}")

    if pd.notna(player.get("value_eur")):
        st.write(f"**Value:** {format_money(player['value_eur'])}")


def render_comparison(players, stats, title):
    if not players:
        st.warning("No players were found for this comparison.")
        return

    st.subheader(title)

    columns = st.columns(len(players))
    for column, (display_name, player) in zip(columns, players.items()):
        with column:
            display_basic_player_card(player, display_name)

    valid_stats = [
        stat
        for stat in stats
        if stat in fc26.columns
    ]

    if not valid_stats:
        st.warning("No comparison statistics are available.")
        return

    comparison_table = pd.DataFrame(
        {
            display_name: [player.get(stat, np.nan) for stat in valid_stats]
            for display_name, player in players.items()
        },
        index=[format_stat_name(stat) for stat in valid_stats]
    ).apply(pd.to_numeric, errors="coerce")

    st.subheader("📋 Statistics Table")
    st.dataframe(comparison_table, use_container_width=True)

    st.subheader("📊 Attributes Comparison")
    fig_bar, ax_bar = plt.subplots(figsize=(14, 7))
    comparison_table.T.plot(kind="bar", ax=ax_bar)
    ax_bar.set_title(title, fontsize=17, fontweight="bold")
    ax_bar.set_xlabel("Player")
    ax_bar.set_ylabel("Rating")
    ax_bar.set_ylim(0, 100)
    ax_bar.tick_params(axis="x", rotation=0)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    ax_bar.legend(
        title="Statistic",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    radar_stats = [stat for stat in valid_stats if stat != "overall"]

    if len(radar_stats) >= 3:
        st.subheader("🕸️ Radar Chart Comparison")

        labels = [format_stat_name(stat) for stat in radar_stats]
        angles = np.linspace(
            0,
            2 * np.pi,
            len(radar_stats),
            endpoint=False
        ).tolist()
        angles += angles[:1]

        fig_radar, ax_radar = plt.subplots(
            figsize=(9, 9),
            subplot_kw={"polar": True}
        )

        for display_name, player in players.items():
            values = []
            for stat in radar_stats:
                value = player.get(stat, 0)
                values.append(0 if pd.isna(value) else float(value))
            values += values[:1]

            ax_radar.plot(
                angles,
                values,
                linewidth=2,
                label=display_name
            )
            ax_radar.fill(angles, values, alpha=0.08)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(labels)
        ax_radar.set_ylim(0, 100)
        ax_radar.set_title(title, fontsize=17, fontweight="bold", pad=25)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
        plt.tight_layout()
        st.pyplot(fig_radar)
        plt.close(fig_radar)


def player_options(dataframe):
    columns = ["short_name"]
    available = dataframe.dropna(subset=columns).copy()

    if "overall" in available.columns:
        available = available.sort_values("overall", ascending=False)

    return available["short_name"].astype(str).drop_duplicates().tolist()


def get_player_by_short_name(dataframe, player_name):
    matches = dataframe[
        dataframe["short_name"].astype(str) == str(player_name)
    ]

    if matches.empty:
        return None

    return matches.sort_values("overall", ascending=False).iloc[0]


# ==========================================================
# Prepared Comparisons — Neymar Removed
# ==========================================================

comparison_groups = {
    "Cristiano Ronaldo vs Lionel Messi": {
        "players": {
            "Cristiano Ronaldo": [
                "Cristiano Ronaldo",
                "Cristiano Ronaldo dos Santos Aveiro",
                "Cristiano Ronaldo dos Santos Aveiro"
            ],
            "Lionel Messi": [
                "Lionel Messi",
                "Lionel Andrés Messi",
                "L. Messi"
            ]
        },
        "stats": [
            "overall", "pace", "shooting", "passing",
            "dribbling", "physic", "attacking_finishing"
        ]
    },
    "Alisson vs Courtois vs Ter Stegen": {
        "players": {
            "Alisson": ["Alisson", "Alisson Ramses Becker"],
            "Thibaut Courtois": ["Thibaut Courtois", "T. Courtois"],
            "Marc-André ter Stegen": [
                "Marc-André ter Stegen",
                "Marc Andre ter Stegen",
                "ter Stegen"
            ]
        },
        "stats": [
            "overall", "goalkeeping_diving", "goalkeeping_handling",
            "goalkeeping_kicking", "goalkeeping_positioning",
            "goalkeeping_reflexes"
        ]
    },
    "Van Dijk vs Saliba vs Rüdiger": {
        "players": {
            "Virgil van Dijk": ["Virgil van Dijk", "V. van Dijk"],
            "William Saliba": ["William Saliba", "W. Saliba"],
            "Antonio Rüdiger": [
                "Antonio Rüdiger", "Antonio Rudiger", "A. Rüdiger"
            ]
        },
        "stats": [
            "overall", "pace", "defending", "physic",
            "mentality_interceptions", "defending_marking_awareness",
            "defending_standing_tackle", "defending_sliding_tackle"
        ]
    },
    "Marmoush vs Salah": {
        "players": {
            "Omar Marmoush": ["Omar Marmoush", "O. Marmoush"],
            "Mohamed Salah": ["Mohamed Salah", "M. Salah"]
        },
        "stats": [
            "overall", "pace", "shooting", "passing",
            "dribbling", "physic", "attacking_finishing"
        ]
    },
    "Mbappé vs Haaland vs Harry Kane": {
        "players": {
            "Kylian Mbappé": ["Kylian Mbappé", "Kylian Mbappe", "K. Mbappé"],
            "Erling Haaland": ["Erling Haaland", "E. Haaland"],
            "Harry Kane": ["Harry Kane", "H. Kane"]
        },
        "stats": [
            "overall", "pace", "shooting", "passing", "dribbling",
            "physic", "attacking_finishing", "power_shot_power"
        ]
    },
    "Bellingham vs Vitinha vs Pedri": {
        "players": {
            "Jude Bellingham": ["Jude Bellingham", "J. Bellingham"],
            "Vitinha": ["Vitinha", "Vítor Machado Ferreira"],
            "Pedri": ["Pedri", "Pedro González López"]
        },
        "stats": [
            "overall", "pace", "shooting", "passing", "dribbling",
            "defending", "physic", "mentality_vision",
            "skill_ball_control"
        ]
    }
}


# ==========================================================
# Sidebar Navigation
# ==========================================================

st.sidebar.title("⚽ FC26 Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🏆 Top Players",
        "⚔️ Compare Players",
        "🔎 Search Player",
        "🎯 Filters",
        "💡 Insights"
    ]
)


# ==========================================================
# Home
# ==========================================================

if page == "🏠 Home":
    st.title("⚽ FC26 Players Analysis")
    st.write(
        "An interactive dashboard for exploring player ratings, "
        "finding top performers, filtering the dataset and comparing players."
    )

    st.divider()

    metric_columns = st.columns(4)

    with metric_columns[0]:
        st.metric("Players", f"{len(fc26):,}")

    with metric_columns[1]:
        st.metric(
            "Clubs",
            f"{fc26['club_name'].nunique():,}"
            if "club_name" in fc26.columns else "Unknown"
        )

    with metric_columns[2]:
        st.metric(
            "Nationalities",
            f"{fc26['nationality_name'].nunique():,}"
            if "nationality_name" in fc26.columns else "Unknown"
        )

    with metric_columns[3]:
        st.metric(
            "Highest Overall",
            int(fc26["overall"].max())
            if "overall" in fc26.columns else "Unknown"
        )

    st.subheader("Dashboard Sections")
    st.write(
        "Use the sidebar to open Top Players, Player Comparison, "
        "Player Search, Filters and Insights."
    )


# ==========================================================
# Top Players
# ==========================================================

elif page == "🏆 Top Players":
    st.title("🏆 Top Players")

    all_dashboard_columns = ["overall"] + existing_stat_columns

    selection_column, count_column = st.columns([3, 1])

    with selection_column:
        selected_stat = st.selectbox(
            "Select Statistic",
            all_dashboard_columns,
            format_func=format_stat_name
        )

    with count_column:
        number_of_players = st.selectbox(
            "Number of Players",
            [5, 10],
            index=0
        )

    selected_label = format_stat_name(selected_stat)
    st.subheader(f"Top {number_of_players} Players by {selected_label}")

    selected_players = display_players(
        fc26,
        selected_stat,
        number_of_players
    )

    st.divider()
    display_bar_chart(
        selected_players,
        selected_stat,
        f"Top {number_of_players} Players by {selected_label}"
    )


# ==========================================================
# Compare Players
# ==========================================================

elif page == "⚔️ Compare Players":
    st.title("⚔️ Player Comparison")

    comparison_mode = st.radio(
        "Comparison Type",
        ["Prepared Comparisons", "Custom Comparison"],
        horizontal=True
    )

    if comparison_mode == "Prepared Comparisons":
        selected_group_name = st.selectbox(
            "Select Comparison",
            list(comparison_groups.keys())
        )

        selected_group = comparison_groups[selected_group_name]
        found_players = {}
        missing_players = []

        for display_name, aliases in selected_group["players"].items():
            player = find_player_by_aliases(fc26, aliases)
            if player is None:
                missing_players.append(display_name)
            else:
                found_players[display_name] = player

        if missing_players:
            st.warning(
                "Players not found: " + ", ".join(missing_players)
            )

        render_comparison(
            found_players,
            selected_group["stats"],
            selected_group_name
        )

    else:
        options = player_options(fc26)

        number_to_compare = st.radio(
            "Number of Players",
            [2, 3],
            horizontal=True
        )

        selection_columns = st.columns(number_to_compare)
        selected_names = []

        default_indices = [0, 1, 2]

        for index, column in enumerate(selection_columns):
            with column:
                selected_name = st.selectbox(
                    f"Player {index + 1}",
                    options,
                    index=min(default_indices[index], len(options) - 1),
                    key=f"custom_player_{index}"
                )
                selected_names.append(selected_name)

        selected_names = list(dict.fromkeys(selected_names))

        if len(selected_names) < number_to_compare:
            st.warning("Choose different players for the comparison.")
        else:
            default_stats = [
                stat
                for stat in [
                    "overall", "pace", "shooting", "passing",
                    "dribbling", "defending", "physic"
                ]
                if stat in fc26.columns
            ]

            selected_stats = st.multiselect(
                "Select Comparison Statistics",
                options=["overall"] + existing_stat_columns,
                default=default_stats,
                format_func=format_stat_name
            )

            custom_players = {
                name: get_player_by_short_name(fc26, name)
                for name in selected_names
            }

            custom_players = {
                name: player
                for name, player in custom_players.items()
                if player is not None
            }

            render_comparison(
                custom_players,
                selected_stats,
                "Custom Player Comparison"
            )


# ==========================================================
# Search Player
# ==========================================================

elif page == "🔎 Search Player":
    st.title("🔎 Search Player")

    options = player_options(fc26)
    selected_name = st.selectbox("Choose Player", options)
    player = get_player_by_short_name(fc26, selected_name)

    if player is not None:
        card_column, chart_column = st.columns([1, 2])

        with card_column:
            display_basic_player_card(player)

        available_profile_stats = [
            stat
            for stat in [
                "pace", "shooting", "passing", "dribbling",
                "defending", "physic"
            ]
            if stat in player.index and pd.notna(player.get(stat))
        ]

        with chart_column:
            if len(available_profile_stats) >= 3:
                labels = [format_stat_name(stat) for stat in available_profile_stats]
                values = [float(player[stat]) for stat in available_profile_stats]
                angles = np.linspace(
                    0,
                    2 * np.pi,
                    len(values),
                    endpoint=False
                ).tolist()
                angles += angles[:1]
                values += values[:1]

                fig_profile, ax_profile = plt.subplots(
                    figsize=(7, 7),
                    subplot_kw={"polar": True}
                )
                ax_profile.plot(angles, values, linewidth=2)
                ax_profile.fill(angles, values, alpha=0.12)
                ax_profile.set_xticks(angles[:-1])
                ax_profile.set_xticklabels(labels)
                ax_profile.set_ylim(0, 100)
                ax_profile.set_title(
                    f"{selected_name} Player Profile",
                    fontsize=16,
                    fontweight="bold",
                    pad=20
                )
                plt.tight_layout()
                st.pyplot(fig_profile)
                plt.close(fig_profile)

        detail_columns = [
            column
            for column in [
                "long_name", "club_name", "league_name",
                "nationality_name", "player_positions", "age",
                "overall", "potential", "value_eur", "wage_eur"
            ] + stat_columns
            if column in fc26.columns
        ]

        player_details = pd.DataFrame(
            {
                "Attribute": [format_stat_name(column) for column in detail_columns],
                "Value": [player.get(column) for column in detail_columns]
            }
        )

        st.subheader("Full Player Details")
        st.dataframe(player_details, use_container_width=True, hide_index=True)


# ==========================================================
# Filters
# ==========================================================

elif page == "🎯 Filters":
    st.title("🎯 Player Filters")

    filtered_data = fc26.copy()

    filter_columns = st.columns(3)

    with filter_columns[0]:
        if "league_name" in fc26.columns:
            leagues = sorted(fc26["league_name"].dropna().astype(str).unique())
            selected_leagues = st.multiselect("League", leagues)
            if selected_leagues:
                filtered_data = filtered_data[
                    filtered_data["league_name"].isin(selected_leagues)
                ]

    with filter_columns[1]:
        if "club_name" in fc26.columns:
            clubs = sorted(filtered_data["club_name"].dropna().astype(str).unique())
            selected_clubs = st.multiselect("Club", clubs)
            if selected_clubs:
                filtered_data = filtered_data[
                    filtered_data["club_name"].isin(selected_clubs)
                ]

    with filter_columns[2]:
        if "nationality_name" in fc26.columns:
            nationalities = sorted(
                filtered_data["nationality_name"].dropna().astype(str).unique()
            )
            selected_nationalities = st.multiselect(
                "Nationality",
                nationalities
            )
            if selected_nationalities:
                filtered_data = filtered_data[
                    filtered_data["nationality_name"].isin(selected_nationalities)
                ]

    if "age" in fc26.columns:
        minimum_age = int(fc26["age"].dropna().min())
        maximum_age = int(fc26["age"].dropna().max())
        selected_age = st.slider(
            "Age Range",
            minimum_age,
            maximum_age,
            (minimum_age, maximum_age)
        )
        filtered_data = filtered_data[
            filtered_data["age"].between(selected_age[0], selected_age[1])
        ]

    stat_for_filter = st.selectbox(
        "Rank Filtered Players By",
        ["overall"] + existing_stat_columns,
        format_func=format_stat_name
    )

    filtered_top = get_top_players(
        filtered_data,
        stat_for_filter,
        min(10, len(filtered_data))
    )

    st.write(f"**Matching Players:** {len(filtered_data):,}")

    if filtered_top.empty:
        st.warning("No players match the selected filters.")
    else:
        display_players(
            filtered_data,
            stat_for_filter,
            min(10, len(filtered_data))
        )

        display_bar_chart(
            filtered_top,
            stat_for_filter,
            f"Filtered Players by {format_stat_name(stat_for_filter)}"
        )


# ==========================================================
# Insights
# ==========================================================

elif page == "💡 Insights":
    st.title("💡 Dataset Insights")

    insight_stats = {
        "Highest Overall": "overall",
        "Fastest Player": "pace",
        "Best Shooting": "shooting",
        "Best Passing": "passing",
        "Best Dribbling": "dribbling",
        "Best Defending": "defending",
        "Best Physical": "physic",
        "Best GK Reflexes": "goalkeeping_reflexes"
    }

    available_insights = [
        (label, stat)
        for label, stat in insight_stats.items()
        if stat in fc26.columns
    ]

    for start_index in range(0, len(available_insights), 4):
        columns = st.columns(4)

        for column, (label, stat) in zip(
            columns,
            available_insights[start_index:start_index + 4]
        ):
            top_player = get_top_players(fc26, stat, 1)

            with column:
                if not top_player.empty:
                    player = top_player.iloc[0]
                    st.metric(label, int(player[stat]))
                    st.write(f"**{player['short_name']}**")

    st.divider()

    if "value_eur" in fc26.columns:
        most_valuable = fc26.dropna(subset=["value_eur"]).nlargest(10, "value_eur")
        st.subheader("Most Valuable Players")
        st.dataframe(
            most_valuable[
                [
                    column
                    for column in [
                        "short_name", "club_name", "overall",
                        "age", "value_eur"
                    ]
                    if column in most_valuable.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    if "league_name" in fc26.columns and "overall" in fc26.columns:
        league_summary = (
            fc26.dropna(subset=["league_name", "overall"])
            .groupby("league_name")
            .agg(
                Players=("short_name", "count"),
                Average_Overall=("overall", "mean")
            )
            .sort_values("Average_Overall", ascending=False)
            .head(10)
            .round(2)
        )

        st.subheader("Top Leagues by Average Overall")
        st.dataframe(league_summary, use_container_width=True)
