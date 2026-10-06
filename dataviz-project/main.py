#!/usr/bin/env python
import ast
import base64
import os
from functools import lru_cache
from io import BytesIO
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # off-screen rendering for tooltip scatterplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    RadioGroup,
    RadioButtonGroup,
    CheckboxButtonGroup,
    Div,
    Button,
    Select,
    ColumnDataSource,
    FixedTicker,
    HoverTool,
    Range1d,
    LinearColorMapper,
    ColorBar,
    BasicTicker,
    GeoJSONDataSource,
    LayoutDOM,
    Widget,
    InlineStyleSheet,
    Title,
)
from bokeh.layouts import column, row
from bokeh.palettes import Cividis256, PuOr11
from bokeh.themes import Theme


SLIDE_WIDTH = 1200
CONTENT_WIDTH = 1160 

# ==============================================================================
# MAP LOADING AND PROJECTION (Native Robinson via +proj=robin)
# ==============================================================================
base_dir = os.path.dirname(__file__)
WORLD_PATH = os.path.join(base_dir, "data", "world-countries.json")

try:
    print(f"Attempting to load map from: {WORLD_PATH}")
    _world_raw = gpd.read_file(WORLD_PATH)
    WORLD_GEO = _world_raw.to_crs("+proj=robin")
    WORLD_GEO["geometry"] = WORLD_GEO.geometry.simplify(10000, preserve_topology=True)
    print("Map successfully loaded and projected!")
except Exception as e:
    print(f"ERROR loading map: {e}")
    try:
        WORLD_GEO = _world_raw.to_crs("EPSG:4326")
        WORLD_GEO["geometry"] = WORLD_GEO.geometry.simplify(0.1, preserve_topology=True)
    except Exception:
        WORLD_GEO = None


WIDGET_CENTER_CSS = """/*center*/
:host { text-align: center; }
.bk-btn { justify-content: center; text-align: center; }
.bk-btn-group { justify-content: center; }
.bk-input-group { justify-content: center; text-align: center; }
.bk-input-group label { text-align: center; }
.bk-slider-title { justify-content: center; text-align: center; }
"""


def _merge_styles(model, extra):
    current = model.styles if isinstance(model.styles, dict) else {}
    model.styles = {**current, **extra}


def center_everything(root, is_root=False):
    from bokeh.models import Column, Row

    for model in root.references():
        if not isinstance(model, LayoutDOM):
            continue

        if isinstance(model, Div):
            if model.sizing_mode in ("stretch_width", "stretch_both"):
                model.sizing_mode = None
                model.width = CONTENT_WIDTH
            _merge_styles(model, {"text-align": "center"})
        elif isinstance(model, Column):
            _merge_styles(model, {"align-items": "center"})
        elif isinstance(model, Row):
            _merge_styles(model, {"justify-content": "center"})

        if isinstance(model, Widget) and not any(
            isinstance(sheet, InlineStyleSheet) and "/*center*/" in sheet.css
            for sheet in model.stylesheets
        ):
            model.stylesheets = [
                *model.stylesheets,
                InlineStyleSheet(css=WIDGET_CENTER_CSS),
            ]

        if model is not root:
            model.align = "center"


class InteractivePresentation:

    theme_path = os.path.join(base_dir, "theme.yaml")
    curdoc().theme = Theme(filename=theme_path)

    def __init__(self, filename="wines_enhanced.csv"):
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(base_dir, "data", filename)

        # Load dataset
        self.df = pd.read_csv(data_path)
        self.current_slide = 0
        self.total_slides = 7
        self.slides = []
        self.auto_play = False
        self.auto_play_callback = None

        self.create_slides()
        self.create_navigation()
        self.create_layout()

    def stack(self, rows):
        """Stack rows of items vertically. A row with several items becomes a Row."""
        return column(
            *[
                row(*r, width=CONTENT_WIDTH, styles={"justify-content": "center"})
                for r in rows
            ],
            sizing_mode="stretch_width",
        )

    def create_navigation(self):
        """Create themed navigation controls matching wine palette and centering rules"""

        self.style_div = Div(
            text="""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Lusitana:wght@400;700&display=swap');

            /* Global Page Background & Centering Overrides */
            html, body {
                background-color: #FDFCF7 !important;
                font-family: 'Lusitana', Georgia, serif !important;
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                min-height: 100vh !important;
            }

            /* Center Bokeh container and all inner layout components */
            .bk-root {
                background-color: transparent !important;
                width: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                margin: 0 auto !important;
            }

            .bk-root > .bk,
            .bk-root .bk-Column,
            .bk-root .bk-Row,
            .bk-root .bk-layout-fixed,
            .bk-root .bk-content,
            .bk-root .bk-Control {
                margin-left: auto !important;
                margin-right: auto !important;
                align-self: center !important;
            }

            /* Force text-align centering for all inner text and div containers */
            .bk-root, .bk-root *, .bk-content, .bk-content * {
                text-align: center !important;
            }

            .bk-root h1, .bk-root h2, .bk-root h3, .bk-root h4, .bk-root p, 
            .bk-root div, .bk-root span, .bk-root td, .bk-root th, 
            .bk-root ul, .bk-root li {
                text-align: center !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }

            .bk-root div.bk > div {
                margin-left: auto !important;
                margin-right: auto !important;
                text-align: center !important;
            }

            .bk-root input, .bk-root select, .bk-root button, .bk-root textarea {
                font-family: 'Lusitana', Georgia, serif !important;
            }

            /* Theme Buttons */
            .theme-btn button.bk-btn {
                font-family: 'Lusitana', Georgia, serif !important;
                font-weight: bold !important;
                background-color: #FFFFFF !important;
                color: #AF1B3F !important;
                border: 1.5px solid #AF1B3F !important;
                border-radius: 6px !important;
                padding: 6px 14px !important;
                font-size: 13px !important;
                box-shadow: 0 2px 4px rgba(175, 27, 63, 0.05) !important;
                transition: all 0.2s ease-in-out !important;
            }

            .theme-btn button.bk-btn:hover:not(:disabled) {
                background-color: #AF1B3F !important;
                color: #FFFFFF !important;
                border-color: #AF1B3F !important;
                cursor: pointer !important;
                box-shadow: 0 3px 6px rgba(175, 27, 63, 0.15) !important;
            }

            .theme-btn button.bk-btn:disabled {
                background-color: #FDFCF7 !important;
                color: #C2B8B2 !important;
                border-color: #E2D7C3 !important;
                opacity: 0.65 !important;
                cursor: not-allowed !important;
            }

            /* Active / Playing Button Accent */
            .theme-btn-active button.bk-btn {
                background-color: #AF1B3F !important;
                color: #FFFFFF !important;
                border-color: #AF1B3F !important;
            }

            /* Dropdown Selector Styling */
            .theme-select select {
                font-family: 'Lusitana', Georgia, serif !important;
                background-color: #FFFFFF !important;
                color: #211B18 !important;
                border: 1.5px solid #AF1B3F !important;
                border-radius: 6px !important;
                padding: 5px 10px !important;
                font-size: 13px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
                transition: border-color 0.2s ease !important;
            }

            .theme-select select:focus {
                border-color: #AF1B3F !important;
                outline: none !important;
            }

            .theme-select label {
                font-family: 'Lusitana', Georgia, serif !important;
                color: #AF1B3F !important;
                font-weight: bold !important;
                font-size: 13px !important;
                letter-spacing: 0.3px !important;
                text-align: center !important;
                display: block !important;
            }
        </style>
        """,
            width=0,
            height=0,
        )

        self.prev_button = Button(
            label="◀ Previous", width=105, css_classes=["theme-btn"], align="center"
        )
        self.next_button = Button(
            label="Next ▶", width=105, css_classes=["theme-btn"], align="center"
        )
        self.home_button = Button(
            label="🏠 Home", width=100, css_classes=["theme-btn"], align="center"
        )

        self.play_button = Button(
            label="▶ Auto Play", width=115, css_classes=["theme-btn"], align="center"
        )

        slide_options = [
            (str(i), f"Slide {i + 1}: {self.get_slide_title(i)}")
            for i in range(self.total_slides)
        ]
        self.slide_select = Select(
            title="Jump to:",
            value="0",
            options=slide_options,
            width=280,
            css_classes=["theme-select"],
            align="center",
        )

        self.progress_div = Div(
            text=self.get_progress_html(),
            width=220,
            align="center",
        )

        self.prev_button.on_click(self.prev_slide)
        self.next_button.on_click(self.next_slide)
        self.home_button.on_click(self.go_home)
        self.play_button.on_click(self.toggle_auto_play)
        self.slide_select.on_change("value", self.jump_to_slide)

    def get_slide_title(self, index):
        """Get title for each slide."""
        titles = [
            "Welcome",
            "Price vs Rating",
            "Price and Rating by Provenance",
            "Correlation Matrix",
            "Food Pairing Radar",
            "Blend vs Variety Violin",
            "Conclusions",
        ]
        return titles[index] if index < len(titles) else f"Slide {index + 1}"

    def get_progress_html(self):
        """Generate progress bar HTML matching wine theme"""
        progress_pct = ((self.current_slide + 1) / self.total_slides) * 100
        return f"""
        <div style="font-family: 'Lusitana', Georgia, serif; text-align: center; padding: 2px 5px; margin: 0 auto;">
            <div style="font-size: 12px; font-weight: bold; color: #AF1B3F; letter-spacing: 0.5px; margin-bottom: 4px; text-align: center;">
                SLIDE {self.current_slide + 1} OF {self.total_slides}
            </div>
            <div style="width: 100%; background-color: #FFFFFF; border: 1px solid #E2D7C3; height: 12px; border-radius: 6px; padding: 1px; box-sizing: border-box; margin: 0 auto;">
                <div style="width: {progress_pct}%; background-color: #AF1B3F; height: 100%; border-radius: 4px; transition: width 0.3s ease-in-out;"></div>
            </div>
        </div>
        """

    def create_slides(self):
        """Create the merged presentation slides."""
        self.slides = [
            self.create_slide_title(),                
            self.create_slide_1_price_vs_rating(),    
            self.create_slide_2_map(),   
            self.create_slide_3_correlation(),            
            self.create_slide_4_radar(),         
            self.create_slide_5_violin(),         
            self.create_slide_6_conclusions(),
        ]

    def create_slide_title(self):
        """Slide 1: Title Card"""
        title_banner = Div(
            text="""
        <div style="position: relative; text-align: center; background-color: #FFFFFF; border: 1px solid #E2D7C3; border-left: 6px solid #AF1B3F; padding: 100px 40px; border-radius: 12px; margin: 40px auto 0 auto; overflow: hidden; box-shadow: 0 4px 15px rgba(175, 27, 63, 0.05); max-width: 1100px;">
        
            <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; border-radius: 50%; border: 12px solid rgba(175, 27, 63, 0.08); box-shadow: inset 0 0 15px rgba(175, 27, 63, 0.05); pointer-events: none;"></div>
            <div style="position: absolute; top: -20px; right: -20px; width: 120px; height: 120px; border-radius: 50%; border: 6px solid rgba(255, 188, 66, 0.25); pointer-events: none;"></div>
        
            <div style="position: absolute; bottom: -50px; left: -30px; width: 160px; height: 160px; border-radius: 50%; border: 10px solid rgba(33, 131, 128, 0.1); transform: rotate(-15deg); pointer-events: none;"></div>

            <h1 style="position: relative; z-index: 1; font-size: 48px; color: #AF1B3F; font-family: 'Lusitana', serif; margin: 0 0 20px 0; font-weight: 700; line-height: 1.2; text-align: center;">
                Wine: should you splash out on the bottle?
            </h1>
            <p style="position: relative; z-index: 1; font-size: 22px; color: #5C4A42; font-style: italic; margin: 0; font-weight: 400; text-align: center;">
                Visualizing relationship between price, ratings and wine features
            </p>
        </div>
        """,
            sizing_mode="stretch_width",
            height=340,
            align="center",
        )

        return self.stack([[title_banner]])

    def create_slide_1_price_vs_rating(self):
        """Slide 1: Scatter plot of price vs rating with a robust linear regression fit (Huber, on log price)"""
         
        title = Div(
            text="""
            <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif; margin-bottom: 5px;">
                Average Rating vs. Price
            </h2>
            <p style="text-align: center; color: #5C4A42; font-size: 15px; margin-top: 0; margin-bottom: 12px;">
                Select a wine category below to inspect rating and price distribution alongside a robust linear regression fit.
            </p>
            
            <!-- Fixed Model Description across all wine categories -->
            <div style="max-width: 1150px; margin: 0 auto 10px auto; padding: 10px 16px; background: #FDFCF7; border: 1px solid #EAE5DC; border-radius: 6px; font-family: sans-serif; font-size: 14px; color: #4A3B32; line-height: 1.5; text-align: center;">
                A robust linear fit with Huber loss was chosen to capture the main trend, reducing the impact of outliers.
            </div>
            """,
            sizing_mode="stretch_width",
            align="center",
        )

        rng = np.random.default_rng(123)
        df = self.df.copy()
        df["Rating_jittered"] = df["Rating"] + rng.normal(scale=0.02, size=len(df))

        kinds = ["Red", "White", "Rose", "Sparkling"]

        wine_colors = {
            "Red": "#AF1B3F",
            "White": "#FFBC42",
            "Rose": "#C99DA3",
            "Sparkling": "#218380",
        }

        initial_kind = "Red"
        initial_df = df[df["Kind"] == initial_kind]
        source = ColumnDataSource(data=ColumnDataSource.from_df(initial_df))

        def compute_robust_fit(sub_df, grid_points=100):
            valid = sub_df.dropna(subset=["Rating_jittered", "Price"])
            valid = valid[valid["Price"] > 0]

            if len(valid) < 3:
                return dict(x_fit=[], y_fit=[])

            log_price = np.log10(valid["Price"].values)
            rating = valid["Rating_jittered"].values

            try:
                model = sm.RLM(
                    rating,
                    sm.add_constant(log_price),
                    M=sm.robust.norms.HuberT(),
                ).fit()

                grid = np.linspace(log_price.min(), log_price.max(), grid_points)
                y_fit = model.predict(sm.add_constant(grid))
                return dict(x_fit=10**grid, y_fit=y_fit)
            except Exception:
                return dict(x_fit=[], y_fit=[])

        fit_source = ColumnDataSource(data=compute_robust_fit(initial_df))

        p = figure(
            width=1150,
            height=650,
            title=f"{initial_kind} Wine",
            x_axis_type="log",
            x_axis_label="Price (€, log scale)",
            y_axis_label="Average Rating",
            tools="pan,wheel_zoom,reset,hover,save",
            align="center",
        )

        p.title.align = "left"

        p.background_fill_color = "#FFFFFF"
        p.background_fill_alpha = 1.0
        p.border_fill_color = "#FDFCF7"
        p.grid.grid_line_color = "#EAE5DC"
        p.grid.grid_line_alpha = 0.8

        scatter = p.scatter(
            x="Price",
            y="Rating_jittered",
            source=source,
            color=wine_colors[initial_kind],
            alpha=0.7,
            size=8.5,
            line_color="#FFFFFF",
            line_width=0.5,
        )

        fit_line = p.line(
            x="x_fit",
            y="y_fit",
            source=fit_source,
            color="#000000",
            line_width=2.5
        )

        x_min = df.loc[df["Price"] > 0, "Price"].min()
        x_max = df["Price"].max()
        x_pad = (x_max / x_min) ** 0.03
        p.x_range = Range1d(x_min / x_pad, x_max * x_pad)

        y_min = df["Rating_jittered"].min()
        y_max = df["Rating_jittered"].max()
        y_pad = 0.05 * (y_max - y_min)
        p.y_range = Range1d(y_min - y_pad, y_max + y_pad)

        hover = p.select_one(HoverTool)
        hover.renderers = [scatter]

        compact_tooltip_html = """
        <div style="max-width: 190px; max-height: 110px; overflow: hidden; font-size: 11px; padding: 6px 8px; border-radius: 4px; background: #FFFFFF; border: 1.5px solid #AF1B3F; box-shadow: 0 2px 6px rgba(0,0,0,0.1); line-height: 1.4; text-align: center; font-family: 'Lusitana', Georgia, serif;">
            <div style="font-weight: bold; color: #AF1B3F; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #E2D7C3; padding-bottom: 2px; margin-bottom: 4px; text-align: center;">
                @Label
            </div>
            <div style="color: #5C4A42; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: center;">
                @Country &bull; @Region
            </div>
            <div style="color: #5C4A42; text-align: center;"><b>Vintage:</b> @Vintage</div>
            <div style="margin-top: 3px; border-top: 1px dashed #E2D7C3; padding-top: 3px; text-align: center;">
                <b style="color: #AF1B3F;">@Rating pts | Number of Ratings: @NumberOfRatings </b> 
            </div>
                <b style="color: #218380;">€@Price{0.00}</b>
            </div>
        </div>
        """

        hover.tooltips = compact_tooltip_html
        hover.point_policy = "snap_to_data"
        hover.mode = "mouse"

        toggle = RadioGroup(
            labels=kinds,
            active=0,
            inline=True,
            width=400,
            align="center",
        )

        def update_plot(attr, old, new):
            selected_kind = kinds[new]
            p.title.text = f"{selected_kind} Wine"

            new_df = df[df["Kind"] == selected_kind]
            source.data = ColumnDataSource.from_df(new_df)
            fit_source.data = compute_robust_fit(new_df)

            scatter.glyph.fill_color = wine_colors[selected_kind]
            scatter.glyph.line_color = wine_colors[selected_kind]

        toggle.on_change("active", update_plot)

        description = Div(text = 
            """
             <div style="max-width: 1150px; margin: 0 auto 10px auto; padding: 10px 16px; background: #FDFCF7; border: 1px solid #EAE5DC; border-radius: 6px; font-family: sans-serif; font-size: 14px; color: #4A3B32; line-height: 1.5; text-align: center;">
                <b style="color: #AF1B3F;"> Alt-text:</b> The figure shows a scatter plot where each point represents a wine. The x axis is the wine's price on a log scale; the y axis is the wine's rating. A toggle allows the user to select a wine type. Each scatter is accompanied by a robust linear fit highlighting the general trend. For all wine types, an increasing trend is observed, i.e. as the wine's price increases, the rating also increases.
                However, for red wines, white wines, and rose wines, some outlying points, with a rating much lower than expected given their price.
            </div>
            """
        )

        return self.stack([[title], [toggle], [p], [description]])

    def create_slide_2_map(self):
        """Slide 2: Geographic Wine Analysis (Price & Rating World Maps - Vertically Stacked)"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif; margin-bottom: 5px;">
            Geographic Wine Analysis: Price & Rating by Country
        </h2>
        <p style="text-align: center; color: #5C4A42; font-family: 'Lusitana', Georgia, serif; font-size: 14px; margin-top: 0; margin-bottom: 15px;">
            Select a wine category below to update the maps. Hover over any country to inspect detailed price and rating statistics. Note that the limits of the color scale change according to the selected wine type.
        </p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        if WORLD_GEO is None or WORLD_GEO.empty:
            error_div = Div(
                text="<p style='color: #AF1B3F; text-align: center; font-family: \"Lusitana\", serif;'>Unable to load world country geometries.</p>",
                sizing_mode="stretch_width",
                align="center",
            )
            return self.stack([[title], [error_div]])

        options = ["Global", "Red", "White", "Rose", "Sparkling"]

        # Ocean = the plot background behind the country polygons
        OCEAN_COLOR = "#FFFFFF"


        bounds = WORLD_GEO.total_bounds
        geo_width = bounds[2] - bounds[0]
        geo_height = bounds[3] - bounds[1]
        map_aspect = geo_width / geo_height if geo_height != 0 else 2.0

        x_bounds = (float(bounds[0]), float(bounds[2]))
        y_bounds = (float(bounds[1]), float(bounds[3]))

        @lru_cache(maxsize=None)
        def get_price_geojson(wine_type):
            df_sub = self.df if wine_type == "Global" else self.df[self.df["Kind"] == wine_type]
            stats = (
                df_sub.groupby("Country")["Price"]
                .agg(
                    Price_Mean="mean",
                    Price_Median="median",
                    Price_Min="min",
                    Price_Max="max",
                    Count="count",
                )
                .round(2)
                .reset_index()
            )

            merged = WORLD_GEO.merge(stats, how="left", left_on="name", right_on="Country")
            merged[
                ["Price_Mean", "Price_Median", "Price_Min", "Price_Max", "Count"]
            ] = merged[
                ["Price_Mean", "Price_Median", "Price_Min", "Price_Max", "Count"]
            ].fillna("N/A")
            return merged[
                [
                    "geometry",
                    "name",
                    "Price_Mean",
                    "Price_Median",
                    "Price_Min",
                    "Price_Max",
                    "Count",
                ]
            ].to_json()

        @lru_cache(maxsize=None)
        def get_rating_geojson(wine_type):
            df_sub = self.df if wine_type == "Global" else self.df[self.df["Kind"] == wine_type]
            stats = (
                df_sub.groupby("Country")["Rating"]
                .agg(
                    Rating_Mean="mean",
                    Rating_Median="median",
                    Rating_Min="min",
                    Rating_Max="max",
                    Count="count",
                )
                .round(2)
                .reset_index()
            )

            merged = WORLD_GEO.merge(stats, how="left", left_on="name", right_on="Country")
            merged[
                ["Rating_Mean", "Rating_Median", "Rating_Min", "Rating_Max", "Count"]
            ] = merged[
                ["Rating_Mean", "Rating_Median", "Rating_Min", "Rating_Max", "Count"]
            ].fillna("N/A")
            return merged[
                [
                    "geometry",
                    "name",
                    "Rating_Mean",
                    "Rating_Median",
                    "Rating_Min",
                    "Rating_Max",
                    "Count",
                ]
            ].to_json()

        @lru_cache(maxsize=None)
        def get_hatch_geojson(wine_type):
            """Geometry only, for countries with no wines in the selection."""
            df_sub = self.df if wine_type == "Global" else self.df[self.df["Kind"] == wine_type]
            present = df_sub["Country"].unique()
            return WORLD_GEO.loc[~WORLD_GEO["name"].isin(present), ["geometry"]].to_json()

        price_source = GeoJSONDataSource(geojson=get_price_geojson("Global"))
        hatch_source = GeoJSONDataSource(geojson=get_hatch_geojson("Global"))
        rating_source = GeoJSONDataSource(geojson=get_rating_geojson("Global"))

        p_means = self.df.groupby("Country")["Price"].mean()
        price_mapper = LinearColorMapper(
            palette=Cividis256,
            low=float(p_means.min()) if not p_means.empty else 0.0,
            high=float(p_means.max()) if not p_means.empty else 100.0,
            nan_color="#EBEBEB",
        )

        r_means = self.df.groupby("Country")["Rating"].mean()
        r_min = float(r_means.min()) if not r_means.empty else 3.0
        r_max = float(r_means.max()) if not r_means.empty else 5.0

        rating_mapper = LinearColorMapper(
            palette=PuOr11,
            low=r_min,
            high=r_max,
            nan_color="#EBEBEB",
        )

        p_price = figure(
            width=1160,
            height=500,
            title="Average Wine Price by Country (€)",
            tools="pan,wheel_zoom,reset,save",
            x_axis_location=None,
            y_axis_location=None,
            x_range=x_bounds,
            y_range=y_bounds,
            match_aspect=True,
            aspect_ratio=map_aspect,
            align="center",
        )
        p_price.title.align = "left"
        p_price.background_fill_color = OCEAN_COLOR
        p_price.border_fill_color = "#FDFCF7"
        p_price.grid.grid_line_color = None

        price_patches = p_price.patches(
            "xs",
            "ys",
            source=price_source,
            fill_color={"field": "Price_Mean", "transform": price_mapper},
            fill_alpha=0.9,
            line_color="#000000",
            line_width=0.6,
            nonselection_fill_alpha=0.9,
            nonselection_fill_color={"field": "Price_Mean", "transform": price_mapper},
        )

        # Texture overlay: one constant pattern (fast), shared by both maps.
        p_price.patches(
            "xs",
            "ys",
            source=hatch_source,
            fill_alpha=0,
            line_alpha=0,
            hatch_pattern="/",
            hatch_color="#B8B2A7",
            hatch_alpha=0.6,
            hatch_scale=7,
            hatch_weight=0.8,
        )

        cb_price = ColorBar(
            color_mapper=price_mapper,
            width=14,
            location=(0, 0),
            title="€",
            title_text_font_size="10pt",
            label_standoff=4,
        )
        p_price.add_layout(cb_price, "right")

        price_hover = HoverTool(
            tooltips="""
            <div style="font-family: 'Lusitana', Georgia, serif; padding: 6px 10px; font-size: 12px; background: #FFFFFF; border: 1.5px solid #AF1B3F; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center;">
                <strong style="color: #AF1B3F; font-size: 13px; text-align: center;">📍 @name</strong><br/>
                <div style="margin-top: 4px; line-height: 1.4; color: #5C4A42; text-align: center;">
                    <b>Mean Price:</b> €@Price_Mean<br/>
                    <b>Median Price:</b> €@Price_Median<br/>
                    <b>Min:</b> €@Price_Min | <b>Max:</b> €@Price_Max<br/>
                    <b>Wines Count:</b> @Count
                </div>
            </div>
            """,
            renderers=[price_patches],
            attachment="above",
            mode="mouse",
        )
        p_price.add_tools(price_hover)

        p_rating = figure(
            width=1160,
            height=500,
            title="Average Wine Rating by Country",
            tools="pan,wheel_zoom,reset,save",
            x_axis_location=None,
            y_axis_location=None,
            x_range=p_price.x_range,
            y_range=p_price.y_range,
            match_aspect=True,
            aspect_ratio=map_aspect,
            align="center",
        )
        p_rating.title.align = "left"
        p_rating.background_fill_color = OCEAN_COLOR
        p_rating.border_fill_color = "#FDFCF7"
        p_rating.grid.grid_line_color = None

        rating_patches = p_rating.patches(
            "xs",
            "ys",
            source=rating_source,
            fill_color={"field": "Rating_Mean", "transform": rating_mapper},
            fill_alpha=0.9,
            line_color="#000000",
            line_width=0.6,
            nonselection_fill_alpha=0.9,
            nonselection_fill_color={"field": "Rating_Mean", "transform": rating_mapper},
        )

        # Texture overlay: one constant pattern (fast), shared by both maps.
        p_rating.patches(
            "xs",
            "ys",
            source=hatch_source,
            fill_alpha=0,
            line_alpha=0,
            hatch_pattern="/",
            hatch_color="#B8B2A7",
            hatch_alpha=0.6,
            hatch_scale=7,
            hatch_weight=0.8,
        )

        cb_rating = ColorBar(
            color_mapper=rating_mapper,
            width=14,
            location=(0, 0),
            title="⭐",
            title_text_font_size="10pt",
            label_standoff=4,
        )
        p_rating.add_layout(cb_rating, "right")

        rating_hover = HoverTool(
            tooltips="""
            <div style="font-family: 'Lusitana', Georgia, serif; padding: 6px 10px; font-size: 12px; background: #FFFFFF; border: 1.5px solid #AF1B3F; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center;">
                <strong style="color: #AF1B3F; font-size: 13px; text-align: center;">📍 @name</strong><br/>
                <div style="margin-top: 4px; line-height: 1.4; color: #5C4A42; text-align: center;">
                    <b>Mean Rating:</b> @Rating_Mean ⭐<br/>
                    <b>Median Rating:</b> @Rating_Median ⭐<br/>
                    <b>Min:</b> @Rating_Min ⭐ | <b>Max:</b> @Rating_Max ⭐<br/>
                    <b>Wines Count:</b> @Count
                </div>
            </div>
            """,
            renderers=[rating_patches],
            attachment="above",
            mode="mouse",
        )
        p_rating.add_tools(rating_hover)

        def country_mean_range(column, wine_type, default):
            df_sub = self.df if wine_type == "Global" else self.df[self.df["Kind"] == wine_type]
            means = df_sub.groupby("Country")[column].mean().dropna()
            if means.empty:
                return default
            low, high = float(means.min()), float(means.max())
            if low == high:
                high = low + 1e-6
            return low, high

        select_wine = Select(
            title="Filter Wine Category:",
            value="Global",
            options=options,
            width=260,
            css_classes=["theme-select"],
            align="center",
            margin=(0, 0, 15, 0),
        )

        def update_maps(attr, old, new):
            price_source.geojson = get_price_geojson(new)
            rating_source.geojson = get_rating_geojson(new)
            hatch_source.geojson = get_hatch_geojson(new)

            price_mapper.low, price_mapper.high = country_mean_range(
                "Price", new, (price_mapper.low, price_mapper.high)
            )
            rating_mapper.low, rating_mapper.high = country_mean_range(
                "Rating", new, (rating_mapper.low, rating_mapper.high)
            )

            title_suffix = f"({new})" if new != "Global" else "(All Wines)"
            p_price.title.text = f"Average Wine Price by Country {title_suffix}"
            p_rating.title.text = f"Average Wine Rating by Country {title_suffix}"

        select_wine.on_change("value", update_maps)

        return self.stack([
            [title],
            [select_wine],
            [p_price],
            [p_rating],
        ])

    def create_slide_3_correlation(self):
        """Lower-triangular correlation heatmap; hovering a cell shows scatterplot."""
        COLOR_BLUE = "#0173B2"

        colorblind_heatmap_colors = [
            "#0173B2",
            "#56B4E9",
            "#94d4f7",
            "#d1eaf8",
            "#f7f7f7",
            "#fbe3c3",
            "#f8be81",
            "#DE8F05",
            "#b87000",
        ]
        color_mapper = LinearColorMapper(
            palette=colorblind_heatmap_colors, low=-1, high=1
        )

        numeric_cols = ["Rating", "Log Price", "ABV", "Vintage"]
        data = self.df[["Rating", "Price", "ABV", "Vintage"]].apply(pd.to_numeric, errors="coerce")
        data.loc[data["Price"] <= 0, "Price"] = np.nan
        data["Log Price"] = np.log10(data["Price"])
        data = data[numeric_cols]
        corr_df = data.corr()

        lower_mask = np.tril(np.ones(corr_df.shape, dtype=bool), k=0)
        corr_lower = corr_df.where(lower_mask)

        corr_unstacked = corr_lower.stack().reset_index()
        corr_unstacked.columns = ["feature_x", "feature_y", "value"]

        labels = list(corr_df.columns)
        n = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        rev_labels = list(reversed(labels))
        rev_label_to_idx = {label: i for i, label in enumerate(rev_labels)}

        corr_unstacked["x_num"] = corr_unstacked["feature_x"].map(label_to_idx) + 0.5
        corr_unstacked["y_num"] = (
            corr_unstacked["feature_y"].map(rev_label_to_idx) + 0.5
        )

        plots_dict = {}
        rng = np.random.default_rng(123)
        JITTER_SD = 0.02
        JITTERED_VARS = ("Rating", "Vintage")
        for _, r in corr_unstacked.iterrows():
            c1, c2 = r["feature_x"], r["feature_y"]

            fig_scatter, ax = plt.subplots(figsize=(4.5, 3.6), dpi=120)

            pair = data[[c1, c2]].dropna() if c1 != c2 else data[[c1]].dropna()

            x_label = c1
            y_label = c2
            x_data = pair[c1]
            y_data = pair[c2]

            if c1 != c2:
                if c1 in JITTERED_VARS:
                    x_data = x_data + rng.normal(scale=JITTER_SD, size=len(x_data))
                if c2 in JITTERED_VARS:
                    y_data = y_data + rng.normal(scale=JITTER_SD, size=len(y_data))

            ax.scatter(
                x_data, y_data, alpha=0.5, s=7, color=COLOR_BLUE, edgecolors="none"
            )
            ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
            ax.set_title(
                f"Scatterplot: {x_label} vs {y_label}",
                fontsize=11,
                pad=10,
                fontweight="bold",
                loc="left",
            )

            ax.set_xticks(np.linspace(x_data.min(), x_data.max(), num=5))
            ax.set_yticks(np.linspace(y_data.min(), y_data.max(), num=5))
            ax.set_xlim(x_data.min(), x_data.max())
            ax.set_ylim(y_data.min(), y_data.max())

            ax.grid(True, which="major", linestyle="--", alpha=0.3, zorder=0)
            ax.tick_params(labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig_scatter.tight_layout()

            buf = BytesIO()
            fig_scatter.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig_scatter)

            plots_dict[f"{c1}vs{c2}"] = f"data:image/png;base64,{img_b64}"

        corr_unstacked["value_str"] = corr_unstacked["value"].map(lambda x: f"{x:.2f}")
        corr_unstacked["text_color"] = corr_unstacked["value"].map(
            lambda val: "#ffffff" if abs(val) > 0.60 else "#000000"
        )
        corr_unstacked["img_src"] = [
            plots_dict[f"{x}vs{y}"]
            for x, y in zip(corr_unstacked["feature_x"], corr_unstacked["feature_y"])
        ]
        corr_source = ColumnDataSource(corr_unstacked)

        p = figure(
            x_range=(0, n),
            y_range=(0, n),
            x_axis_location="above",
            width=680,
            height=640,
            tools="hover,save,pan,box_zoom,reset",
            toolbar_location="right",
            align="center",
        )

        p.xaxis.ticker = FixedTicker(ticks=[i + 0.5 for i in range(n)])
        p.yaxis.ticker = FixedTicker(ticks=[i + 0.5 for i in range(n)])
        p.xaxis.major_label_overrides = {i + 0.5: l for i, l in enumerate(labels)}
        p.yaxis.major_label_overrides = {i + 0.5: l for i, l in enumerate(rev_labels)}

        p.xgrid.ticker = FixedTicker(ticks=list(range(n + 1)))
        p.ygrid.ticker = FixedTicker(ticks=list(range(n + 1)))
        p.grid.grid_line_color = "#b0b0b0"
        p.grid.grid_line_width = 1.5
        p.grid.grid_line_alpha = 0.8

        p.add_layout(
            Title(
                text="Correlation Matrix - Wine Characteristics",
                text_font_size="14pt",
                text_font_style="bold",
                align="left",
            ),
            "above",
        )
        p.add_layout(
            Title(
                text=f"💡 Tip: Hover over cells to see detailed pair scatterplots. \n Note: Non-Vintage wines were excluded from this visualization.",
                text_font_size="9pt",
                text_font_style="italic",
                text_color="#555555",
                align="left",
            ),
            "above",
        )

        p.rect(
            x="x_num",
            y="y_num",
            width=1,
            height=1,
            source=corr_source,
            fill_color={"field": "value", "transform": color_mapper},
            line_color=None,
        )
        p.text(
            x="x_num",
            y="y_num",
            text="value_str",
            source=corr_source,
            text_align="center",
            text_baseline="middle",
            text_font_size="11pt",
            text_font_style="bold",
            text_color="text_color",
        )

        hover = p.select_one(HoverTool)
        hover.tooltips = """
            <div style="padding: 14px; background-color: #ffffff; border: 1px solid #cccccc; border-radius: 8px; text-align: center; box-shadow: 3px 4px 12px rgba(0,0,0,0.22);">
                <div style="font-size: 15px; font-weight: bold; margin-bottom: 6px; color: #111111;">
                    @feature_x vs @feature_y
                </div>
                <div style="font-size: 13px; margin-bottom: 10px; color: #444444;">
                    Correlation Coefficient: <strong>@value_str</strong>
                </div>
                <div>
                    <img src="@img_src"
                         alt="Scatterplot showing the relationship between @feature_x and @feature_y"
                         style="width: 360px; height: auto; border: 1px solid #dddddd; border-radius: 6px;" />
                </div>
            </div>
        """

        color_bar = ColorBar(
            color_mapper=color_mapper,
            ticker=BasicTicker(desired_num_ticks=10),
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title="Correlation",
            title_text_font_style="bold",
            title_text_font_size="9pt",
        )
        p.add_layout(color_bar, "right")

        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.xaxis.major_label_orientation = 45
        p.xaxis.major_label_text_font_style = "bold"
        p.yaxis.major_label_text_font_style = "bold"

        takeaway = Div(
                text="""
                <div style="max-width: 1150px; margin: 15px auto 0 auto; padding: 14px 20px; background: #FDFCF7; border: 1px solid #EAE5DC; border-left: 6px solid #AF1B3F; border-radius: 6px; font-family: sans-serif; font-size: 15px; color: #4A3B32; line-height: 1.6;">
                    <div style="margin-bottom: 6px;">
                        <b style="color: #AF1B3F; font-size: 17px;">• Log Price (vs Rating: 0.76):</b> Strongest positive relationship in the matrix; price is the primary driver and indicator of perceived quality.
                    </div>
                    <div style="margin-bottom: 6px;">
                        <b style="color: #AF1B3F; font-size: 17px;">• Vintage (vs Price: -0.59 | vs Rating: -0.37):</b> Reflects the harvest season; earlier years correlate with higher price and rating due to cellaring time. Multi-vintage blends (NV) lack a numeric harvest year and were excluded to preserve temporal linear correlation.
                    </div>
                    <div>
                        <b style="color: #AF1B3F; font-size: 17px;">• ABV Alcohol % (vs Price: 0.21 | vs Rating: 0.21):</b> Weak positive correlation; higher alcohol percentage slightly aligns with higher rating and price, but it is not a key value driver.
                    </div>
                </div>
                """,
                sizing_mode="stretch_width",
                align="center",
            )

        return self.stack([[p], [takeaway]])


    def create_slide_4_radar(self):
        """Slide 4: Interactive radar chart of food pairings by wine type."""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif; margin-bottom: 5px;">
            Wine & Food Pairing Profiles
        </h2>
        <p style="text-align: center; color: #5C4A42; font-family: 'Lusitana', Georgia, serif; font-size: 14px; margin-top: 0; margin-bottom: 15px;">
            For each wine type see w<b style="color: #AF1B3F;">hat percentage of wine goes with each food. By selecting the appropriate button you can restrict your attention to the top 10% wines in the category in terms of rating or in terms of price.
        </p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        # Map the detailed Harmonize labels to the eight presentation groups.
        food_groups = {
            "Beef": "Red meat & game",
            "Lamb": "Red meat & game",
            "Veal": "Red meat & game",
            "Game Meat": "Red meat & game",
            "Poultry": "Poultry & pork",
            "Chicken": "Poultry & pork",
            "Pork": "Poultry & pork",
            "Lean Fish": "Fish & seafood",
            "Rich Fish": "Fish & seafood",
            "Fish": "Fish & seafood",
            "Seafood": "Fish & seafood",
            "Shellfish": "Fish & seafood",
            "Blue Cheese": "Cheese",
            "Goat Cheese": "Cheese",
            "Hard Cheese": "Cheese",
            "Maturated Cheese": "Cheese",
            "Mild Cheese": "Cheese",
            "Soft Cheese": "Cheese",
            "Cheese": "Cheese",
            "Vegetarian": "Vegetarian",
            "Salad": "Vegetarian",
            "Mushrooms": "Vegetarian",
            "Spicy Food": "Spicy & cured foods",
            "Cured Meat": "Spicy & cured foods",
            "Dessert": "Desserts & fruit",
            "Sweet Dessert": "Desserts & fruit",
            "Fruit Dessert": "Desserts & fruit",
            "Cake": "Desserts & fruit",
            "Fruit": "Desserts & fruit",
            "Pasta": "Other savoury dishes",
            "Pizza": "Other savoury dishes",
            "Barbecue": "Other savoury dishes",
            "Appetizer": "Other savoury dishes",
            "Aperitif": "Other savoury dishes",
            "Snack": "Other savoury dishes",
            "Soufflé": "Other savoury dishes",
            "Cream": "Other savoury dishes",
        }

        categories = [
            "Cheese",
            "Desserts & fruit",
            "Fish & seafood",
            "Other savoury dishes",
            "Poultry & pork",
            "Red meat & game",
            "Spicy & cured foods",
            "Vegetarian",
        ]
        wine_types = ["Red", "White", "Rose", "Sparkling"]

        radar_df = self.df.copy()
        radar_df["Price"] = pd.to_numeric(radar_df["Price"], errors="coerce")
        radar_df["Rating"] = pd.to_numeric(radar_df["Rating"], errors="coerce")

        def parse_harmonize(value):
            """Convert Harmonize values to Python lists robustly."""
            if isinstance(value, list):
                return value
            if pd.isna(value):
                return []
            if isinstance(value, str):
                value = value.strip()
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple, set)):
                        return list(parsed)
                except (ValueError, SyntaxError):
                    pass
                return [
                    item.strip().strip("'\"")
                    for item in value.strip("[]").split(",")
                    if item.strip().strip("'\"")
                ]
            return []

        radar_df["_pairing"] = radar_df["Harmonize"].apply(parse_harmonize)

        def create_foodgroup_percentages(df_input):
            """Return % of wines in each wine type matched to each food group."""
            if df_input.empty:
                return pd.DataFrame(0.0, index=wine_types, columns=categories)

            exploded = df_input[["Name", "Kind", "_pairing"]].explode("_pairing")
            exploded["FoodGroup"] = exploded["_pairing"].map(food_groups)
            exploded = exploded.dropna(subset=["FoodGroup"])

            wine_food = exploded[["Name", "Kind", "FoodGroup"]].drop_duplicates()
            counts = pd.crosstab(wine_food["Kind"], wine_food["FoodGroup"])
            totals = df_input.groupby("Kind")["Name"].nunique()

            percentages = pd.DataFrame(0.0, index=wine_types, columns=categories)
            for kind in wine_types:
                total = totals.get(kind, 0)
                if total == 0:
                    continue
                for category in categories:
                    count = counts.loc[kind, category] if kind in counts.index and category in counts.columns else 0
                    percentages.loc[kind, category] = 100.0 * count / total

            return percentages

        # Calculate the 90th-percentile cutoff separately within each wine type.
        # This means, for example, that Red wines are compared only with other
        # Red wines when selecting the top 10% by price or rating.
        price_cutoffs = radar_df.groupby("Kind")["Price"].quantile(0.90)
        rating_cutoffs = radar_df.groupby("Kind")["Rating"].quantile(0.90)

        top_10_price = radar_df[
            radar_df["Price"] >= radar_df["Kind"].map(price_cutoffs)
        ].copy()

        top_10_rating = radar_df[
            radar_df["Rating"] >= radar_df["Kind"].map(rating_cutoffs)
        ].copy()

        dataset_percentages = {
            "All wines": create_foodgroup_percentages(radar_df),
            "Top 10% priciest": create_foodgroup_percentages(top_10_price),
            "Top 10% rated": create_foodgroup_percentages(top_10_rating),
        }

        dataset_titles = {
            "All wines": "All wines",
            "Top 10% priciest": "Top 10% priciest within each wine type",
            "Top 10% rated": "Top 10% rated within each wine type",
        }

        def cutoff_summary(cutoffs, decimals=2):
            parts = []
            for kind in wine_types:
                if kind in cutoffs.index and pd.notna(cutoffs.loc[kind]):
                    parts.append(f"{kind}: {cutoffs.loc[kind]:.{decimals}f}")
            return " | ".join(parts)

        dataset_notes = {
            "All wines": "Showing all matched wines in the dataset.",
            "Top 10% priciest": (
                "Top 10% by price within each wine type. 90th-percentile cutoffs: "
                + cutoff_summary(price_cutoffs, 2)
            ),
            "Top 10% rated": (
                "Top 10% by rating within each wine type. 90th-percentile cutoffs: "
                + cutoff_summary(rating_cutoffs, 2)
            ),
        }

        # Radar geometry: convert polar coordinates to ordinary x/y coordinates.
        n_categories = len(categories)
        angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_categories, endpoint=False)
        closed_angles = np.append(angles, angles[0])

        # Keep the radar panel square, with enough plotting room for labels.
        # The x/y ranges are deliberately identical so the radar grid remains
        # circular instead of being stretched into an ellipse.
        p = figure(
            width=660,
            height=660,
            x_range=Range1d(-160, 160),
            y_range=Range1d(-160, 160),
            tools="reset,save",
            toolbar_location="above",
            match_aspect=True,
            align="center",
        )
        p.title.text = dataset_titles["All wines"]
        p.title.align = "center"
        p.background_fill_color = "#FFFFFF"
        p.border_fill_color = "#FDFCF7"
        p.outline_line_color = None
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.grid.visible = False

        # Concentric percentage rings (20% to 100%).
        for radius in [20, 40, 60, 80, 100]:
            ring_x = radius * np.cos(np.linspace(0, 2 * np.pi, 241))
            ring_y = radius * np.sin(np.linspace(0, 2 * np.pi, 241))
            p.line(ring_x, ring_y, line_color="#D9D3CC", line_width=1, line_alpha=0.8)
            p.text(
                x=[3],
                y=[radius],
                text=[f"{radius}%"],
                text_font_size="9pt",
                text_color="#7A6A61",
                text_align="left",
                text_baseline="middle",
            )

        # Category spokes and labels. Long labels are split across two lines
        # and placed as separate text glyphs, so they fit comfortably around
        # the square radar chart without being clipped.
        label_radius = 118
        label_lines = {
            "Cheese": ["Cheese"],
            "Desserts & fruit": ["Desserts", "& fruit"],
            "Fish & seafood": ["Fish", "& seafood"],
            "Other savoury dishes": ["Other savoury", "dishes"],
            "Poultry & pork": ["Poultry", "& pork"],
            "Red meat & game": ["Red meat", "& game"],
            "Spicy & cured foods": ["Spicy & cured", "foods"],
            "Vegetarian": ["Vegetarian"],
        }

        for angle, category in zip(angles, categories):
            p.line(
                [0, 100 * np.cos(angle)],
                [0, 100 * np.sin(angle)],
                line_color="#D9D3CC",
                line_width=1,
            )

            x_label = label_radius * np.cos(angle)
            y_label = label_radius * np.sin(angle)

            align = "center"
            if x_label > 20:
                align = "left"
            elif x_label < -20:
                align = "right"

            lines = label_lines.get(category, [category])
            line_spacing = 7
            start_offset = (len(lines) - 1) * line_spacing / 2

            for i, label_text in enumerate(lines):
                p.text(
                    x=[x_label],
                    y=[y_label + start_offset - i * line_spacing],
                    text=[label_text],
                    text_font_size="9pt",
                    text_color="#211B18",
                    text_align=align,
                    text_baseline="middle",
                )

        wine_colors = {
            "Red": "#AF1B3F",
            "White": "#D4A72C",
            "Rose": "#D77A8A",
            "Sparkling": "#218380",
        }

        def make_radar_source(percentages, wine_kind):
            """Return separate closed polygon and unique hover-point data."""
            values = percentages.loc[wine_kind, categories].to_numpy(dtype=float)

            # Closed coordinates are needed for the polygon and line only.
            closed_values = np.append(values, values[0])
            polygon_x = closed_values * np.cos(closed_angles)
            polygon_y = closed_values * np.sin(closed_angles)

            # Hover/scatter coordinates contain each food group exactly once.
            # This avoids a duplicate tooltip at the first/last radar point.
            point_x = values * np.cos(angles)
            point_y = values * np.sin(angles)

            polygon_data = dict(x=polygon_x, y=polygon_y)
            point_data = dict(
                x=point_x,
                y=point_y,
                value=values,
                category=categories,
                wine=[wine_kind] * n_categories,
            )
            return polygon_data, point_data

        active_dataset = "All wines"
        sources = {}
        renderers = []
        for wine_kind in wine_types:
            polygon_data, point_data = make_radar_source(
                dataset_percentages[active_dataset], wine_kind
            )
            polygon_source = ColumnDataSource(polygon_data)
            point_source = ColumnDataSource(point_data)
            sources[wine_kind] = {
                "polygon": polygon_source,
                "points": point_source,
            }

            patch = p.patch(
                "x",
                "y",
                source=polygon_source,
                fill_color=wine_colors[wine_kind],
                fill_alpha=0.08,
                line_color=None,
            )
            line = p.line(
                "x",
                "y",
                source=polygon_source,
                line_width=3,
                line_color=wine_colors[wine_kind],
            )
            points = p.scatter(
                "x",
                "y",
                source=point_source,
                size=7,
                fill_color=wine_colors[wine_kind],
                line_color="#FFFFFF",
                line_width=1,
            )
            renderers.append((patch, line, points))

            p.add_tools(
                HoverTool(
                    renderers=[points],
                    tooltips=[
                        ("Wine", "@wine"),
                        ("Food group", "@category"),
                        ("Share", "@value{0.0}%"),
                    ],
                )
            )

        dataset_toggle = RadioButtonGroup(
            labels=list(dataset_percentages.keys()),
            active=0,
            width=760,
            css_classes=["theme-btn"],
            align="center",
        )
        dataset_toggle.stylesheets = [
            *dataset_toggle.stylesheets,
            InlineStyleSheet(
                css="""
                .bk-btn-group .bk-btn {
                    font-weight: 600;
                    border-color: #AF1B3F;
                    color: #AF1B3F;
                    background-color: #FFFFFF;
                }
                .bk-btn-group .bk-btn.bk-active {
                    color: #FFFFFF;
                    background-color: #AF1B3F;
                }
                """
            ),
        ]

        # The colored toggle buttons double as the legend, so a separate plot
        # legend is deliberately omitted to keep the radar area uncluttered.
        wine_toggle = CheckboxButtonGroup(
            labels=wine_types,
            active=[0, 1, 2, 3],
            width=600,
            css_classes=["theme-btn"],
            align="center",
        )

        # Give each toggle the same color as its radar series. Active buttons
        # are filled; inactive buttons keep a colored outline/text so the color
        # mapping remains clear even when a wine type is hidden.
        wine_toggle.stylesheets = [
            *wine_toggle.stylesheets,
            InlineStyleSheet(
                css="""
                .bk-btn-group .bk-btn {
                    font-weight: 600;
                    border-width: 2px;
                    transition: opacity 0.15s ease, background-color 0.15s ease;
                }

                .bk-btn-group .bk-btn:nth-child(1) {
                    color: #AF1B3F;
                    border-color: #AF1B3F;
                    background-color: #FFFFFF;
                }
                .bk-btn-group .bk-btn:nth-child(1).bk-active {
                    color: #FFFFFF;
                    background-color: #AF1B3F;
                }

                .bk-btn-group .bk-btn:nth-child(2) {
                    color: #8A6B00;
                    border-color: #D4A72C;
                    background-color: #FFFFFF;
                }
                .bk-btn-group .bk-btn:nth-child(2).bk-active {
                    color: #211B18;
                    background-color: #D4A72C;
                }

                .bk-btn-group .bk-btn:nth-child(3) {
                    color: #B14F63;
                    border-color: #D77A8A;
                    background-color: #FFFFFF;
                }
                .bk-btn-group .bk-btn:nth-child(3).bk-active {
                    color: #FFFFFF;
                    background-color: #D77A8A;
                }

                .bk-btn-group .bk-btn:nth-child(4) {
                    color: #218380;
                    border-color: #218380;
                    background-color: #FFFFFF;
                }
                .bk-btn-group .bk-btn:nth-child(4).bk-active {
                    color: #FFFFFF;
                    background-color: #218380;
                }
                """
            ),
        ]

        note = Div(
            text=f"""
        <p style="text-align: center; color: #7A6A61; font-size: 12px; margin-top: 0;">
            <b>{dataset_notes[active_dataset]}</b><br>
            Scale: 0–100%. A value of 100% means that every wine in that selected sample has at least one pairing in the corresponding food group.
        </p>
        """,
            width=900,
            align="center",
        )

        def update_visibility(attr, old, new):
            active = set(new)
            for index, renderer_group in enumerate(renderers):
                visible = index in active
                for renderer in renderer_group:
                    renderer.visible = visible

        def update_dataset(attr, old, new):
            dataset_name = dataset_toggle.labels[new]
            percentages = dataset_percentages[dataset_name]
            p.title.text = dataset_titles[dataset_name]

            for wine_kind in wine_types:
                polygon_data, point_data = make_radar_source(percentages, wine_kind)
                sources[wine_kind]["polygon"].data = polygon_data
                sources[wine_kind]["points"].data = point_data

            note.text = f"""
        <p style="text-align: center; color: #7A6A61; font-size: 12px; margin-top: 0;">
            <b>{dataset_notes[dataset_name]}</b><br>
            Scale: 0–100%. A value of 100% means that every wine in that selected sample has at least one pairing in the corresponding food group.
        </p>
        """

        wine_toggle.on_change("active", update_visibility)
        dataset_toggle.on_change("active", update_dataset)

        selector_label = Div(
            text="""
        <p style="text-align: center; color: #5C4A42; font-size: 13px; margin: 0 0 3px 0;">
            Select sample
        </p>
        """,
            width=900,
            align="center",
        )
        wine_label = Div(
            text="""
        <p style="text-align: center; color: #5C4A42; font-size: 13px; margin: 6px 0 3px 0;">
            Select wine types
        </p>
        """,
            width=900,
            align="center",
        )

        return self.stack([
            [title],
            [wine_label],
            [wine_toggle],
            [selector_label],
            [dataset_toggle],
            [p],
            [note],
        ])

    def create_slide_5_violin(self):
        """Slide 5: Blend versus single-variety violin plots."""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif; margin-bottom: 5px;">
            Blend versus Single-Variety Wines
        </h2>
        <p style="text-align: center; color: #5C4A42; font-family: 'Lusitana', Georgia, serif; font-size: 14px; margin-top: 0; margin-bottom: 10px;">
            Compare the distributions of ratings and prices for blended wines and single-variety wines.
        </p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        violin_df = self.df.copy()
        violin_df["Rating"] = pd.to_numeric(violin_df["Rating"], errors="coerce")
        violin_df["Price"] = pd.to_numeric(violin_df["Price"], errors="coerce")

        # Match the notebook logic: Varietal/100% is single-variety; all other elaborate types are blends.
        violin_df["Blend type"] = np.where(
            violin_df["Elaborate"].eq("Varietal/100%"),
            "Single-variety",
            "Blend",
        )

        # Use log price to reduce the visual dominance of extreme high-price wines.
        violin_df = violin_df[(violin_df["Price"] > 0) & violin_df["Rating"].notna()].copy()
        violin_df["Log price"] = np.log(violin_df["Price"])

        blend_order = ["Blend", "Single-variety"]
        blend_colors = {
            "Single-variety": "#218380",
            "Blend": "#AF1B3F",
        }

        def smooth_density(values, x_grid):
            """Small dependency-free KDE-like density for Bokeh violin shapes."""
            values = pd.Series(values).dropna().to_numpy(dtype=float)
            if len(values) < 2:
                return np.zeros_like(x_grid)

            std = np.std(values, ddof=1)
            if std <= 0 or not np.isfinite(std):
                std = 1e-6

            # Silverman's rule of thumb, with a small floor to avoid overly narrow spikes.
            bandwidth = max(1.06 * std * (len(values) ** (-1 / 5)), std * 0.08, 1e-6)
            z = (x_grid[:, None] - values[None, :]) / bandwidth
            density = np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))
            return density

        def make_violin_data(data, value_col, x_padding=0.05):
            values_all = data[value_col].dropna().to_numpy(dtype=float)
            x_min, x_max = float(values_all.min()), float(values_all.max())
            span = x_max - x_min
            if span == 0:
                span = 1.0
            x_grid = np.linspace(x_min - x_padding * span, x_max + x_padding * span, 160)

            patch_sources = []
            stat_rows = []
            for y_pos, blend_type in enumerate(blend_order):
                values = data.loc[data["Blend type"] == blend_type, value_col].dropna().to_numpy(dtype=float)
                density = smooth_density(values, x_grid)
                if density.max() > 0:
                    width = density / density.max() * 0.34
                else:
                    width = np.zeros_like(density)

                xs = np.concatenate([x_grid, x_grid[::-1]])
                ys = np.concatenate([y_pos + width, (y_pos - width)[::-1]])

                if len(values) > 0:
                    q1, median, q3 = np.percentile(values, [25, 50, 75])
                    mean = float(np.mean(values))
                else:
                    q1 = median = q3 = mean = np.nan

                source = ColumnDataSource(
                    data=dict(
                        x=xs,
                        y=ys,
                        blend_type=[blend_type] * len(xs),
                        n=[len(values)] * len(xs),
                        median=[median] * len(xs),
                        q1=[q1] * len(xs),
                        q3=[q3] * len(xs),
                    )
                )
                patch_sources.append((source, blend_type, y_pos, q1, median, q3, len(values), mean))
                stat_rows.append((blend_type, len(values), mean, median, q1, q3))

            return patch_sources, stat_rows, (x_min - x_padding * span, x_max + x_padding * span)

        def create_violin_plot(value_col, title_text, x_label, width=560, height=410):
            patch_sources, stat_rows, x_range = make_violin_data(violin_df, value_col)

            p = figure(
                width=width,
                height=height,
                x_range=Range1d(*x_range),
                y_range=Range1d(-0.65, 1.65),
                title=title_text,
                tools="pan,wheel_zoom,reset,save",
                toolbar_location="above",
                align="center",
            )
            p.title.align = "center"
            p.background_fill_color = "#FFFFFF"
            p.border_fill_color = "#FDFCF7"
            p.outline_line_color = "#E2D7C3"
            p.xaxis.axis_label = x_label
            p.yaxis.ticker = [0, 1]
            p.yaxis.major_label_overrides = {0: "Single-variety", 1: "Blend"}
            p.ygrid.grid_line_color = None
            p.xgrid.grid_line_color = "#E2D7C3"
            p.xgrid.grid_line_alpha = 0.45

            hover_renderers = []
            for source, blend_type, y_pos, q1, median, q3, n, mean in patch_sources:
                p.patch(
                    "x",
                    "y",
                    source=source,
                    fill_color=blend_colors[blend_type],
                    fill_alpha=0.32,
                    line_color=blend_colors[blend_type],
                    line_width=2,
                )

                # Quartile line and median marker inside the violin.
                if np.isfinite(q1) and np.isfinite(q3):
                    p.segment(
                        q1, y_pos, q3, y_pos,
                        line_color="#211B18",
                        line_width=3,
                        line_alpha=0.65,
                    )

                # Use a dedicated one-row source for hover information.
                # Hovering a polygon patch does not provide a stable data-row index
                # and therefore produced Bokeh's "???" placeholders.
                if np.isfinite(median):
                    stat_source = ColumnDataSource(
                        data=dict(
                            x=[median],
                            y=[y_pos],
                            blend_type=[blend_type],
                            n=[n],
                            median=[median],
                            q1=[q1],
                            q3=[q3],
                        )
                    )
                    median_point = p.scatter(
                        "x",
                        "y",
                        source=stat_source,
                        size=11,
                        color="#211B18",
                        line_color="#FFFFFF",
                        line_width=1,
                    )
                    hover_renderers.append(median_point)

            p.add_tools(
                HoverTool(
                    renderers=hover_renderers,
                    tooltips=[
                        ("Wine type", "@blend_type"),
                        ("Number of wines", "@n{0,0}"),
                        ("Median", "@median{0.00}"),
                        ("Q1", "@q1{0.00}"),
                        ("Q3", "@q3{0.00}"),
                    ],
                    attachment="above",
                )
            )
            return p, stat_rows

        rating_plot, rating_stats = create_violin_plot(
            "Rating",
            "Rating distribution",
            "Rating",
        )
        price_plot, price_stats = create_violin_plot(
            "Log price",
            "Price distribution",
            "Log price",
        )

        def stat_html(label, rows):
            body_rows = "".join(
                f"""
                <tr>
                    <td style='padding: 3px 8px; text-align: center;'>{blend_type}</td>
                    <td style='padding: 3px 8px; text-align: center;'>{n:,}</td>
                    <td style='padding: 3px 8px; text-align: center;'>{median:.2f}</td>
                    <td style='padding: 3px 8px; text-align: center;'>{q1:.2f} - {q3:.2f}</td>
                </tr>
                """
                for blend_type, n, mean, median, q1, q3 in rows
            )
            return f"""
                <h4 style='color: #AF1B3F; margin: 2px 0 4px 0; text-align: center;'>{label}</h4>
                <table style='width: 100%; border-collapse: collapse; font-size: 12px; color: #211B18; text-align: center;'>
                    <tr style='border-bottom: 1px solid #E2D7C3;'>
                        <th style='padding: 3px 8px; text-align: center;'>Type</th>
                        <th style='padding: 3px 8px; text-align: center;'>n</th>
                        <th style='padding: 3px 8px; text-align: center;'>Median</th>
                        <th style='padding: 3px 8px; text-align: center;'>IQR</th>
                    </tr>
                    {body_rows}
                </table>
            """

        summary = Div(
            text=f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 10px 16px; border-radius: 8px; color: #211B18; text-align: center; margin: 0 auto;">
            <p style="font-size: 13px; margin: 0 0 8px 0; color: #5C4A42; text-align: center;">
                Filled shapes show the distribution; the black dot marks the median and the horizontal line shows the interquartile range.
            </p>
            <div style="display: flex; gap: 24px; justify-content: center; align-items: flex-start;">
                <div style="width: 48%; text-align: center;">{stat_html('Rating', rating_stats)}</div>
                <div style="width: 48%; text-align: center;">{stat_html('Log price', price_stats)}</div>
            </div>
        </div>
        """,
            width=1120,
            height=155,
            align="center",
        )

        return self.stack([[title], [rating_plot, price_plot], [summary]])

    def create_slide_6_conclusions(self):
        """Slide 6: Conclusions"""
        TAKEAWAYS = [
            """<b style="color: #AF1B3F"> Finding 1</b>: In general, higher price means higher rating, although there are some exception""",
            """<b style="color: #AF1B3F"> Finding 2</b>: Geographic provenance impacts price more than it does rating""",
            """<b style="color: #AF1B3F"> Finding 3</b>: Higher ABV is positively correlated to (log) price and rating. Vintage is negatively correlated to (log) price and rating""",
            """<b style="color: #AF1B3F"> Finding 4</b>: Top priced wines tend to pair with fewer food groups. Same for rating, but to a lower extent""",
            """<b style="color: #AF1B3F"> Finding 5</b>: Elaborate impacts price but does not impact rating"""
        ]

        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif; margin-bottom: 5px;">
            Conclusions
        </h2>
        <p style="text-align: center; color: #5C4A42; font-family: 'Lusitana', Georgia, serif; font-size: 14px; margin-top: 0; margin-bottom: 15px;">
            What the data tells us about price, rating and wine features
        </p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        bullets = "".join(
            f"<li style='margin-bottom: 14px; padding-left: 4px;'>{item}</li>"
            for item in TAKEAWAYS
        )

        card = Div(
            text=f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; border-left: 6px solid #AF1B3F; padding: 28px 44px; border-radius: 10px; color: #211B18; font-family: 'Lusitana', Georgia, serif; box-shadow: 0 4px 15px rgba(175, 27, 63, 0.05);">
            <ul style="font-size: 20px; line-height: 1.5; text-align: left !important; margin: 0 !important; padding-left: 24px; list-style-position: outside; list-style-type: disc;">
                {bullets}
            </ul>
        </div>
        """,
            width=1000,
            height=60 + 62 * len(TAKEAWAYS),
            align="center",
        )

        return self.stack([[title], [card]])

    def update_slide(self):
        """Update current slide display and UI elements"""
        self.prev_button.disabled = self.current_slide == 0
        self.next_button.disabled = self.current_slide == self.total_slides - 1

        self.progress_div.text = self.get_progress_html()
        self.slide_select.value = str(self.current_slide)
        self.main_content.children = [self.slides[self.current_slide]]

    def prev_slide(self):
        if self.current_slide > 0:
            self.current_slide -= 1
            self.update_slide()

    def next_slide(self):
        if self.current_slide < self.total_slides - 1:
            self.current_slide += 1
            self.update_slide()
        elif self.auto_play:
            self.current_slide = 0
            self.update_slide()

    def go_home(self):
        self.current_slide = 0
        self.update_slide()

    def jump_to_slide(self, attr, old, new):
        self.current_slide = int(new)
        self.update_slide()

    def toggle_auto_play(self):
        """Single button: starts autoplay when idle, pauses it when playing."""
        if self.auto_play:
            self.stop_auto_play()
        else:
            self.start_auto_play()

    def start_auto_play(self):
        if self.auto_play:
            return
        self.auto_play = True
        self.auto_play_callback = curdoc().add_periodic_callback(
            self.auto_advance, 5000
        )
        self.play_button.label = "⏸ Pause"
        self.play_button.css_classes = ["theme-btn", "theme-btn-active"]

    def stop_auto_play(self):
        if not self.auto_play:
            return
        self.auto_play = False
        if self.auto_play_callback is not None:
            # must be given the callback object returned by add_periodic_callback
            curdoc().remove_periodic_callback(self.auto_play_callback)
            self.auto_play_callback = None
        self.play_button.label = "▶ Auto Play"
        self.play_button.css_classes = ["theme-btn"]

    def auto_advance(self):
        self.next_slide()

    def create_layout(self):
        """Assemble main presentation layout centered horizontally"""

        separator = Div(
            text="<hr style='border: 0; height: 1.5px; background-color: #E2D7C3; margin: 12px auto 20px auto;'>",
            sizing_mode="stretch_width",
            height=15,
        )

        nav_bar = row(
            self.style_div,
            self.prev_button,
            self.home_button,
            self.next_button,
            self.slide_select,
            self.play_button,
            self.progress_div,
            width=SLIDE_WIDTH,
            styles={"justify-content": "center"},
        )

        self.main_content = column(self.slides[0], width=SLIDE_WIDTH)

        # Root: fixed width, centered on the page with auto side margins.
        self.layout = column(
            nav_bar,
            separator,
            self.main_content,
            width=SLIDE_WIDTH,
            styles={"margin-left": "auto", "margin-right": "auto", "max-width": "100%"},
        )

        # Center every element of the layout and of every slide.
        center_everything(self.layout, is_root=True)
        for slide in self.slides:
            center_everything(slide)

        self.update_slide()


presentation = InteractivePresentation()
curdoc().add_root(presentation.layout)
curdoc().title = "Wine: should you splash out on bottle?"