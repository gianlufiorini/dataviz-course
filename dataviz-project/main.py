#!/usr/bin/env python
import base64
import json
import os
from io import BytesIO
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # off-screen rendering (no GUI) for the tooltip scatterplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    RadioGroup,
    Div,
    Button,
    Slider,
    Select,
    ColumnDataSource,
    FixedTicker,
    HoverTool,
    Range1d,
    LinearColorMapper,
    ColorBar,
    BasicTicker,
    PrintfTickFormatter,
    CustomJS,
    GeoJSONDataSource,
    LayoutDOM,
    Widget,
    InlineStyleSheet,
    Title,
)
from bokeh.layouts import column, row
from bokeh.palettes import Cividis256, PuOr11, RdYlBu11
from bokeh.transform import factor_cmap
from bokeh.themes import Theme

# Fixed slide width: everything is centered inside this column instead of
# stretching across the full browser window.
SLIDE_WIDTH = 1200
CONTENT_WIDTH = 1160  # width of full-width text blocks (titles, banners)

# ==============================================================================
# MAP LOADING AND PROJECTION (Native Robinson via +proj=robin)
# ==============================================================================
base_dir = os.path.dirname(__file__)
WORLD_PATH = os.path.join(base_dir, "data", "world-countries.json")

try:
    print(f"Attempting to load map from: {WORLD_PATH}")
    _world_raw = gpd.read_file(WORLD_PATH)
    WORLD_GEO = _world_raw.to_crs("+proj=robin")
    print("Map successfully loaded and projected!")
except Exception as e:
    print(f"ERROR loading map: {e}")
    try:
        WORLD_GEO = _world_raw.to_crs("EPSG:4326")
    except Exception:
        WORLD_GEO = None


# Injected into every widget's shadow DOM (page-level CSS cannot reach inside it)
# so button groups, radio buttons, dropdown titles and slider titles are centered.
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
    """Centers every layout element horizontally, and all text inside Divs."""
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
    """
    Main application class for the Bokeh presentation system.
    Styled with crisp white plot containers, off-white background (#FDFCF7),
    user-specified hex colors for wine categories, and centered text/layouts.
    """

    theme_path = os.path.join(base_dir, "theme.yaml")
    curdoc().theme = Theme(filename=theme_path)

    def __init__(self, filename="wines_enhanced.csv"):
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(base_dir, "data", filename)

        # Load dataset
        self.df = pd.read_csv(data_path)
        self.current_slide = 0
        self.total_slides = 8
        self.slides = []
        self.auto_play = False
        self.auto_play_callback = None
        self._is_updating = False

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
        """Get title for each slide"""
        titles = [
            "Welcome",
            "Price vs Rating",
            "Price and Rating by Provenance",
            "Data Overview",
            "Interactive Analysis",
            "Time Series Trends",
            "Correlation Explorer",
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
        """Create all presentation slides"""
        self.slides = [
            self.create_slide_title(),
            self.create_slide_1_price_vs_rating(),
            self.create_slide_2_visual_vocabulary(),
            self.create_slide_3_overview(),
            self.create_slide_4_interactive(),
            self.create_slide_5_timeseries(),
            self.create_slide_6_correlation(),
            self.create_slide_7_conclusions(),
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
            <p style="text-align: center; color: #5C4A42; font-size: 14px; margin-top: 0; margin-bottom: 15px;">
                    Select a wine category below to inspect rating and price distribution alongside a robust linear regression fit (Huber loss, fitted on log price).
            </p>
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
            "Sparkling": "#218380"
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
            x_axis_label="Price (€)",
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
            line_width=0.5
        )

        fit_line = p.line(
            x="x_fit",
            y="y_fit",
            source=fit_source,
            color="#000000",
            line_width=2.5,
            legend_label="Robust linear fit (Huber)"
        )

        x_min = df.loc[df["Price"] > 0, "Price"].min()
        x_max = df["Price"].max()
        x_pad = (x_max / x_min) ** 0.03
        p.x_range = Range1d(x_min / x_pad, x_max * x_pad)

        y_min = df["Rating_jittered"].min()
        y_max = df["Rating_jittered"].max()
        y_pad = 0.05 * (y_max - y_min)
        p.y_range = Range1d(y_min - y_pad, y_max + y_pad)

        p.legend.location = "top_left"
        p.legend.background_fill_alpha = 0.85

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

        return self.stack([[title], [toggle], [p]])

    def create_slide_2_visual_vocabulary(self):
        """Slide 2: Geographic Wine Analysis (Price & Rating World Maps - Vertically Stacked)"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif; margin-bottom: 5px;">
            Geographic Wine Analysis: Price & Rating by Country
        </h2>
        <p style="text-align: center; color: #5C4A42; font-family: 'Lusitana', Georgia, serif; font-size: 14px; margin-top: 0; margin-bottom: 15px;">
            Select a wine category below to update the maps. Hover over any country to inspect detailed price and rating statistics.
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

        bounds = WORLD_GEO.total_bounds
        geo_width = bounds[2] - bounds[0]
        geo_height = bounds[3] - bounds[1]
        map_aspect = geo_width / geo_height if geo_height != 0 else 2.0

        x_bounds = (float(bounds[0]), float(bounds[2]))
        y_bounds = (float(bounds[1]), float(bounds[3]))

        def get_price_geojson(wine_type):
            df_sub = self.df if wine_type == "Global" else self.df[self.df["Kind"] == wine_type]
            stats = df_sub.groupby("Country")["Price"].agg(
                Price_Mean="mean",
                Price_Median="median",
                Price_Min="min",
                Price_Max="max",
                Count="count"
            ).round(2).reset_index()

            merged = WORLD_GEO.merge(stats, how="left", left_on="name", right_on="Country")
            merged[["Price_Mean", "Price_Median", "Price_Min", "Price_Max", "Count"]] = \
                merged[["Price_Mean", "Price_Median", "Price_Min", "Price_Max", "Count"]].fillna("N/A")
            return merged[["geometry", "name", "Price_Mean", "Price_Median", "Price_Min", "Price_Max", "Count"]].to_json()

        def get_rating_geojson(wine_type):
            df_sub = self.df if wine_type == "Global" else self.df[self.df["Kind"] == wine_type]
            stats = df_sub.groupby("Country")["Rating"].agg(
                Rating_Mean="mean",
                Rating_Median="median",
                Rating_Min="min",
                Rating_Max="max",
                Count="count"
            ).round(2).reset_index()

            merged = WORLD_GEO.merge(stats, how="left", left_on="name", right_on="Country")
            merged[["Rating_Mean", "Rating_Median", "Rating_Min", "Rating_Max", "Count"]] = \
                merged[["Rating_Mean", "Rating_Median", "Rating_Min", "Rating_Max", "Count"]].fillna("N/A")
            return merged[["geometry", "name", "Rating_Mean", "Rating_Median", "Rating_Min", "Rating_Max", "Count"]].to_json()

        price_source = GeoJSONDataSource(geojson=get_price_geojson("Global"))
        rating_source = GeoJSONDataSource(geojson=get_rating_geojson("Global"))

        p_means = self.df.groupby("Country")["Price"].mean()
        price_mapper = LinearColorMapper(
            palette=Cividis256,
            low=float(p_means.min()) if not p_means.empty else 0.0,
            high=float(p_means.max()) if not p_means.empty else 100.0,
            nan_color="#EBEBEB"
        )

        r_means = self.df.groupby("Country")["Rating"].mean()
        r_min = float(r_means.min()) if not r_means.empty else 3.0
        r_max = float(r_means.max()) if not r_means.empty else 5.0

        rating_mapper = LinearColorMapper(
            palette=PuOr11,
            low=r_min,
            high=r_max,
            nan_color="#EBEBEB"
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
        p_price.background_fill_color = "#FFFFFF"
        p_price.border_fill_color = "#FDFCF7"
        p_price.grid.grid_line_color = None

        price_patches = p_price.patches(
            "xs", "ys",
            source=price_source,
            fill_color={"field": "Price_Mean", "transform": price_mapper},
            fill_alpha=0.9,
            line_color="#FFFFFF",
            line_width=0.6,
            nonselection_fill_alpha=0.9,
            nonselection_fill_color={"field": "Price_Mean", "transform": price_mapper},
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
            mode="mouse"
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
        p_rating.background_fill_color = "#FFFFFF"
        p_rating.border_fill_color = "#FDFCF7"
        p_rating.grid.grid_line_color = None

        rating_patches = p_rating.patches(
            "xs", "ys",
            source=rating_source,
            fill_color={"field": "Rating_Mean", "transform": rating_mapper},
            fill_alpha=0.9,
            line_color="#FFFFFF",
            line_width=0.6,
            nonselection_fill_alpha=0.9,
            nonselection_fill_color={"field": "Rating_Mean", "transform": rating_mapper},
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
            mode="mouse"
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

    def create_correlation_matrix_plot(self):
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

            plots_dict[f"{c1}_vs_{c2}"] = f"data:image/png;base64,{img_b64}"

        corr_unstacked["value_str"] = corr_unstacked["value"].map(lambda x: f"{x:.2f}")
        corr_unstacked["text_color"] = corr_unstacked["value"].map(
            lambda val: "#ffffff" if abs(val) > 0.60 else "#000000"
        )
        corr_unstacked["img_src"] = [
            plots_dict[f"{x}_vs_{y}"]
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
                text="💡 Tip: Hover over cells to see detailed pair scatterplots.",
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

        return p

    def create_slide_3_overview(self):
        """Slide 3: Correlation matrix of the wine characteristics"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif;">Correlation Matrix</h2>
        <p style="text-align: center; color: #5C4A42;">Hover over a cell to see the scatterplot of that pair of variables</p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        corr_plot = self.create_correlation_matrix_plot()

        return self.stack([[title], [corr_plot]])

    def create_slide_4_interactive(self):
        """Slide 4: Interactive Analysis with Controls"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif;">🎮 Interactive Data Explorer</h2>
        <p style="text-align: center; color: #5C4A42;">Adjust parameters to explore different data visualizations</p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        self.slide4_source = ColumnDataSource(data=dict(x=[], y=[]))

        p = figure(width=780, height=450, title="Interactive Function Plotter", align="center")
        p.title.align = "left"
        self.slide4_line = p.line(
            "x", "y", source=self.slide4_source, line_width=2, color="#AF1B3F"
        )

        self.func_select = Select(
            title="Function:",
            value="sin",
            options=["sin", "cos", "exp", "log", "polynomial"],
            css_classes=["theme-select"],
            width=360,
            align="center",
        )
        self.param_slider = Slider(
            start=0.1, end=5, value=1, step=0.1, title="Parameter", width=360, align="center"
        )
        self.points_slider = Slider(
            start=50, end=500, value=100, step=50, title="Number of Points", width=360, align="center"
        )
        self.noise_slider = Slider(
            start=0, end=1, value=0, step=0.05, title="Noise Level", width=360, align="center"
        )

        def update_slide4():
            func = self.func_select.value
            param = self.param_slider.value
            n_points = int(self.points_slider.value)
            noise = self.noise_slider.value

            x = np.linspace(0, 10, n_points)
            if func == "sin":
                y = np.sin(param * x)
            elif func == "cos":
                y = np.cos(param * x)
            elif func == "exp":
                y = np.exp(-param * x / 5)
            elif func == "log":
                y = np.log(param * x + 1)
            else:
                y = param * x**2 - 2 * x + 1

            if noise > 0:
                y += np.random.normal(0, noise, len(y))

            self.slide4_source.data = dict(x=x, y=y)
            p.title.text = f"{func.upper()} Function (param={param:.1f}, noise={noise:.1f})"

        self.func_select.on_change("value", lambda a, o, n: update_slide4())
        self.param_slider.on_change("value", lambda a, o, n: update_slide4())
        self.points_slider.on_change("value", lambda a, o, n: update_slide4())
        self.noise_slider.on_change("value", lambda a, o, n: update_slide4())

        update_slide4()

        info = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 15px; border-radius: 8px; color: #211B18; text-align: center; margin: 0 auto;">
            <h3 style="color: #AF1B3F; margin-top: 0; text-align: center;">🎯 Try These:</h3>
            <ul style="font-size: 13px; margin-bottom: 0; text-align: center; list-style-position: inside; padding-left: 0;">
                <li>Change function type</li>
                <li>Adjust parameter to modify shape</li>
                <li>Add noise for realistic data</li>
            </ul>
        </div>
        """,
            width=360,
            height=150,
            align="center",
        )

        controls = column(
            self.func_select,
            self.param_slider,
            self.points_slider,
            self.noise_slider,
            info,
            align="center",
        )

        return self.stack([[title], [p, controls]])

    def create_slide_5_timeseries(self):
        """Slide 5: Time Series Analysis"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif;">📅 Time Series Analysis</h2>
        <p style="text-align: center; color: #5C4A42;">Exploring temporal patterns and trends</p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        base_trend = np.linspace(100, 150, 365)
        seasonal = 10 * np.sin(np.arange(365) * 2 * np.pi / 365)
        noise = np.random.randn(365) * 5
        values = base_trend + seasonal + noise

        ma7 = pd.Series(values).rolling(window=7).mean()
        ma30 = pd.Series(values).rolling(window=30).mean()

        source = ColumnDataSource(
            data=dict(dates=dates, values=values, ma7=ma7, ma30=ma30)
        )

        p = figure(
            width=1160,
            height=420,
            x_axis_type="datetime",
            title="Time Series with Moving Averages",
            align="center",
        )
        p.title.align = "left"

        p.line("dates", "values", source=source, line_width=1, color="#C8C2BC", alpha=0.7, legend_label="Daily")
        p.line("dates", "ma7", source=source, line_width=2, color="#218380", legend_label="7-day MA")
        p.line("dates", "ma30", source=source, line_width=2.5, color="#AF1B3F", legend_label="30-day MA")

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        hover = HoverTool(
            tooltips=[
                ("Date", "@dates{%F}"),
                ("Value", "@values{0.00}"),
                ("7-day MA", "@ma7{0.00}"),
                ("30-day MA", "@ma30{0.00}"),
            ],
            formatters={"@dates": "datetime"},
            point_policy='follow_mouse',
            attachment='above'
        )

        p.add_tools(hover)

        stats = Div(
            text=f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 15px 20px; border-radius: 8px; color: #211B18; text-align: center; margin: 0 auto;">
            <h3 style="color: #AF1B3F; margin-top: 0; text-align: center;">📈 Time Series Statistics:</h3>
            <table style="width: 100%; font-size: 14px; text-align: center; margin: 0 auto;">
                <tr><td style="text-align: center;"><b>Period:</b> {dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")} &nbsp;|&nbsp; <b>Mean Value:</b> {np.mean(values):.2f}</td></tr>
            </table>
        </div>
        """,
            width=1160,
            height=100,
            align="center",
        )

        return self.stack([[title], [p], [stats]])

    def create_slide_6_correlation(self):
        """Slide 6: Correlation Matrix Explorer"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif;">🔗 Correlation Analysis</h2>
        <p style="text-align: center; color: #5C4A42;">Exploring relationships between variables</p>
        """,
            sizing_mode="stretch_width",
            align="center",
        )

        n_vars = 8
        n_samples = 100
        var_names = [f"Var_{i + 1}" for i in range(n_vars)]

        data = np.random.randn(n_samples, n_vars)
        data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
        data[:, 2] = data[:, 0] * -0.6 + np.random.randn(n_samples) * 0.3
        data[:, 4] = data[:, 3] * 0.7 + np.random.randn(n_samples) * 0.3

        corr_matrix = np.corrcoef(data.T)

        corr_data = []
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                corr_data.append((var1, var2, corr_matrix[i, j]))

        source = ColumnDataSource(
            data=dict(
                var1=[x[0] for x in corr_data],
                var2=[x[1] for x in corr_data],
                corr=[x[2] for x in corr_data],
            )
        )

        p = figure(
            x_range=var_names,
            y_range=list(reversed(var_names)),
            width=680,
            height=500,
            title="Correlation Matrix",
            toolbar_location="above",
            tools="hover,save",
            align="center",
        )
        p.title.align = "left"

        mapper = LinearColorMapper(palette=RdYlBu11[::-1], low=-1, high=1)

        p.rect(
            "var1",
            "var2",
            width=1,
            height=1,
            source=source,
            fill_color={"field": "corr", "transform": mapper},
            line_color="#FFFFFF",
        )

        p.hover.tooltips = [
            ("Variables", "@var1 - @var2"),
            ("Correlation", "@corr{0.00}"),
        ]

        color_bar = ColorBar(
            color_mapper=mapper,
            width=10,
            location=(0, 0),
            ticker=BasicTicker(desired_num_ticks=10),
            formatter=PrintfTickFormatter(format="%.1f"),
        )
        p.add_layout(color_bar, "right")

        guide = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 220px; text-align: center; margin: 0 auto;">
            <h3 style="color: #AF1B3F; margin-top: 0; text-align: center;">📊 Interpretation Guide:</h3>
            <p style="font-size: 14px; line-height: 1.5; text-align: center;">Hover over individual cells to inspect precise correlation coefficients between dataset variables.</p>
        </div>
        """,
            width=440,
            height=260,
            align="center",
        )

        return self.stack([[title], [p, guide]])

    def create_slide_7_conclusions(self):
        """Slide 7: Conclusions and Summary"""
        title = Div(
            text="""
        <h1 style="text-align: center; color: #AF1B3F; font-family: 'Lusitana', serif;">
            🎯 Conclusions & Key Takeaways
        </h1>
        """,
            sizing_mode="stretch_width",
            height=80,
            align="center",
        )

        card1 = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 180px; text-align: center; margin: 0 auto;">
            <h3 style="color: #AF1B3F; margin-top: 0; text-align: center;">✅ What We've Demonstrated</h3>
            <ul style="font-size: 14px; line-height: 1.5; text-align: center; list-style-position: inside; padding-left: 0;">
                <li>Interactive visualizations with real-time updates</li>
                <li>Multiple chart types and responsive layouts</li>
                <li>Custom theme synchronization</li>
            </ul>
        </div>
        """,
            width=560,
            height=220,
            align="center",
        )

        card2 = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 180px; text-align: center; margin: 0 auto;">
            <h3 style="color: #AF1B3F; margin-top: 0; text-align: center;">🚀 Bokeh Advantages</h3>
            <ul style="font-size: 14px; line-height: 1.5; text-align: center; list-style-position: inside; padding-left: 0;">
                <li>Python callbacks for complex logic</li>
                <li>Real-time data streaming</li>
                <li>Server-side computation</li>
            </ul>
        </div>
        """,
            width=560,
            height=220,
            align="center",
        )

        thanks = Div(
            text="""
        <div style="text-align: center; margin: 20px auto 0 auto; font-family: 'Lusitana', serif;">
            <h2 style="color: #AF1B3F; text-align: center;">Thank You! 🙏</h2>
            <p style="font-size: 16px; color: #5C4A42; text-align: center;">
                This presentation was built entirely with Bokeh Server<br>
                All visualizations are live and interactive
            </p>
        </div>
        """,
            sizing_mode="stretch_width",
            height=120,
            align="center",
        )

        return self.stack([[title], [card1, card2], [thanks]])

    def update_slide(self):
        """Update current slide display and UI elements"""
        self._is_updating = True
        try:
            self.prev_button.disabled = self.current_slide == 0
            self.next_button.disabled = self.current_slide == self.total_slides - 1

            self.progress_div.text = self.get_progress_html()
            self.slide_select.value = str(self.current_slide)
            self.main_content.children = [self.slides[self.current_slide]]
        finally:
            self._is_updating = False

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
        if getattr(self, "_is_updating", False):
            return
        try:
            self.current_slide = int(new)
        except (ValueError, TypeError):
            for i in range(self.total_slides):
                if str(i) == str(new) or f"Slide {i + 1}" in str(new):
                    self.current_slide = i
                    break
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

        self.layout = column(
            nav_bar,
            separator,
            self.main_content,
            width=SLIDE_WIDTH,
            styles={"margin-left": "auto", "margin-right": "auto", "max-width": "100%"},
        )

        center_everything(self.layout, is_root=True)
        for slide in self.slides:
            center_everything(slide)

        self.update_slide()


presentation = InteractivePresentation()
curdoc().add_root(presentation.layout)
curdoc().title = "Wine: should you splash out on bottle?"