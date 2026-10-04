#!/usr/bin/env python
import base64
import os
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
    HoverTool,
    LinearColorMapper,
    ColorBar,
    BasicTicker,
    PrintfTickFormatter,
)
from bokeh.layouts import column, row, layout
from bokeh.palettes import RdYlBu11
from bokeh.transform import factor_cmap


class InteractivePresentation:
    """
    Main application class for the Bokeh presentation system.
    Styled with crisp white plot containers, warm parchment background (#F7F5F0),
    user-specified hex colors for wine categories, and a black regression line.
    """

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

    def create_navigation(self):
        """Create themed navigation controls matching wine palette"""

        self.style_div = Div(
            text="""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Lusitania:wght@400;700&display=swap');

            /* Global Page Background & Centering Overrides */
            html, body {
                background-color: #F7F5F0 !important;
                font-family: 'Lusitania', Georgia, serif !important;
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
            }

            /* Center Bokeh container and all inner layout components */
            .bk-root {
                background-color: #F7F5F0 !important;
                width: 100% !important;
                display: flex !important;
                justify-content: center !important;
                margin: 0 auto !important;
                padding-top: 20px !important;
            }

            .bk-root > .bk {
                margin: 0 auto !important;
            }

            /* Force fixed layout blocks to auto-center */
            .bk-layout-fixed {
                margin-left: auto !important;
                margin-right: auto !important;
            }

            .bk-root input, .bk-root select, .bk-root button, .bk-root textarea {
                font-family: 'Lusitania', Georgia, serif !important;
            }

            /* Theme Buttons */
            .theme-btn button.bk-btn {
                font-family: 'Lusitania', Georgia, serif !important;
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
                background-color: #F7F5F0 !important;
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
                font-family: 'Lusitania', Georgia, serif !important;
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
                font-family: 'Lusitania', Georgia, serif !important;
                color: #AF1B3F !important;
                font-weight: bold !important;
                font-size: 13px !important;
                letter-spacing: 0.3px !important;
            }
        </style>
        """,
            width=0,
            height=0,
        )

        self.prev_button = Button(
            label="◀ Previous", width=105, css_classes=["theme-btn"]
        )
        self.next_button = Button(
            label="Next ▶", width=105, css_classes=["theme-btn"]
        )
        self.home_button = Button(
            label="🏠 Home", width=100, css_classes=["theme-btn"]
        )

        self.play_button = Button(
            label="▶ Auto Play", width=115, css_classes=["theme-btn"]
        )
        self.stop_button = Button(
            label="⏸ Stop", width=95, css_classes=["theme-btn"]
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
        )

        self.progress_div = Div(
            text=self.get_progress_html(),
            width=220,
        )

        self.prev_button.on_click(self.prev_slide)
        self.next_button.on_click(self.next_slide)
        self.home_button.on_click(self.go_home)
        self.play_button.on_click(self.start_auto_play)
        self.stop_button.on_click(self.stop_auto_play)
        self.slide_select.on_change("value", self.jump_to_slide)

    def get_slide_title(self, index):
        """Get title for each slide"""
        titles = [
            "Welcome",
            "Price vs Rating (Cubic Gamma GLM)",
            "Visual Vocabulary",
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
        <div style="font-family: 'Lusitania', Georgia, serif; text-align: center; padding: 2px 5px;">
            <div style="font-size: 12px; font-weight: bold; color: #AF1B3F; letter-spacing: 0.5px; margin-bottom: 4px;">
                SLIDE {self.current_slide + 1} OF {self.total_slides}
            </div>
            <div style="width: 100%; background-color: #FFFFFF; border: 1px solid #E2D7C3; height: 12px; border-radius: 6px; padding: 1px; box-sizing: border-box;">
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
        <div style="position: relative; text-align: center; background-color: #FFFFFF; border: 1px solid #E2D7C3; border-left: 6px solid #AF1B3F; padding: 100px 40px; border-radius: 12px; margin-top: 40px; overflow: hidden; box-shadow: 0 4px 15px rgba(175, 27, 63, 0.05);">
        
            <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; border-radius: 50%; border: 12px solid rgba(175, 27, 63, 0.08); box-shadow: inset 0 0 15px rgba(175, 27, 63, 0.05); pointer-events: none;"></div>
            <div style="position: absolute; top: -20px; right: -20px; width: 120px; height: 120px; border-radius: 50%; border: 6px solid rgba(255, 188, 66, 0.25); pointer-events: none;"></div>
        
            <div style="position: absolute; bottom: -50px; left: -30px; width: 160px; height: 160px; border-radius: 50%; border: 10px solid rgba(33, 131, 128, 0.1); transform: rotate(-15deg); pointer-events: none;"></div>

            <h1 style="position: relative; z-index: 1; font-size: 48px; color: #AF1B3F; font-family: 'Lusitania', serif; margin: 0 0 20px 0; font-weight: 700; line-height: 1.2;">
                Wine: should you splash out on the bottle?
            </h1>
            <p style="position: relative; z-index: 1; font-size: 22px; color: #5C4A42; font-style: italic; margin: 0; font-weight: 400;">
                Visualizing relationship between price, ratings and wine features
            </p>
        </div>
        """,
            width=1000,
            height=320,
        )

        return layout([[title_banner]])

    def create_slide_1_price_vs_rating(self):
        """Slide 1: Scatter plot of price vs rating with Cubic Gamma GLM Fit (Log Link)"""
        title = Div(
            text="""
             <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif; margin-bottom: 5px;">
                Average Rating vs. Price
            </h2>
            <p style="text-align: center; color: #5C4A42; font-size: 14px; margin-top: 0; margin-bottom: 15px;">
                    Select a wine category below to inspect rating and price distribution alongside a Cubic Gamma Regression fit (Log Link).
            </p>
            """,
            width=1000,
        )

        # Rating jittering
        rng = np.random.default_rng(123)
        df = self.df.copy()
        df["Rating_jittered"] = df["Rating"] + rng.normal(scale=0.02, size=len(df))

        kinds = ["Red", "White", "Rose", "Sparkling"]

        # Exact user-specified hex colors
        wine_colors = {
            "Red": "#AF1B3F",
            "White": "#FFBC42",
            "Rose": "#C99DA3",
            "Sparkling": "#218380"
        }

        # Helper function for Cubic Gamma Fit
        def compute_gamma_fit(sub_df, grid_points=100):
            valid = sub_df.dropna(subset=["Rating_jittered", "Price"])
            valid = valid[valid["Price"] > 0]

            if len(valid) < 4:
                return dict(x_fit=[], y_fit=[])

            x_vals = valid["Rating_jittered"].values
            y_vals = valid["Price"].values

            X = np.column_stack([x_vals, x_vals**2, x_vals**3])
            X = sm.add_constant(X)

            try:
                gamma_model = sm.GLM(
                    endog=y_vals,
                    exog=X,
                    family=sm.families.Gamma(link=sm.families.links.Log()),
                ).fit()

                x_grid = np.linspace(x_vals.min(), x_vals.max(), grid_points)
                X_grid = np.column_stack([x_grid, x_grid**2, x_grid**3])
                X_grid = sm.add_constant(X_grid)

                y_fit = gamma_model.predict(X_grid)
                return dict(x_fit=x_grid, y_fit=y_fit)
            except Exception:
                return dict(x_fit=[], y_fit=[])

        # ColumnDataSources
        initial_kind = "Red"
        initial_df = df[df["Kind"] == initial_kind]
        source = ColumnDataSource(data=ColumnDataSource.from_df(initial_df))
        fit_source = ColumnDataSource(data=compute_gamma_fit(initial_df))

        # Plot Setup
        p = figure(
            width=900,
            height=500,
            title=f"{initial_kind} Wine",
            y_axis_type="log",
            x_axis_label="Average Rating",
            y_axis_label="Price ($)",
            tools="pan,wheel_zoom,reset,hover,save",
        )

        p.background_fill_color = "#FFFFFF"
        p.background_fill_alpha = 1.0
        p.border_fill_color = "#F7F5F0"
        p.grid.grid_line_color = "#EAE5DC"
        p.grid.grid_line_alpha = 0.8

        # Scatter points
        scatter = p.scatter(
            x="Rating_jittered",
            y="Price",
            source=source,
            color=wine_colors[initial_kind],
            alpha=0.7,
            size=8.5,
            line_color="#FFFFFF",
            line_width=0.5
        )

        # Black Fit Line (#000000)
        fit_line = p.line(
            x="x_fit",
            y="y_fit",
            source=fit_source,
            color="#000000",
            line_width=3,
            legend_label="Cubic Gamma GLM Fit"
        )
        p.legend.location = "top_left"
        p.legend.background_fill_alpha = 0.85

        # Hover Tool
        hover = p.select_one(HoverTool)
        hover.renderers = [scatter]

        compact_tooltip_html = """
        <div style="max-width: 190px; max-height: 110px; overflow: hidden; font-size: 11px; padding: 6px 8px; border-radius: 4px; background: #FFFFFF; border: 1.5px solid #AF1B3F; box-shadow: 0 2px 6px rgba(0,0,0,0.1); line-height: 1.4;">
            <div style="font-weight: bold; color: #AF1B3F; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #E2D7C3; padding-bottom: 2px; margin-bottom: 4px;">
                @Label
            </div>
            <div style="color: #5C4A42; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                @Country &bull; @Region
            </div>
            <div style="color: #5C4A42;"><b>Vintage:</b> @Vintage</div>
            <div style="margin-top: 3px; border-top: 1px dashed #E2D7C3; padding-top: 3px;">
                <b style="color: #AF1B3F;">@Rating pts</b> | <b style="color: #218380;">$@Price{0.00}</b>
            </div>
        </div>
        """

        hover.tooltips = compact_tooltip_html
        hover.point_policy = "snap_to_data"
        hover.mode = "mouse"

        # Toggle Controls
        toggle = RadioGroup(
            labels=kinds,
            active=0,
            inline=True,
            width=400,
        )

        def update_plot(attr, old, new):
            selected_kind = kinds[new]
            p.title.text = f"{selected_kind} Wine"

            new_df = df[df["Kind"] == selected_kind]
            source.data = ColumnDataSource.from_df(new_df)
            fit_source.data = compute_gamma_fit(new_df)

            # Update scatter point color using user-defined hex colors
            scatter.glyph.fill_color = wine_colors[selected_kind]
            scatter.glyph.line_color = wine_colors[selected_kind]

        toggle.on_change("active", update_plot)
        toggle_container = row(toggle, align="center")

        return layout([[title], [toggle_container], [p]])

    def create_slide_2_visual_vocabulary(self):
        """Slide 2: Visual Vocabulary"""
        title = Div(
            text="""
        <h1 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif;">
            📊 Visual Vocabulary - Financial Times Guide
        </h1>
        <p style="text-align: center; font-size: 16px; color: #5C4A42;">
            A comprehensive guide to selecting the right chart type for your data story
        </p>
        """,
            width=1000,
            height=100,
        )

        image_path = os.path.join(
            os.path.dirname(__file__), "visual-vocabulary-ft.png"
        )
        image_html = ""

        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
                image_html = f"""
        <div style="text-align: center; margin: 10px auto;">
            <img src="data:image/png;base64,{encoded_string}" 
                 style="max-width: 100%; height: auto; max-height: 550px; border: 1px solid #E2D7C3; border-radius: 6px; box-shadow: 0 4px 10px rgba(175, 27, 63, 0.06);"
                 alt="Visual Vocabulary - Financial Times">
        </div>
        """
        else:
            image_html = """
        <div style="text-align: center; margin: 20px auto; padding: 50px; background-color: #FFFFFF; border: 1px solid #E2D7C3; border-radius: 8px;">
            <h3 style="color: #AF1B3F;">Visual Vocabulary Image</h3>
            <p>Image file not found: visual-vocabulary-ft.png</p>
        </div>
        """

        image_div = Div(text=image_html, width=1000, height=580)

        info_panel = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 15px 20px; border-radius: 8px; color: #211B18;">
            <h3 style="color: #AF1B3F; margin-top: 0;">📌 About the Visual Vocabulary</h3>
            <p style="margin-bottom: 0;">The Financial Times Visual Vocabulary is a poster and guide that helps you select the most appropriate chart type based on the story you want to tell with your data.</p>
        </div>
        """,
            width=1000,
            height=100,
        )

        return layout([[title], [image_div], [info_panel]])

    def create_slide_3_overview(self):
        """Slide 3: Data Overview Dashboard"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif;">📈 Data Overview Dashboard</h2>
        <p style="text-align: center; color: #5C4A42;">Multiple synchronized visualizations showing different aspects of the dataset</p>
        """,
            width=1000,
        )

        categories = ["Product A", "Product B", "Product C", "Product D", "Product E"]
        bar_data = pd.DataFrame(
            {
                "categories": categories,
                "values": np.random.randint(50, 200, len(categories)),
            }
        )
        bar_source = ColumnDataSource(bar_data)

        line_data = pd.DataFrame(
            {"x": range(50), "y": np.cumsum(np.random.randn(50)) + 100}
        )
        line_source = ColumnDataSource(line_data)

        p1 = figure(
            x_range=categories,
            width=480,
            height=280,
            title="Sales by Product",
            toolbar_location="above",
        )
        p1.vbar(
            x="categories",
            top="values",
            width=0.8,
            source=bar_source,
            color=factor_cmap(
                "categories",
                palette=["#AF1B3F", "#FFBC42", "#C99DA3", "#218380", "#000000"],
                factors=categories,
            ),
        )
        p1.y_range.start = 0

        p2 = figure(width=480, height=280, title="Trend Analysis")
        p2.line("x", "y", source=line_source, line_width=2, color="#AF1B3F")
        p2.scatter("x", "y", source=line_source, size=5, color="#AF1B3F", alpha=0.6)

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        heatmap_data = []
        for month in months:
            for day in days:
                heatmap_data.append((month, day, np.random.randint(0, 100)))

        hm_source = ColumnDataSource(
            data=dict(
                months=[x[0] for x in heatmap_data],
                days=[x[1] for x in heatmap_data],
                values=[x[2] for x in heatmap_data],
            )
        )

        p3 = figure(
            x_range=months,
            y_range=days,
            width=480,
            height=280,
            title="Activity Heatmap",
            toolbar_location="above",
        )

        mapper = LinearColorMapper(palette=RdYlBu11[::-1], low=0, high=100)
        p3.rect(
            x="months",
            y="days",
            width=1,
            height=1,
            source=hm_source,
            fill_color={"field": "values", "transform": mapper},
        )

        color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
        p3.add_layout(color_bar, "right")

        stats = Div(
            text=f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 235px; box-sizing: border-box;">
            <h3 style="color: #AF1B3F; margin-top: 0;">📊 Key Metrics:</h3>
            <table style="width: 100%; font-size: 14px; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #E2D7C3;"><td style="padding: 8px 0;"><b>Total Products:</b></td><td>{len(categories)}</td></tr>
                <tr style="border-bottom: 1px solid #E2D7C3;"><td style="padding: 8px 0;"><b>Average Sales:</b></td><td>${bar_data["values"].mean():.2f}</td></tr>
                <tr><td style="padding: 8px 0;"><b>Max Sales:</b></td><td>${bar_data["values"].max()}</td></tr>
            </table>
        </div>
        """,
            width=480,
            height=280,
        )

        return layout([[title], [p1, p2], [p3, stats]])

    def create_slide_4_interactive(self):
        """Slide 4: Interactive Analysis with Controls"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif;">🎮 Interactive Data Explorer</h2>
        <p style="text-align: center; color: #5C4A42;">Adjust parameters to explore different data visualizations</p>
        """,
            width=1000,
        )

        self.slide4_source = ColumnDataSource(data=dict(x=[], y=[]))

        p = figure(width=620, height=400, title="Interactive Function Plotter")
        self.slide4_line = p.line(
            "x", "y", source=self.slide4_source, line_width=2, color="#AF1B3F"
        )

        self.func_select = Select(
            title="Function:",
            value="sin",
            options=["sin", "cos", "exp", "log", "polynomial"],
            css_classes=["theme-select"],
            width=320,
        )
        self.param_slider = Slider(
            start=0.1, end=5, value=1, step=0.1, title="Parameter", width=320
        )
        self.points_slider = Slider(
            start=50, end=500, value=100, step=50, title="Number of Points", width=320
        )
        self.noise_slider = Slider(
            start=0, end=1, value=0, step=0.05, title="Noise Level", width=320
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
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 15px; border-radius: 8px; color: #211B18;">
            <h3 style="color: #AF1B3F; margin-top: 0;">🎯 Try These:</h3>
            <ul style="font-size: 13px; margin-bottom: 0;">
                <li>Change function type</li>
                <li>Adjust parameter to modify shape</li>
                <li>Add noise for realistic data</li>
            </ul>
        </div>
        """,
            width=320,
            height=150,
        )

        controls = column(
            self.func_select,
            self.param_slider,
            self.points_slider,
            self.noise_slider,
            info,
        )

        return layout([[title], [p, controls]])

    def create_slide_5_timeseries(self):
        """Slide 5: Time Series Analysis"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif;">📅 Time Series Analysis</h2>
        <p style="text-align: center; color: #5C4A42;">Exploring temporal patterns and trends</p>
        """,
            width=1000,
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
            width=960,
            height=380,
            x_axis_type="datetime",
            title="Time Series with Moving Averages",
        )

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
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 15px 20px; border-radius: 8px; color: #211B18;">
            <h3 style="color: #AF1B3F; margin-top: 0;">📈 Time Series Statistics:</h3>
            <table style="width: 100%; font-size: 14px;">
                <tr><td><b>Period:</b> {dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}</td><td><b>Mean:</b> {np.mean(values):.2f}</td></tr>
            </table>
        </div>
        """,
            width=960,
            height=100,
        )

        return layout([[title], [p], [stats]])

    def create_slide_6_correlation(self):
        """Slide 6: Correlation Matrix Explorer"""
        title = Div(
            text="""
        <h2 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif;">🔗 Correlation Analysis</h2>
        <p style="text-align: center; color: #5C4A42;">Exploring relationships between variables</p>
        """,
            width=1000,
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
            width=540,
            height=460,
            title="Correlation Matrix",
            toolbar_location="above",
            tools="hover,save",
        )

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
            width=8,
            location=(0, 0),
            ticker=BasicTicker(desired_num_ticks=10),
            formatter=PrintfTickFormatter(format="%.1f"),
        )
        p.add_layout(color_bar, "right")

        guide = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 200px;">
            <h3 style="color: #AF1B3F; margin-top: 0;">📊 Interpretation Guide:</h3>
            <p style="font-size: 13px; line-height: 1.5;">Hover over individual cells to inspect precise correlation coefficients between dataset variables.</p>
        </div>
        """,
            width=360,
            height=240,
        )

        return layout([[title], [p, guide]])

    def create_slide_7_conclusions(self):
        """Slide 7: Conclusions and Summary"""
        title = Div(
            text="""
        <h1 style="text-align: center; color: #AF1B3F; font-family: 'Lusitania', serif;">
            🎯 Conclusions & Key Takeaways
        </h1>
        """,
            width=1000,
            height=80,
        )

        card1 = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 180px;">
            <h3 style="color: #AF1B3F; margin-top: 0;">✅ What We've Demonstrated</h3>
            <ul style="font-size: 14px; line-height: 1.5;">
                <li>Interactive visualizations with real-time updates</li>
                <li>Multiple chart types and responsive layouts</li>
                <li>Custom theme synchronization</li>
            </ul>
        </div>
        """,
            width=460,
            height=220,
        )

        card2 = Div(
            text="""
        <div style="background-color: #FFFFFF; border: 1px solid #E2D7C3; padding: 20px; border-radius: 8px; color: #211B18; height: 180px;">
            <h3 style="color: #AF1B3F; margin-top: 0;">🚀 Bokeh Advantages</h3>
            <ul style="font-size: 14px; line-height: 1.5;">
                <li>Python callbacks for complex logic</li>
                <li>Real-time data streaming</li>
                <li>Server-side computation</li>
            </ul>
        </div>
        """,
            width=460,
            height=220,
        )

        thanks = Div(
            text="""
        <div style="text-align: center; margin-top: 20px; font-family: 'Lusitania', serif;">
            <h2 style="color: #AF1B3F;">Thank You! 🙏</h2>
            <p style="font-size: 16px; color: #5C4A42;">
                This presentation was built entirely with Bokeh Server<br>
                All visualizations are live and interactive
            </p>
        </div>
        """,
            width=1000,
            height=120,
        )

        return layout([[title], [card1, card2], [thanks]])

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

    def start_auto_play(self):
        if not self.auto_play:
            self.auto_play = True
            self.auto_play_callback = curdoc().add_periodic_callback(
                self.auto_advance, 5000
            )
            self.play_button.label = "⏸ Pause"
            self.play_button.css_classes = ["theme-btn", "theme-btn-active"]

    def stop_auto_play(self):
        if self.auto_play:
            self.auto_play = False
            if self.auto_play_callback:
                curdoc().remove_periodic_callback(self.auto_play_callback)
            self.play_button.label = "▶ Auto Play"
            self.play_button.css_classes = ["theme-btn"]

    def auto_advance(self):
        self.next_slide()

    def create_layout(self):
        """Assemble main presentation layout centered horizontally"""

        separator = Div(
            text="<hr style='border: 0; height: 1.5px; background-color: #E2D7C3; margin: 12px 0 20px 0;'>",
            width=1000,
            height=15,
        )

        nav_bar = row(
            self.style_div,
            self.prev_button,
            self.home_button,
            self.next_button,
            self.slide_select,
            self.play_button,
            self.stop_button,
            self.progress_div,
        )

        self.main_content = column(self.slides[0])

        self.layout = column(
            nav_bar,
            separator,
            self.main_content,
        )

        self.update_slide()


presentation = InteractivePresentation()
curdoc().add_root(presentation.layout)
curdoc().title = "Wine: should you splash out on bottle?"