import re
import time
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import accumulate
import math
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

# Color codes/labels used in terminal text and plotting charts
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

# Default no. of worker threads used in run() to speed up API calls
WORKERS = 5

# Default size of items to plot
DEFAULT_SIZE = 200

# Helper function to create average cost label
def format_cost(average_cost: float) -> str:
    """Per-item API cost, shown per 1,000 items while that reads better than a row of zeros."""
    return f"${average_cost:,.2f} each" if average_cost >= 0.01 else f"${average_cost * 1000:,.3f} per 1k"

# Helper function to generate average latency label
def format_latency(average_seconds: float) -> str:
    return f"{average_seconds * 1000:,.0f}ms" if average_seconds < 1 else f"{average_seconds:,.2f}s"

# Class to test the model's prediction of item prices
class Tester:

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, workers=WORKERS):
        
        # Function to call model for prediction and inference cost
        self.predictor = predictor

        # Dataset of items
        self.data = data

        self.title = title or self.make_title(predictor)
        
        self.size = size
        
        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []
        self.costs = []
        self.latencies = []
        
        self.elapsed = 0.0
        
        # No. of worker threads to be used for testing
        self.workers = workers

    # Helper function to generate title to be used in chart plots
    @staticmethod
    def make_title(predictor) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    # Prediction value from model is a string, this helper function converts the string to a float 
    # value for computations
    @staticmethod
    def post_process(value):
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0
        else:
            return value

    # Helper function to generate color label for prediction
    def color_for(self, error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    # Processes an item to generate data (i.e. cost, latency, error, color, title) to be plotted in chart 
    def run_datapoint(self, i):
        datapoint = self.data[i]
        started = time.perf_counter()
        
        value, cost = self.predictor(datapoint)

        latency = time.perf_counter() - started
        guess = self.post_process(value)
        truth = datapoint.price
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        return title, guess, truth, error, color, cost, latency

    # Draw the scatter plot of predicted vs. actual price
    def chart(self, title):
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )

        # Pre-format hover text
        df["hover"] = [
            f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]

        # Establish shared axis limit, make the x- and y-axes span the same range
        max_val = float(max(df["truth"].max(), df["guess"].max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price", "guess": "Predicted Price"},
            width=1000,
            height=800,
        )

        # Assign customdata per trace (one color/category = one trace)
        for tr in fig.data:
            # Element-wise comparison to produce a boolean panda Series - True when the row's color matched the trace's name
            mask = df["color"] == tr.name

            # Boolean indexing to keep only the rows where mask is True i.e. only the rows belonging to the trace's color
            #
            # ["hover"] is the column selector that returns the 2-D DataFrame with one column
            #
            # to_numpy() converts the DataFrame, into a 2-D array shaped (n_rows, 1)
            #
            # customdata is a Plotly trace attribute that lets you attach arbitrary extra data to each point, 
            # data that isn't x or y
            # It is row-aligned with the trace's points, one row per point, one or more columns per point
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()

            # customdata[0] picks column 0 of each point's row in customdata, i.e. that point's hover string.
            # <extra></extra> is a Plotly hover-template tag that tells Plotly to hide the secondary hover box
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"

            tr.marker.update(size=6)

        # Reference line y=x
        # Append a 4th trace, the dashed diagonal line, directly onto fig.data
        # Is is not derived from df but built from 2 points: the origin and max x, y values
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",

                # Skips hover entirely for that trace (no tooltip at all, as the line is just a visual guide and 
                # hovering over it wouldn't be informative).
                hoverinfo="skip",

                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    # Draw line chart showing how the average error evolves as more test items are scored, with shaded confidence 
    # band around it.
    def error_trend_chart(self):
        n = len(self.errors)

        # Running mean and std (pure Python)
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        # 95% confidence interval for mean
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        # Plot
        fig = go.Figure()

        # Shaded confidence interval band
        fig.add_trace(
            go.Scatter(
                # Draw the upper curve left-to-right, then the lower curve right-to-left (x[::-1] reverses the list), 
                # so the path traces the top, then loops back along the bottom, forming a closed shape.
                x=x + x[::-1],
                y=upper + lower[::-1],

                # Shades the interior gray, and the invisible line color hides the outline
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                
                line=dict(color="rgba(255,255,255,0)"),

                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )

        # Main line with hover text showing confidence interval
        fig.add_trace(
            go.Scatter(

                # Solid red line for the running mean, sitting inside the shaded CI band
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),

                name="Cumulative Avg Error",
                customdata=list(
                    zip(
                        ci,
                    )
                ),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=$%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

        # Title with final stats
        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} Error: ${final_mean:,.2f} ± ${final_ci:,.2f}"

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    # Computes summary stats, builds a headline string out of them, and triggers both charts
    def report(self):
        # Derived summary statistics
        average_error = sum(self.errors) / self.size
        average_cost = sum(self.costs) / self.size
        average_latency = sum(self.latencies) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100

        title = (
            f"{self.title} results<br><b>Error:</b> ${average_error:,.2f} "
            f"<b>MSE:</b> {mse:,.0f} <b>r²:</b> {r2:.1f}% "
            f"<b>Cost:</b> {format_cost(average_cost)} <b>Time:</b> {format_latency(average_latency)}"
        )
        self.error_trend_chart()
        self.chart(title)

    def run(self):
        started = time.perf_counter()

        # Invoke run_datapoint() on all the items in parallel, using a pool of worker threads (5), and processes each 
        # result as it comes back.
        
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            # Wrapping the map in tqdm(..., total=self.size) draws a progress bar that advances by one each time a result 
            # is yielded. total is passed explicitly because ex.map's return value doesn't expose a length on its own.
            for title, guess, truth, error, color, cost, latency in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
                self.costs.append(cost)
                self.latencies.append(latency)
                print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
        self.elapsed = time.perf_counter() - started
        print(
            f"{RESET}\nTotal API cost ${sum(self.costs):,.4f} for {self.size} items, "
            f"{self.elapsed:,.1f}s wall clock with {self.workers} workers"
        )
        
        # All items have been scored, self's seven lists are fully populated in item order, and the total API cost,  
        # elapsed time computed, so that the two charts in report() can be generated from this same data set.
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    """The predictor is called with an Item and returns (estimate, inference_cost)."""
    Tester(function, data, size=size, workers=workers).run()
