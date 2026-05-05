import numpy as np
import pandas as pd
import re
import math

def report_card_to_swims(csv_path, obscure = 0, age_limit = 99):
    rc = pd.read_csv(csv_path)
    # Find all meet numbers present in the columns
    meet_nums = set()
    for col in rc.columns:
        m = re.match(r"Meet(\d+)-Name", col)
        if m:
            meet_nums.add(int(m.group(1)))
    meet_nums = sorted(meet_nums)

    swims = []
    for _, row in rc.iterrows():
        if row["Age"] > age_limit:
            continue
        sex_str = str(row.get("AgeGroup", "")).lower()
        base_info = {
            "AgeGroup": row["AgeGroup"],
            "Sex": ("F" if re.search(r"\b(girls|women|female|f)\b", sex_str)
                    else ("M" if re.search(r"\b(boys|men|male|m)\b", sex_str) else None)),
            "AthleteId": row["AthleteId"],
            "LastName": row["LastName"] if obscure == 0 else hash(row["LastName"]),
            "FirstName": row["FirstName"] if obscure == 0 else hash(row["FirstName"]),
            "FullName": row["FirstName"] + " " + row["LastName"] if obscure == 0 else str(hash(row["FirstName"])) + str(hash(row["LastName"])) ,
            "Age": row["Age"],
            "EventDistance": row["EventDistance"],
            "EventStroke": row["EventStroke"],
            "FullEvent": str(row["EventDistance"])+row['EventStroke']
        }
        for n in meet_nums:
            meet_name = row.get(f"Meet{n}-Name")
            result = row.get(f"Meet{n}-Result")
            result_sec = row.get(f"Meet{n}-ResultSec")
            date_str = row.get(f"Meet{n}-Date")
            if math.isnan(result_sec):
                continue
            if pd.notna(result) and result != "":
                swim = base_info.copy()
                swim.update({
                    "Meet": meet_name,
                    "Result": result,
                    "ResultSec": result_sec,
                    "Date": np.datetime64(pd.to_datetime(date_str, format="%m/%d/%y")) if pd.notna(date_str) and date_str != "" else None, #as date object in python
                })
                swims.append(swim)
    return pd.DataFrame(swims)


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D



#Retrieved from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def events_from_age(swims, age):
    #Get Age Event time and Info
    events_by_age = swims.filter(["Age", "FullEvent", "EventDistance", "EventStroke"])
    events_by_age = events_by_age[events_by_age["Age"] == age]

    #Data Cleanup and sorting
    events_by_age = (events_by_age[events_by_age["Age"] > 0 ]).drop_duplicates()
    events_by_age = events_by_age.sort_values(by=["Age", "EventStroke", "EventDistance"]).reset_index(drop=True)


    return events_by_age


def swim_score_from_swims(swims):
    #Get Swimmer PersonalRecords by event and their max (current) age
    prs = swims.groupby(["Sex", "FullName", "FullEvent", "EventDistance", "EventStroke"], as_index=False).agg(
        ResultSec=("ResultSec", "min"),
        Age=("Age", "max"),
    )

    # Get min time for each event and age combo
    event_age_min = (
        swims
        .groupby(["Sex", "FullEvent", "Age"], as_index=False)
        .agg(MinResultSec=("ResultSec", "min"))
    )

    # Merge benchmarks into PRs
    prs = prs.merge(
        event_age_min,
        on=["Sex", "FullEvent", "Age"],
        how="left",
        validate="many_to_one"  # ensures clean merge (each PR maps to one benchmark)
    )

    # Calculate swim score as % of max time for that event/age
    prs["TeamAgeSwimScore"] = (
        (prs["MinResultSec"] / prs["ResultSec"]) * 100
    ).round(1)

    # data clean up
    scores = (
        prs[
            (prs["ResultSec"] > 0) &
            (prs["MinResultSec"] > 0)
        ]
        .sort_values(["Age", "FullName", "EventStroke", "EventDistance"])
        .reset_index(drop=True)
    )

    return scores



def build_swim_score_chart(swimmer_scores, events, path):
    N = np.size(events, 0)
    theta = radar_factory(N, frame='polygon')

    # Align swimmer_scores to the event order provided by `events` and fill null swims with 0
    name = swimmer_scores["FullName"].dropna().unique()[0]
    swimmer_scores = swimmer_scores.set_index("FullEvent").reindex(events["FullEvent"]).fillna({"FullName": name, "TeamAgeSwimScore": 0})

    fig, ax = plt.subplots(subplot_kw={'projection': 'radar'})
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    spoke_labels = events["FullEvent"].to_list()


    values = swimmer_scores["TeamAgeSwimScore"].to_list()

    ax.plot(theta, values)
    ax.fill(theta, values, facecolor="blue", alpha=0.25, label='_nolegend_')

    title = f"{swimmer_scores.iloc[0]["FullName"]} ({int(swimmer_scores["Age"].max())})"
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                        horizontalalignment='center', verticalalignment='center')

    ax.set_xticks(theta)
    ax.set_xticklabels(spoke_labels)
    ax.set_yticklabels([])


    ax.set_yticks([20, 40, 60, 80, 100])

    # Return the Figure and Axes so caller can show/save/modify the plot
    if path is not None:
        plt.savefig(path)
    plt.close()