import pandas as pd
import data_tools
import os


swims = pd.DataFrame()

for root, dirs, files in os.walk("reportcards"):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            swims = pd.concat([swims, data_tools.report_card_to_swims(file_path, obscure = 0, age_limit=18)], ignore_index=True)


ss = data_tools.swim_score_from_swims(swims)


for (name, sex), group in ss.groupby(["FullName", "Sex"]):
    age = group["Age"].max()  # max age
    age_events = data_tools.events_from_age(swims, age)
    file_name = f"real_charts/{int(age)}-{name}.png"
    data_tools.build_swim_score_chart(group, age_events, file_name)



