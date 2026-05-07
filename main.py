import pandas as pd
import data_tools
import os



#For each file in the report cards directory (not included in this repo)
#Report card files exported from swimtopia
swims = pd.DataFrame()

for root, dirs, files in os.walk("reportcards"):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            swims = pd.concat([swims, data_tools.report_card_to_swims(file_path, obscure = 0, age_limit=18)], ignore_index=True)

#Generate "swimscores" for each swim, swimscore = (swimtime in sec)/(best swim time in that age and sex in sec)
ss = data_tools.swim_score_from_swims(swims)

#Get all swims for each swimmer and iterate over the group (of swims)
#Group by sex as well, to help avoid name collisions
for (name, sex), group in ss.groupby(["FullName", "Sex"]):
    #get the max age from the list of swims for the swimmer, this should be their current age
    age = group["Age"].max()  # max age

    #get the events that a swimmer of that age could swim
    age_events = data_tools.events_from_age(swims, age)

    file_name = f"real_charts/{int(age)}-{name}.png"

    #create a chart that charts the swim scores based on the swims, and age events and save to file
    data_tools.build_swim_score_chart(group, age_events, file_name)



