import json 

f = open('data/scenario_chicago.json', 'r')
data = json.load(f)

print(data["topology_graph"])
