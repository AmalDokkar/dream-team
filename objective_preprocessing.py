import json, re, string
import numpy as np

f = open('/home/jordina/Desktop/datathon24/datathon_participants.json')

data = json.load(f)
n = len(data)

dic = {
    "prize-hunting" : [ 
        "come out on top", "trophy", "competition", "prize", "gold", "victory", "come out on top", "outsmart and outwork",
        "game changing solution", "bring it on", "win", "push myself", "first place", "trophy", "compete", "competing", 
        "crush", "all in", "gold", "contender", "top spot", "champion" ,"victorious", "crown", "laser focus",
    ],
    "portfolio-building": [
        "portfolio", "techniques", "machine learning", "projects", "abilities", "hands-on experience", 
        "applications", "show off", "projects that stand out", "expand my toolkit", "solutions"
    ],
    "learning new skills" : [
        "growth", "learn", "skill", "better programmer","mentorship sessions", "technologies", "better programmer", 
        "exposed to ideas",  "push my limits", "tool", "expand my knowledge", "level up", "challenges", "improve my coding", 
        "expertise", "step up my game", "absorb knowledge", "refine my techniques"
    ],
    "meeting new people": [
        "meet", "vibes", "awesome new friends", "connections", "community", "vibing", "sharing stories", "picking brains", 
        "try new things", "memories", "networking", "collaborating", "conversation", "expand my network", 
        "work with talented people", "sharing ideas", "work with others", "friends", "team building", 
        "meeting fellow coders", "building relationships", "find a coding buddy", "gather new perspectives", "friendship", 
        "socializing", "hang out", "buddy"
    ]
}

vector = []

def remove_punctuation(s):
    ans = ""
    for c in s:
        if 'a' <= c <= 'z' or c == "'" or c == ' ':
            ans += c
    return ans

count = 0
for i in data:
    v = [0, 0, 0, 0]
    s = i["objective"].lower()
    sentences = re.split(r'(?<=[.!?,]) +', s)
    for j in sentences:
        j = remove_punctuation(j)

        if "not" in j or "n't" in j:
            flag = False
        else:
            flag = True
        
        if flag:
            for i in range(len(j)):
                for expr in dic['prize-hunting']:
                    if(j[i: i+len(expr)] == expr):
                        v[0] += 1
                for expr in dic['portfolio-building']:
                    if(j[i: i+len(expr)] == expr):
                        v[1] += 1
                for expr in dic['learning new skills']:
                    if(j[i: i+len(expr)] == expr):
                        v[2] += 1
                for expr in dic['meeting new people']:
                    if(j[i: i+len(expr)] == expr):
                        v[3] += 1

    norma = np.linalg.norm(v)
    v[0] /= norma
    v[1] /= norma
    v[2] /= norma
    v[3] /= norma

    vector.append(v)


for i, person in enumerate(data):
    person['objective'] = vector[i]

f.close()

with open('/home/jordina/Desktop/datathon24/datathon_participants_updated.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)