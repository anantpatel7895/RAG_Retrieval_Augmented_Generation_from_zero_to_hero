import numpy as np

DIMS = ["programming", "ml_ai", "databases", "cooking", "health",
        "finance", "travel", "sports", "music_arts", "science"]

# (chunk title, {dimension: weight})  -> 10-D embedding, unit-normalised
RAW = [
 ("Python for beginners",          {"programming": .9, "databases": .2}),
 ("Intro to Java",                 {"programming": .9}),
 ("SQL joins explained",           {"programming": .4, "databases": .9}),
 ("Postgres indexing guide",       {"programming": .3, "databases": .9}),
 ("Intro to machine learning",     {"ml_ai": .9, "databases": .3, "programming": .3}),
 ("Neural networks explained",     {"ml_ai": .95, "programming": .3, "science": .2}),
 ("Vector DBs for RAG",            {"ml_ai": .6, "databases": .8, "programming": .3}),
 ("Pasta carbonara recipe",        {"cooking": .95, "travel": .15}),
 ("Sourdough bread baking",        {"cooking": .9, "science": .2}),
 ("Healthy meal prep",             {"cooking": .6, "health": .7}),
 ("Marathon training plan",        {"health": .6, "sports": .8}),
 ("Strength training basics",      {"health": .8, "sports": .5}),
 ("Index funds for beginners",     {"finance": .95}),
 ("Stock market basics",           {"finance": .9, "databases": .2}),
 ("Budget travel in Italy",        {"travel": .9, "finance": .4, "cooking": .3}),
 ("Hiking in the Alps",            {"travel": .6, "sports": .5, "science": .5}),
 ("Jazz piano improvisation",      {"music_arts": .95}),
 ("Climate change science",        {"science": .9, "databases": .2}),
 ("Fraud detection with ML",       {"finance": .6, "ml_ai": .7, "databases": .3}),
 ("Sports analytics in Python",    {"sports": .6, "databases": .5, "programming": .5}),
]

def vec(d):
    v = np.zeros(len(DIMS))
    for k, w in d.items():
        v[DIMS.index(k)] = w
    return v / np.linalg.norm(v)

NAMES = [n for n, _ in RAW] # chunk titles examples : ["Python for beginners", "Intro to Java", ...]
X = np.array([vec(d) for _, d in RAW]) # 10-D embeddings for each chunk title, unit-normalized