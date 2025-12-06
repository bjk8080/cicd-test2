import json
import csv
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# SETTINGS
# ============================================

MOVIE_FILE = "datas/trend_data_with_ai_tags.json"
TAGS_2019_FILE = "datas/tags.csv"
CURRENT_YEAR = datetime.now().year

# ✅ 최종 점수 가중치 (100점 기준)
W_TAG     = 0.50   # 태그 50
W_RATING  = 0.10   # 평점 10
W_RECENT  = 0.10   # 최신성 10
W_POP     = 0.30   # 인기 30

# ✅ Hybrid Tag 가중치
DIRECT_TAG_WEIGHT   = 0.6
EMBEDDING_WEIGHT    = 0.4

# ============================================
# SBERT MODEL
# ============================================

print("🔥 SBERT 모델 로딩...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================
# LOAD DATA
# ============================================

def load_json(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def load_tags_csv(path):
    tags=defaultdict(list)
    with open(path,newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for r in reader:
            try:
                tags[int(r["movieId"])].append(r["tag"])
            except:
                pass
    return tags

print("📂 데이터 로딩 중...")
movies=load_json(MOVIE_FILE)
tags_2019=load_tags_csv(TAGS_2019_FILE)
MAX_POP=max(m.get("popularity",1) for m in movies)

# ============================================
# UTIL
# ============================================

def year_of(m):
    try:
        return int(m["release_date"][:4])
    except:
        return None

def get_tags(m):
    y=year_of(m)
    if y and y<=2019 and "movieId" in m:
        return [(t,1.0) for t in tags_2019.get(m["movieId"],[])]
    return [(t["tag"],float(t.get("score",1))) for t in m.get("predicted_tags",[])]

# ============================================
# TAG EMBEDDING PREBUILD
# ============================================

print("🧠 태그 임베딩 생성중...")
all_tags = list({
    t for m in movies for t,_ in get_tags(m)
})

tag_vectors = {}
embeddings = embed_model.encode(all_tags, normalize_embeddings=True)
for t,v in zip(all_tags, embeddings):
    tag_vectors[t]=v

# ============================================
# MOVIE EMBEDDING VECTOR
# ============================================

movie_vectors = {}

for m in movies:
    tags=get_tags(m)
    if not tags:
        continue

    vecs=[]
    for t,_ in tags:
        if t in tag_vectors:
            vecs.append(tag_vectors[t])

    if vecs:
        movie_vectors[id(m)] = np.mean(vecs,axis=0)

# ============================================
# GENRE / OTT LIST
# ============================================

all_genres = sorted({g["name"] for m in movies for g in m.get("genres",[])})
all_otts = sorted({p["provider_name"] for m in movies for p in m.get("providers",[])})

GENRE_MAP = {str(i+1):g for i,g in enumerate(all_genres)}
OTT_MAP   = {str(i+1):o for i,o in enumerate(all_otts)}

def choose_from_map(label, mapping):
    print(f"\n===== {label} =====")
    for k,v in mapping.items():
        print(f"{k}. {v}")
    raw=input("\n번호 선택(쉼표 가능, 엔터=전체): ").strip()
    if not raw: return []
    return [mapping[x.strip()] for x in raw.split(",") if x.strip() in mapping]

# ============================================
# POPULARITY POOL
# ============================================

def popularity_pool(n=300):
    pool=sorted(movies,key=lambda x:x.get("popularity",0),reverse=True)
    pool=[m for m in pool if m.get("vote_average",0)>=6.5]
    return pool[:n]

# ============================================
# PERSONAL LEARN
# ============================================

def collect_preferences(k=10):

    personal=defaultdict(float)
    vectors=[]
    pool=popularity_pool()
    samples=random.sample(pool,min(k,len(pool)))

    print("\n🎯 취향 학습 시작")

    for i,m in enumerate(samples,1):
        print(f"{i}. {m['title']} ({year_of(m)})")
        print(f" ⭐ {m['vote_average']} | 🔥 {m['popularity']:.1f}")
        print(" ",m.get("overview","")[:100]+"...")

        ans=input(" 취향?(Y/N): ").strip().upper()
        pos = ans=="Y"
        factor = 1 if pos else -0.5

        for t,w in get_tags(m):
            personal[t]+=factor*w

        if pos and id(m) in movie_vectors:
            vectors.append(movie_vectors[id(m)])

    user_vector = np.mean(vectors,axis=0) if vectors else None

    return personal, user_vector

# ============================================
# SCORE
# ============================================

def score(movie, pref, user_vec):

    # ----- Direct Tag Score -----
    tags=get_tags(movie)

    direct_details=[]
    raw_sum=0

    for t,w in tags:
        val=pref.get(t,0)*w
        raw_sum+=val
        direct_details.append((t,val))

    direct_score = raw_sum/max(1,len(tags))

    # ----- Embedding Similarity -----
    sim_score=0
    if user_vec is not None and id(movie) in movie_vectors:
        sim_score = float(
            cosine_similarity(
                [user_vec],
                [movie_vectors[id(movie)]]
            )[0][0]
        )
        sim_score = max(0,sim_score)

    hybrid_tag_score = (direct_score*DIRECT_TAG_WEIGHT +
                        sim_score*EMBEDDING_WEIGHT)

    # ----- Semantic scores -----
    rate_score=(movie.get("vote_average",0))/10
    y=year_of(movie)
    rec_score=max(0,1-(CURRENT_YEAR-y)/30) if y else 0
    pop_score=(movie.get("popularity",0)/MAX_POP)

    # ----- 100 Point system -----
    tag_100 = hybrid_tag_score*W_TAG*100
    rate_100 = rate_score*W_RATING*100
    rec_100 = rec_score*W_RECENT*100
    pop_100 = pop_score*W_POP*100

    final = tag_100 + rate_100 + rec_100 + pop_100

    breakdown=[
        (t, v*DIRECT_TAG_WEIGHT*W_TAG*100)
        for t,v in direct_details if abs(v)>0.01
    ]

    return {
        "tag": tag_100,
        "rating": rate_100,
        "recent": rec_100,
        "pop": pop_100,
        "final": final,
        "tag_detail": breakdown,
        "sim": sim_score*EMBEDDING_WEIGHT*W_TAG*100
    }

# ============================================
# FILTER
# ============================================

def filter_movies(max_runtime, genres, otts):

    results=[]
    for m in movies:

        r=m.get("runtime")
        if not r or not(max_runtime-10<=r<=max_runtime):
            continue

        if genres:
            gnames=[g["name"] for g in m.get("genres",[])]
            if not any(g in gnames for g in genres):
                continue

        if otts:
            names=[p["provider_name"] for p in m.get("providers",[])]
            if not any(o in names for o in otts):
                continue

        results.append(m)

    return results

# ============================================
# TOP50
# ============================================

def build_top50(prefs,uvec,max_rt,genres,otts):

    scored=[]
    for m in filter_movies(max_rt,genres,otts):
        if not get_tags(m):
            continue
        s=score(m,prefs,uvec)
        scored.append((s["final"],s,m))

    scored.sort(reverse=True)

    return [(s,m) for _,s,m in scored[:50]]

# ============================================
# PRESENT
# ============================================

def present(pool):

    picks=random.sample(pool,min(5,len(pool)))

    for s,m in picks:

        print(f"\n🎬 {m['title']} ({year_of(m)})")
        print(f" ⏱{m.get('runtime')}분")
        print(" 📺",[p["provider_name"] for p in m.get("providers",[])])

        print("\n  ---- 점수 ----")
        print(f" 태그  : {s['tag']:.1f} / 50")
        print(f" 유사도: {s['sim']:.1f}")
        print(f" 평점  : {s['rating']:.1f} / 10")
        print(f" 최신  : {s['recent']:.1f} / 10")
        print(f" 인기  : {s['pop']:.1f} / 30")
        print("--------------------")
        print(f" FINAL : {s['final']:.1f} / 100")

        if s["tag_detail"]:
            detail=", ".join(
                f"{t}({v:+.1f})"
                for t,v in sorted(
                    s["tag_detail"],
                    key=lambda x:abs(x[1]),
                    reverse=True
                )[:8]
            )
            print(" TAG 근거 →",detail)

    remaining=[x for x in pool if x not in picks]

    return remaining

# ============================================
# MAIN
# ============================================

if __name__=="__main__":

    prefs,uvec = collect_preferences()

    rt=int(input("\n⏱ 최대 러닝타임: "))

    genres=choose_from_map("🎭 장르 선택", GENRE_MAP)
    otts=choose_from_map("📺 OTT 선택", OTT_MAP)

    pool=build_top50(prefs,uvec,rt,genres,otts)

    if not pool:
        print("\n❌ 조건 만족 영화 없음")
        exit()

    print(f"\n✅ 상위 {len(pool)}편 후보 확보\n")

    while pool:

        pool=present(pool)

        if not pool:
            print("\n✅ 모든 추천 완료")
            break

        cmd=input("\n👉 다시 추천 Enter / 종료 q: ").strip().lower()
        if cmd=="q":
            break
