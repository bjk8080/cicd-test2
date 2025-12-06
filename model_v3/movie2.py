#!/usr/bin/env python3
"""
movie_recommender.py

- ratings.csv 로 평점 데이터 로드
- User-Item Matrix 생성
- Item-Based CF (코사인 유사도)
- ALS (implicit)
- (선택) CBF 점수까지 합치는 Hybrid 추천
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import implicit


# ============================================================
# 1. 데이터 로딩 & 인덱싱
# ============================================================

class RatingsData:
    """
    ratings.csv를 로딩하고,
    userId / movieId를 0~N-1 인덱스로 매핑하는 헬퍼 클래스
    """
    def __init__(self, ratings_csv_path: str):
        if not os.path.exists(ratings_csv_path):
            raise FileNotFoundError(f"ratings.csv 파일을 찾을 수 없습니다: {ratings_csv_path}")

        self.ratings_df = pd.read_csv(ratings_csv_path)

        expected_cols = {"userId", "movieId", "rating"}
        if not expected_cols.issubset(set(self.ratings_df.columns)):
            raise ValueError(f"ratings.csv 컬럼에 {expected_cols} 가 모두 있어야 합니다. 현재 컬럼: {self.ratings_df.columns}")

        # 고유 ID 추출
        self.unique_users = self.ratings_df["userId"].unique()
        self.unique_items = self.ratings_df["movieId"].unique()

        # ID → index 매핑
        self.user2idx = {u: i for i, u in enumerate(self.unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}

        self.item2idx = {m: i for i, m in enumerate(self.unique_items)}
        self.idx2item = {i: m for m, i in self.item2idx.items()}

        # 인덱스로 변환한 df
        self.ratings_df["user_idx"] = self.ratings_df["userId"].map(self.user2idx)
        self.ratings_df["item_idx"] = self.ratings_df["movieId"].map(self.item2idx)

        self.n_users = len(self.unique_users)
        self.n_items = len(self.unique_items)

        print(f"[INFO] Loaded ratings: {len(self.ratings_df)} rows")
        print(f"[INFO] Users: {self.n_users}, Items: {self.n_items}")

    def build_user_item_matrix(self, min_rating: float = 0.0) -> csr_matrix:
        """
        User-Item 평점 행렬 (User x Item) 생성
        """
        df = self.ratings_df[self.ratings_df["rating"] >= min_rating]

        rows = df["user_idx"].values
        cols = df["item_idx"].values
        data = df["rating"].values.astype(np.float32)

        ui_matrix = coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        ui_csr = ui_matrix.tocsr()
        return ui_csr


# ============================================================
# 2. Item-Based CF (코사인 유사도)
# ============================================================

class ItemBasedCF:
    """
    Item-Item Collaborative Filtering (코사인 유사도 기반)
    """
    def __init__(self, user_item_matrix: csr_matrix):
        self.user_item_matrix = user_item_matrix  # (n_users x n_items)
        self.n_users, self.n_items = user_item_matrix.shape
        self.item_sim_matrix: csr_matrix | None = None

    def fit(self, topk: int = 100):
        """
        아이템 간 코사인 유사도 행렬 계산.
        topk: 각 아이템별로 가장 비슷한 아이템 몇 개만 남길지 (메모리 절약용)
        """
        item_vectors = self.user_item_matrix.T  # (n_items x n_users)

        print("[ItemCF] 아이템-아이템 코사인 유사도 계산 중...")
        sim = cosine_similarity(item_vectors, dense_output=False)  # sparse matrix

        if topk is not None:
            sim = self._keep_topk(sim, topk=topk)

        self.item_sim_matrix = sim.tocsr()
        print("[ItemCF] 유사도 행렬 생성 완료")

    @staticmethod
    def _keep_topk(sim_matrix: csr_matrix, topk: int) -> csr_matrix:
        """
        각 아이템별로 상위 topk만 남기고 나머지는 0으로 날리는 함수 (sparsify)
        """
        sim_matrix = sim_matrix.tolil()
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix.data[i]
            indices = sim_matrix.rows[i]

            if len(row) > topk:
                top_idx = np.argsort(row)[-topk:]     # 큰 값 상위 topk
                keep_indices = set(np.array(indices)[top_idx])

                new_row = []
                new_cols = []
                for val, col in zip(row, indices):
                    if col in keep_indices:
                        new_row.append(val)
                        new_cols.append(col)
                sim_matrix.data[i] = new_row
                sim_matrix.rows[i] = new_cols

        return sim_matrix.tocsr()

    def recommend_for_user(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        특정 user_idx에 대해 ItemCF 기반 추천 영화 인덱스 리스트 반환
        return: [(item_idx, score), ...]
        """
        if self.item_sim_matrix is None:
            raise RuntimeError("먼저 fit()을 호출해야 합니다.")

        user_ratings = self.user_item_matrix[user_idx]  # (1 x n_items)
        user_rated_items = user_ratings.indices
        user_rated_scores = user_ratings.data

        scores = np.zeros(self.n_items, dtype=np.float32)

        for rated_item, rating in zip(user_rated_items, user_rated_scores):
            sim_vec = self.item_sim_matrix[rated_item].toarray().ravel()
            scores += sim_vec * rating

        if exclude_seen:
            scores[user_rated_items] = -np.inf

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(i), float(scores[i])) for i in top_indices if np.isfinite(scores[i])]


# ============================================================
# 3. ALS (implicit 라이브러리)
# ============================================================

class ALSRecommender:
    """
    implicit 라이브러리의 ALS 모델 래퍼
    """
    def __init__(self, user_item_matrix: csr_matrix):
        # implicit ALS는 (item x user) 행렬 사용
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = user_item_matrix.T.tocsr()

        self.n_users, self.n_items = user_item_matrix.shape

        self.model = implicit.als.AlternatingLeastSquares(
            factors=64,
            regularization=0.01,
            iterations=20,
            random_state=42
        )

    def fit(self):
        print("[ALS] 모델 학습 시작...")
        # 암묵적 피드백이 전제지만, 평점도 적당히 사용 가능
        self.model.fit(self.item_user_matrix)
        print("[ALS] 모델 학습 완료")

    def recommend_for_user(self, user_idx: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        ALS 기반 추천
        return: [(item_idx, score), ...]
        """
        user_items = self.user_item_matrix[user_idx]

        recommended = self.model.recommend(
            user_idx,
            user_items,
            N=k,
            filter_already_liked=True
        )

        return [(int(i), float(s)) for i, s in recommended]


# ============================================================
# 4. CBF 점수 계산 (벡터가 준비되어 있다는 가정 하에)
# ============================================================

def get_cbf_scores_for_user(
    user_vector: np.ndarray,
    movie_vectors: np.ndarray,
    candidate_item_indices: List[int]
) -> Dict[int, float]:
    """
    user_vector: (d,)
    movie_vectors: (n_items, d)
    candidate_item_indices: CF/ALS가 후보로 뽑은 item_idx 리스트
    """
    user_vec = user_vector.reshape(1, -1)
    cand_matrix = movie_vectors[candidate_item_indices]  # (n_candidates, d)

    sim = cosine_similarity(user_vec, cand_matrix)[0]    # (n_candidates,)
    return {int(item_idx): float(s) for item_idx, s in zip(candidate_item_indices, sim)}


# ============================================================
# 5. Hybrid 스코어링 유틸
# ============================================================

def to_score_dict(score_list: List[Tuple[int, float]]) -> Dict[int, float]:
    """
    [(item_idx, score), ...] → {item_idx: score}
    """
    return {i: s for i, s in score_list}


def hybrid_score(
    cbf_scores: Dict[int, float],
    itemcf_scores: Dict[int, float],
    als_scores: Dict[int, float],
    w_cbf: float = 0.5,
    w_itemcf: float = 0.3,
    w_als: float = 0.2,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    세 가지 모델 점수를 가중합해서 최종 추천 리스트 반환
    cbf_scores, itemcf_scores, als_scores: {item_idx: score}
    """
    final_scores: Dict[int, float] = {}

    all_items = set(cbf_scores.keys()) | set(itemcf_scores.keys()) | set(als_scores.keys())

    for item in all_items:
        s_cbf = cbf_scores.get(item, 0.0)
        s_icf = itemcf_scores.get(item, 0.0)
        s_als = als_scores.get(item, 0.0)

        final_scores[item] = w_cbf * s_cbf + w_itemcf * s_icf + w_als * s_als

    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


# ============================================================
# 6. 메인 실행 예시
# ============================================================

def main():
    # ---- 설정 부분 ----
    ratings_path = "ratings.csv"   # 네 ratings.csv 경로
    use_cbf = False                # CBF(SBERT 벡터) 준비되면 True 로
    user_id_to_recommend = None    # 특정 userId에 대해 추천 받고 싶으면 값 채우기 (예: 1)

    # ---- 데이터 로딩 ----
    data = RatingsData(ratings_path)
    ui_matrix = data.build_user_item_matrix(min_rating=0.0)

    # user_idx 선택 (userId가 주어졌을 때)
    if user_id_to_recommend is None:
        target_user_idx = 0
        target_user_id = data.idx2user[target_user_idx]
    else:
        if user_id_to_recommend not in data.user2idx:
            raise ValueError(f"userId {user_id_to_recommend} 는 ratings.csv에 존재하지 않습니다.")
        target_user_idx = data.user2idx[user_id_to_recommend]
        target_user_id = user_id_to_recommend

    print(f"\n[INFO] Target user: userId={target_user_id}, user_idx={target_user_idx}")

    # ---- ItemCF 학습 ----
    itemcf = ItemBasedCF(ui_matrix)
    itemcf.fit(topk=100)
    itemcf_recs = itemcf.recommend_for_user(target_user_idx, k=30)
    itemcf_dict = to_score_dict(itemcf_recs)

    # ---- ALS 학습 ----
    als = ALSRecommender(ui_matrix)
    als.fit()
    als_recs = als.recommend_for_user(target_user_idx, k=30)
    als_dict = to_score_dict(als_recs)

    # ---- CBF 점수 (벡터 준비되었을 때만) ----
    cbf_dict: Dict[int, float] = {}

    if use_cbf:
        # 예시: movie_vectors.npy, user_vector.npy 를 로드한다고 가정
        # movie_vectors.shape == (n_items, d)
        # user_vector.shape == (d,)
        movie_vec_path = "movie_vectors.npy"
        user_vec_path = "user_vector.npy"

        if not (os.path.exists(movie_vec_path) and os.path.exists(user_vec_path)):
            raise FileNotFoundError("CBF 사용 설정(use_cbf=True)이지만 movie_vectors.npy 또는 user_vector.npy를 찾을 수 없습니다.")

        movie_vectors = np.load(movie_vec_path)  # (n_items, d)
        user_vector = np.load(user_vec_path)     # (d,)

        # 후보 아이템: ItemCF와 ALS가 뽑아준 아이템 합집합
        candidate_items = list(set(itemcf_dict.keys()) | set(als_dict.keys()))

        cbf_dict = get_cbf_scores_for_user(
            user_vector=user_vector,
            movie_vectors=movie_vectors,
            candidate_item_indices=candidate_items
        )

    # ---- 하이브리드 스코어 계산 ----
    if use_cbf:
        w_cbf, w_itemcf, w_als = 0.5, 0.3, 0.2
    else:
        w_cbf, w_itemcf, w_als = 0.0, 0.5, 0.5  # CBF가 없으면 CF + ALS만

    hybrid_recs = hybrid_score(
        cbf_scores=cbf_dict,
        itemcf_scores=itemcf_dict,
        als_scores=als_dict,
        w_cbf=w_cbf,
        w_itemcf=w_itemcf,
        w_als=w_als,
        top_k=10
    )

    # ---- 결과 출력 ----
    print("\n=== Hybrid 추천 결과 (movieId, item_idx, score) ===")
    for item_idx, score in hybrid_recs:
        movie_id = data.idx2item[item_idx]
        print(f"movieId={movie_id}, item_idx={item_idx}, score={score:.4f}")


if __name__ == "__main__":
    main()
