import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """추천 시스템 설정"""
    def __init__(self):
        # 모델 설정
        self.model_name = 'all-MiniLM-L6-v2'
        self.embedding_dim = 384
        
        # 가중치 설정
        self.weights = {
            'tag': 0.4,
            'overview': 0.3,
            'genre': 0.2,
            'rating': 0.1
        }
        
        # 필터 설정
        self.min_rating = 6.0
        self.min_votes = 10  # 최소 투표 수
        self.diversity_lambda = 0.3
        
        # 런타임 조합 설정
        self.runtime_tolerance = 15  # ±15분 허용
        self.max_movies_per_recommendation = 10  # 최대 조합 영화 수
        
        # 캐시 경로
        self.cache_dir = 'cache'
        self.embedding_cache = os.path.join(self.cache_dir, 'embeddings.npz')
        self.vector_cache = os.path.join(self.cache_dir, 'vectors.pkl')
        self.user_profile_dir = 'user_profiles'


class UserProfile:
    """사용자 프로필 관리"""
    def __init__(self, user_id):
        self.user_id = user_id
        self.liked_movies = []  # 영화 ID 리스트
        self.disliked_movies = []
        self.preferred_genres = []
        self.subscribed_otts = []
        self.interaction_history = []  # {'movie_id': int, 'rating': int, 'timestamp': str}
        
    def add_interaction(self, movie_id, rating):
        """상호작용 기록 - 중복 제거 포함"""
        self.interaction_history.append({
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        })
        
        # 기존 평가 제거 (반대 리스트에서)
        if movie_id in self.liked_movies:
            self.liked_movies.remove(movie_id)
        if movie_id in self.disliked_movies:
            self.disliked_movies.remove(movie_id)
        
        # 새 평가 추가
        if rating >= 4:  # 5점 만점 기준 4점 이상
            self.liked_movies.append(movie_id)
        elif rating <= 2:
            self.disliked_movies.append(movie_id)
    
    def save(self, directory):
        """프로필 저장"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f'{self.user_id}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
        logger.info(f"사용자 프로필 저장: {filepath}")
    
    @classmethod
    def load(cls, user_id, directory):
        """프로필 로드"""
        filepath = os.path.join(directory, f'{user_id}.json')
        if not os.path.exists(filepath):
            logger.info(f"새 사용자 프로필 생성: {user_id}")
            return cls(user_id)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        profile = cls(user_id)
        profile.__dict__.update(data)
        logger.info(f"사용자 프로필 로드: {filepath}")
        return profile


class MovieRecommender:
    """영화 추천 시스템 메인 클래스"""
    
    def __init__(self, data_path, config=None):
        self.config = config or Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"실행 환경: {self.device.upper()}")
        
        # 캐시 디렉토리 생성
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.user_profile_dir, exist_ok=True)
        
        # 모델 로드
        logger.info("SentenceTransformer 모델 로드 중...")
        self.model = SentenceTransformer(self.config.model_name).to(self.device)
        
        # 데이터 로드
        self.df = self._load_data(data_path)
        logger.info(f"총 {len(self.df)}개 영화 로드 완료")
        
        # 벡터 생성 또는 로드
        self._build_or_load_vectors()
        
        # 메타데이터 추출
        self.available_genres = sorted(list(set([g for sublist in self.df['genre_names'] for g in sublist])))
        self.available_providers = sorted(list(set([p for sublist in self.df['provider_names'] for p in sublist if p])))
        
        logger.info("시스템 초기화 완료")
    
    def _load_data(self, path):
        """데이터 로드 및 전처리"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"데이터 파일 없음: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # 결측치 처리
        df['overview'] = df['overview'].fillna("")
        df['runtime'] = df['runtime'].fillna(0).astype(int)
        df['vote_average'] = df['vote_average'].fillna(0.0).astype(float)
        df['popularity'] = df['popularity'].fillna(0.0).astype(float)
        df['vote_count'] = df.get('vote_count', pd.Series([0] * len(df))).fillna(0).astype(int)
        
        # 파싱
        df['genre_names'] = df['genres'].apply(
            lambda x: [g['name'] for g in x] if isinstance(x, list) else []
        )
        df['provider_names'] = df['providers'].apply(
            lambda x: [p['provider_name'] for p in x] if isinstance(x, list) else []
        )
        df['display_tags'] = df['predicted_tags'].apply(
            lambda x: [t['tag'] for t in sorted(x, key=lambda k: k['score'], reverse=True)] 
            if isinstance(x, list) else []
        )
        
        # 품질 필터링
        df = df[
            (df['runtime'] > 0) & 
            (df['runtime'] < 300) &  # 5시간 이상 제외
            (df['overview'].str.len() > 20)  # 너무 짧은 설명 제외
        ].reset_index(drop=True)
        
        return df
    
    def _build_or_load_vectors(self):
        """벡터 생성 또는 캐시에서 로드"""
        
        # 벡터 캐시 확인
        if os.path.exists(self.config.vector_cache):
            logger.info("벡터 캐시 로드 중...")
            with open(self.config.vector_cache, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.genre_matrix = cache_data['genre_matrix']
            self.tag_matrix = cache_data['tag_matrix']
            self.mlb_genre = cache_data['mlb_genre']
            self.tag_list = cache_data['tag_list']
            self.tag_to_idx = cache_data['tag_to_idx']
            logger.info("벡터 캐시 로드 완료")
        else:
            logger.info("벡터 생성 중...")
            self._build_genre_vectors()
            self._build_tag_vectors()
            
            # 벡터 캐시 저장
            cache_data = {
                'genre_matrix': self.genre_matrix,
                'tag_matrix': self.tag_matrix,
                'mlb_genre': self.mlb_genre,
                'tag_list': self.tag_list,
                'tag_to_idx': self.tag_to_idx
            }
            with open(self.config.vector_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"벡터 캐시 저장: {self.config.vector_cache}")
        
        # 임베딩 로드 또는 생성
        if os.path.exists(self.config.embedding_cache):
            logger.info("임베딩 캐시 로드 중...")
            data = np.load(self.config.embedding_cache)
            self.overview_matrix = torch.from_numpy(data['overview']).to(self.device)
            logger.info("임베딩 캐시 로드 완료")
        else:
            logger.info("Overview 임베딩 생성 중 (시간이 걸릴 수 있습니다)...")
            embeddings = self.model.encode(
                self.df['overview'].tolist(),
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True,
                batch_size=32
            )
            self.overview_matrix = embeddings
            
            # 캐시 저장
            np.savez_compressed(
                self.config.embedding_cache,
                overview=embeddings.cpu().numpy()
            )
            logger.info(f"임베딩 캐시 저장: {self.config.embedding_cache}")
    
    def _build_genre_vectors(self):
        """장르 벡터 생성 (Multi-hot encoding)"""
        self.mlb_genre = MultiLabelBinarizer()
        self.genre_matrix = self.mlb_genre.fit_transform(self.df['genre_names']).astype(np.float32)
    
    def _build_tag_vectors(self):
        """태그 벡터 생성 (Weighted & Normalized)"""
        # 모든 태그 수집
        all_tags = set()
        for tags in self.df['predicted_tags']:
            if isinstance(tags, list):
                for t in tags:
                    all_tags.add(t['tag'])
        
        self.tag_list = sorted(list(all_tags))
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tag_list)}
        
        # 가중치 행렬 생성
        self.tag_matrix = np.zeros((len(self.df), len(self.tag_list)), dtype=np.float32)
        for i, tags in enumerate(self.df['predicted_tags']):
            if isinstance(tags, list):
                for t in tags:
                    if t['tag'] in self.tag_to_idx:
                        self.tag_matrix[i, self.tag_to_idx[t['tag']]] = t['score']
        
        # L2 정규화
        self.tag_matrix = normalize(self.tag_matrix, norm='l2', axis=1)
    
    def get_representative_movies(self, target_genres, top_per_genre=3):
        """장르별 대표 영화 추출"""
        candidates = []
        seen_ids = set()
        
        if not target_genres:
            # 장르 선택 안한 경우: 전체 인기 영화
            top = self.df[self.df['vote_average'] >= self.config.min_rating].sort_values(
                by=['popularity', 'vote_average'],
                ascending=[False, False]
            ).head(top_per_genre * 2)
            
            for _, row in top.iterrows():
                row_data = row.copy()
                row_data['represented_genre'] = "인기 영화"
                candidates.append(row_data)
            return pd.DataFrame(candidates)
        
        # 각 장르별 대표 영화 추출
        for genre in target_genres:
            genre_movies = self.df[
                (self.df['genre_names'].apply(lambda x: genre in x)) &
                (self.df['vote_average'] >= self.config.min_rating)
            ].sort_values(
                by=['popularity', 'vote_average'],
                ascending=[False, False]
            )
            
            count = 0
            for _, row in genre_movies.iterrows():
                if row['id'] not in seen_ids and count < top_per_genre:
                    row_data = row.copy()
                    row_data['represented_genre'] = genre
                    candidates.append(row_data)
                    seen_ids.add(row['id'])
                    count += 1
        
        return pd.DataFrame(candidates)
    
    def _apply_filters(self, user_profile, max_runtime=None):
        """필터링된 영화 인덱스 반환"""
        filtered_indices = []
        
        for i, row in self.df.iterrows():
            # 이미 평가한 영화 제외
            if row['id'] in user_profile.liked_movies:
                continue
            if row['id'] in user_profile.disliked_movies:
                continue
            
            # 평점 필터
            if row['vote_average'] < self.config.min_rating:
                continue
            
            # 런타임 필터
            if max_runtime and row['runtime'] > max_runtime:
                continue
            
            # OTT 필터
            if user_profile.subscribed_otts:
                if not any(ott in row['provider_names'] for ott in user_profile.subscribed_otts):
                    continue
            
            filtered_indices.append(i)
        
        return filtered_indices
    
    def _calculate_similarity_scores(self, user_indices, target_indices):
        """유사도 점수 계산"""
        if not user_indices:
            return None
        
        # 사용자 프로필 벡터 생성
        user_genre_vec = np.mean(self.genre_matrix[user_indices], axis=0).reshape(1, -1)
        user_tag_vec = np.mean(self.tag_matrix[user_indices], axis=0).reshape(1, -1)
        user_overview_vec = torch.mean(self.overview_matrix[user_indices], dim=0).unsqueeze(0)
        
        # 타겟 벡터
        tgt_genre = self.genre_matrix[target_indices]
        tgt_tag = self.tag_matrix[target_indices]
        tgt_overview = self.overview_matrix[target_indices]
        
        # 유사도 계산
        sim_genre = cosine_similarity(user_genre_vec, tgt_genre)[0]
        sim_tag = cosine_similarity(user_tag_vec, tgt_tag)[0]
        sim_overview = util.cos_sim(user_overview_vec, tgt_overview)[0].cpu().numpy()
        
        # 평점 정규화 (0~1)
        ratings = self.df.iloc[target_indices]['vote_average'].values / 10.0
        
        return {
            'genre': sim_genre,
            'tag': sim_tag,
            'overview': sim_overview,
            'rating': ratings
        }
    
    def _mmr_selection(self, candidate_indices, final_scores, similarity_matrix, top_k, diversity_lambda):
        """MMR 알고리즘으로 다양성 확보"""
        if len(candidate_indices) == 0:
            return []
        
        selected_indices = []
        candidate_pool = list(range(len(candidate_indices)))
        
        # 첫 번째: 최고 점수 선택
        best_idx = np.argmax(final_scores)
        selected_indices.append(candidate_pool[best_idx])
        candidate_pool.remove(candidate_pool[best_idx])
        
        # 나머지 선택
        for _ in range(min(top_k - 1, len(candidate_pool))):
            if not candidate_pool:
                break
            
            mmr_scores = []
            for candidate_idx in candidate_pool:
                relevance = final_scores[candidate_idx]
                
                # 이미 선택된 영화들과의 최대 유사도
                current_sim_vec = similarity_matrix[candidate_idx, selected_indices]
                redundancy = np.max(current_sim_vec) if len(selected_indices) > 0 else 0
                
                # MMR 점수
                score = (1 - diversity_lambda) * relevance - (diversity_lambda * redundancy)
                mmr_scores.append(score)
            
            best_mmr_idx = np.argmax(mmr_scores)
            selected_indices.append(candidate_pool[best_mmr_idx])
            candidate_pool.pop(best_mmr_idx)
        
        return selected_indices
    
    def _generate_explanation(self, movie_info, user_profile, breakdown=None):
        """추천 이유 생성"""
        reasons = []
        
        # 좋아하는 영화 기반 설명
        if user_profile.liked_movies and breakdown:
            # 가장 기여도가 높은 요소 찾기
            if breakdown:
                top_factor = max(breakdown.items(), key=lambda x: x[1])
                
                if top_factor[0] == 'genre' and top_factor[1] > 0.7:
                    matching_genres = [g for g in movie_info['genres'] if g in user_profile.preferred_genres]
                    if matching_genres:
                        reasons.append(f"선호 장르({', '.join(matching_genres[:2])}) 포함")
                
                if top_factor[0] == 'tag' and top_factor[1] > 0.5:
                    reasons.append(f"비슷한 분위기의 영화 (태그 유사도 {top_factor[1]:.1%})")
                
                if top_factor[0] == 'overview' and top_factor[1] > 0.6:
                    reasons.append(f"좋아한 영화와 유사한 스토리")
        
        # 평점 기반 설명
        if movie_info['vote_average'] >= 8.0:
            reasons.append(f"높은 평점 (★{movie_info['vote_average']}/10)")
        elif movie_info['vote_average'] >= 7.0:
            reasons.append(f"좋은 평가 (★{movie_info['vote_average']}/10)")
        
        # OTT 정보
        if user_profile.subscribed_otts and movie_info['providers']:
            available_otts = [ott for ott in movie_info['providers'] if ott in user_profile.subscribed_otts]
            if available_otts:
                reasons.append(f"{available_otts[0]}에서 시청 가능")
        
        # 콜드 스타트인 경우
        if not user_profile.liked_movies:
            reasons.append("인기 작품")
        
        return " | ".join(reasons) if reasons else "추천 영화"
    
    def recommend_single(self, user_profile, top_k=5, max_runtime=None):
        """단일 영화 추천"""
        # 필터링
        filtered_indices = self._apply_filters(user_profile, max_runtime)
        
        if not filtered_indices:
            logger.warning("필터 조건을 만족하는 영화가 없습니다.")
            return pd.DataFrame()
        
        # 콜드 스타트 처리
        if not user_profile.liked_movies:
            return self._cold_start_recommend(filtered_indices, user_profile, top_k)
        
        # 좋아하는 영화의 인덱스 찾기
        liked_indices = []
        for movie_id in user_profile.liked_movies:
            idx = self.df[self.df['id'] == movie_id].index
            if len(idx) > 0:
                liked_indices.append(idx[0])
        
        if not liked_indices:
            return self._cold_start_recommend(filtered_indices, user_profile, top_k)
        
        # 유사도 계산
        similarity_scores = self._calculate_similarity_scores(liked_indices, filtered_indices)
        
        # 최종 점수 계산
        weights = self.config.weights
        final_scores = (
            weights['tag'] * similarity_scores['tag'] +
            weights['overview'] * similarity_scores['overview'] +
            weights['genre'] * similarity_scores['genre'] +
            weights['rating'] * similarity_scores['rating']
        )
        
        # MMR 적용
        tgt_overview = self.overview_matrix[filtered_indices]
        item_sim_matrix = util.cos_sim(tgt_overview, tgt_overview).cpu().numpy()
        
        selected_local_indices = self._mmr_selection(
            filtered_indices,
            final_scores,
            item_sim_matrix,
            top_k,
            self.config.diversity_lambda
        )
        
        # 결과 포맷팅
        results = []
        for local_idx in selected_local_indices:
            original_idx = filtered_indices[local_idx]
            row = self.df.iloc[original_idx]
            
            movie_info = {
                'movie_id': row['id'],
                'title': row['title'],
                'score': final_scores[local_idx],
                'vote_average': row['vote_average'],
                'providers': row['provider_names'],
                'runtime': row['runtime'],
                'genres': row['genre_names'],
                'tags': row['display_tags'][:5],
                'overview': row['overview'],
                'breakdown': {
                    'tag': similarity_scores['tag'][local_idx],
                    'overview': similarity_scores['overview'][local_idx],
                    'genre': similarity_scores['genre'][local_idx],
                    'rating': similarity_scores['rating'][local_idx]
                }
            }
            
            # 추천 이유 생성
            movie_info['explanation'] = self._generate_explanation(
                movie_info, user_profile, movie_info['breakdown']
            )
            
            results.append(movie_info)
        
        return pd.DataFrame(results)
    
    def _cold_start_recommend(self, filtered_indices, user_profile, top_k):
        """콜드 스타트: 선호 장르 기반 인기 영화 추천"""
        filtered_df = self.df.iloc[filtered_indices].copy()
        
        # 선호 장르 적용
        if user_profile.preferred_genres:
            filtered_df['genre_match'] = filtered_df['genre_names'].apply(
                lambda x: sum(1 for g in user_profile.preferred_genres if g in x)
            )
            filtered_df = filtered_df[filtered_df['genre_match'] > 0]
            filtered_df = filtered_df.sort_values(
                by=['genre_match', 'popularity', 'vote_average'],
                ascending=[False, False, False]
            )
        else:
            filtered_df = filtered_df.sort_values(
                by=['popularity', 'vote_average'],
                ascending=[False, False]
            )
        
        top_movies = filtered_df.head(top_k)
        
        results = []
        for _, row in top_movies.iterrows():
            movie_info = {
                'movie_id': row['id'],
                'title': row['title'],
                'score': 0.0,
                'vote_average': row['vote_average'],
                'providers': row['provider_names'],
                'runtime': row['runtime'],
                'genres': row['genre_names'],
                'tags': row['display_tags'][:5],
                'overview': row['overview'],
                'breakdown': {'tag': 0.0, 'overview': 0.0, 'genre': 0.0, 'rating': 0.0}
            }
            
            # 추천 이유 생성
            movie_info['explanation'] = self._generate_explanation(
                movie_info, user_profile, None
            )
            
            results.append(movie_info)
        
        return pd.DataFrame(results)
    
    def recommend_for_duration(self, user_profile, target_minutes, max_solutions=5):
        """
        특정 시간에 맞는 영화 조합 추천
        
        Args:
            user_profile: 사용자 프로필
            target_minutes: 목표 시간 (분)
            max_solutions: 최대 추천 조합 수
        
        Returns:
            List[Dict]: 영화 조합 리스트
        """
        logger.info(f"목표 시간 {target_minutes}분에 맞는 영화 조합 추천 시작")
        
        # 후보 영화 추출 (단일 영화로 목표 시간 80% 이하인 것만)
        max_single_runtime = int(target_minutes * 0.8)
        filtered_indices = self._apply_filters(user_profile, max_single_runtime)
        
        if not filtered_indices:
            logger.warning("조건을 만족하는 영화가 없습니다.")
            return []
        
        # 후보 영화 점수 계산
        if user_profile.liked_movies:
            liked_indices = []
            for movie_id in user_profile.liked_movies:
                idx = self.df[self.df['id'] == movie_id].index
                if len(idx) > 0:
                    liked_indices.append(idx[0])
            
            if liked_indices:
                similarity_scores = self._calculate_similarity_scores(liked_indices, filtered_indices)
                weights = self.config.weights
                movie_scores = (
                    weights['tag'] * similarity_scores['tag'] +
                    weights['overview'] * similarity_scores['overview'] +
                    weights['genre'] * similarity_scores['genre'] +
                    weights['rating'] * similarity_scores['rating']
                )
            else:
                # 인기도 기반
                movie_scores = self.df.iloc[filtered_indices]['popularity'].values
                movie_scores = movie_scores / (movie_scores.max() + 1e-8)
        else:
            # 콜드 스타트: 인기도 기반
            movie_scores = self.df.iloc[filtered_indices]['popularity'].values
            movie_scores = movie_scores / (movie_scores.max() + 1e-8)
        
        # 영화 정보 준비
        candidate_movies = []
        for i, idx in enumerate(filtered_indices):
            row = self.df.iloc[idx]
            movie_info = {
                'index': idx,
                'movie_id': row['id'],
                'title': row['title'],
                'runtime': row['runtime'],
                'score': movie_scores[i],
                'vote_average': row['vote_average'],
                'providers': row['provider_names'],
                'genres': row['genre_names'],
                'tags': row['display_tags'][:5],
                'overview': row['overview']
            }
            
            # 추천 이유 생성 (breakdown 정보가 있는 경우)
            if user_profile.liked_movies and liked_indices:
                breakdown = {
                    'tag': similarity_scores['tag'][i] if 'tag' in similarity_scores else 0.0,
                    'overview': similarity_scores['overview'][i] if 'overview' in similarity_scores else 0.0,
                    'genre': similarity_scores['genre'][i] if 'genre' in similarity_scores else 0.0,
                    'rating': similarity_scores['rating'][i] if 'rating' in similarity_scores else 0.0
                }
                movie_info['explanation'] = self._generate_explanation(movie_info, user_profile, breakdown)
            else:
                movie_info['explanation'] = self._generate_explanation(movie_info, user_profile, None)
            
            candidate_movies.append(movie_info)
        
        # 런타임 순 정렬 (큰 것부터)
        candidate_movies.sort(key=lambda x: x['runtime'], reverse=True)
        
        # 조합 찾기 (Greedy + Backtracking)
        solutions = self._find_runtime_combinations(
            candidate_movies,
            target_minutes,
            self.config.runtime_tolerance,
            max_solutions
        )
        
        logger.info(f"{len(solutions)}개 조합 발견")
        return solutions
    
    def _find_runtime_combinations(self, movies, target, tolerance, max_solutions):
        """런타임 조합 찾기 (Backtracking)"""
        solutions = []
        
        def backtrack(start_idx, current_combo, current_runtime, current_score):
            # 종료 조건
            if len(solutions) >= max_solutions:
                return
            
            # 목표 범위 내에 있는 경우
            if abs(current_runtime - target) <= tolerance:
                solutions.append({
                    'movies': current_combo.copy(),
                    'total_runtime': current_runtime,
                    'total_score': current_score,
                    'avg_score': current_score / len(current_combo) if current_combo else 0,
                    'deviation': abs(current_runtime - target)
                })
                return
            
            # 목표 초과
            if current_runtime > target + tolerance:
                return
            
            # 영화 수 제한
            if len(current_combo) >= self.config.max_movies_per_recommendation:
                return
            
            # 탐색
            for i in range(start_idx, len(movies)):
                movie = movies[i]
                new_runtime = current_runtime + movie['runtime']
                
                # 가지치기: 너무 크면 스킵
                if new_runtime > target + tolerance:
                    continue
                
                current_combo.append(movie)
                backtrack(i + 1, current_combo, new_runtime, current_score + movie['score'])
                current_combo.pop()
        
        backtrack(0, [], 0, 0.0)
        
        # 점수 순 정렬 (편차가 작고 점수가 높은 순)
        solutions.sort(key=lambda x: (-x['avg_score'], x['deviation']))
        
        return solutions[:max_solutions]


# -------------------------------------------------------------------------
# CLI UI Functions
# -------------------------------------------------------------------------

def clear_screen():
    """화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text):
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def get_multiselect(options, message):
    """다중 선택 입력"""
    print(f"\n{message}")
    for i, opt in enumerate(options, 1):
        print(f"[{i}] {opt}", end="\t")
        if i % 5 == 0:
            print()
    print()
    
    user_input = input("번호 입력 (쉼표로 구분, 엔터=전체): ").strip()
    if not user_input:
        return []
    
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        return [options[i] for i in indices if 0 <= i < len(options)]
    except (ValueError, IndexError):
        print("잘못된 입력입니다.")
        return []


def display_movie_info(movie, index=None):
    """영화 정보 출력"""
    prefix = f"{index}. " if index else ""
    print(f"\n{prefix}[{movie['title']}]")
    
    # 추천 이유 표시
    if 'explanation' in movie and movie['explanation']:
        print(f"   💡 추천 이유: {movie['explanation']}")
    
    print(f"   ★ 평점: {movie['vote_average']}/10")
    print(f"   📺 OTT: {', '.join(movie['providers']) if movie['providers'] else '정보 없음'}")
    print(f"   ⏱️  런타임: {movie['runtime']}분")
    print(f"   🏷️  장르: {', '.join(movie['genres'])}")
    print(f"   #️⃣  태그: {', '.join(movie['tags'])}")
    print(f"   📝 줄거리: {movie['overview'][:120]}...")
    
    if 'score' in movie and movie['score'] > 0:
        print(f"   🎯 추천 점수: {movie['score']:.4f}")
        if 'breakdown' in movie:
            bd = movie['breakdown']
            print(f"      └ Tag:{bd['tag']:.3f} / Story:{bd['overview']:.3f} / Genre:{bd['genre']:.3f} / Rating:{bd['rating']:.3f}")
    print("-" * 70)


def main():
    """메인 실행 함수"""
    print_header("영화 추천 시스템 초기화")
    
    # 데이터 경로 (실제 경로로 수정 필요)
    data_path = 'datas/trend_data_with_ai_tags.json'
    
    if not os.path.exists(data_path):
        print(f"오류: 데이터 파일을 찾을 수 없습니다 - {data_path}")
        print("현재 디렉토리에 'datas/trend_data_with_ai_tags.json' 파일을 위치시켜주세요.")
        return
    
    # 추천 시스템 초기화
    config = Config()
    recommender = MovieRecommender(data_path, config)
    
    # 사용자 ID 입력
    print("\n사용자 ID를 입력하세요 (신규/기존)")
    user_id = input(">> ").strip() or "default_user"
    
    user_profile = UserProfile.load(user_id, config.user_profile_dir)
    
    # 메인 루프
    while True:
        clear_screen()
        print_header(f"영화 추천 시스템 - 사용자: {user_id}")
        
        print("\n[메뉴]")
        print("1. 선호 장르 설정")
        print("2. OTT 구독 정보 설정")
        print("3. 영화 취향 분석 (대표 영화 평가)")
        print("4. 영화 추천 ⭐")
        print("5. 프로필 확인")
        print("6. 시스템 설정")
        print("0. 종료")
        
        choice = input("\n선택 >> ").strip()
        
        if choice == '1':
            # 선호 장르 설정
            print_header("선호 장르 설정")
            selected_genres = get_multiselect(
                recommender.available_genres,
                "선호하는 장르를 선택하세요:"
            )
            user_profile.preferred_genres = selected_genres
            user_profile.save(config.user_profile_dir)
            print(f"\n✓ {len(selected_genres)}개 장르 저장 완료")
            input("\n엔터를 눌러 계속...")
        
        elif choice == '2':
            # OTT 구독 정보 설정
            print_header("OTT 구독 정보 설정")
            selected_otts = get_multiselect(
                recommender.available_providers,
                "구독 중인 OTT를 선택하세요:"
            )
            user_profile.subscribed_otts = selected_otts
            user_profile.save(config.user_profile_dir)
            print(f"\n✓ {len(selected_otts)}개 OTT 저장 완료")
            input("\n엔터를 눌러 계속...")
        
        elif choice == '3':
            # 영화 취향 분석
            print_header("영화 취향 분석")
            
            if not user_profile.preferred_genres:
                print("⚠️  먼저 선호 장르를 설정해주세요.")
                input("\n엔터를 눌러 계속...")
                continue
            
            rep_movies = recommender.get_representative_movies(user_profile.preferred_genres)
            
            print(f"\n{len(rep_movies)}개 대표 영화를 평가해주세요.")
            print("평가 방법: 1=매우 좋음 / 2=좋음 / 3=보통 / 4=별로 / 5=관심없음")
            
            for idx, row in rep_movies.iterrows():
                print("\n" + "-" * 70)
                print(f"[{row['represented_genre']} 장르 대표]")
                display_movie_info({
                    'title': row['title'],
                    'vote_average': row['vote_average'],
                    'providers': row['provider_names'],
                    'runtime': row['runtime'],
                    'genres': row['genre_names'],
                    'tags': row['display_tags'][:5],
                    'overview': row['overview']
                })
                
                while True:
                    rating_input = input("평가 (1-5) >> ").strip()
                    if rating_input in ['1', '2', '3', '4', '5']:
                        rating = int(rating_input)
                        user_profile.add_interaction(row['id'], 6 - rating)  # 5점 만점으로 변환
                        break
            
            user_profile.save(config.user_profile_dir)
            print("\n✓ 취향 분석 완료!")
            input("\n엔터를 눌러 계속...")
        
        elif choice == '4':
            # 시간 기반 자동 추천 (단일 vs 조합)
            print_header("영화 추천 ⭐")
            
            print("\n시청 가능 시간을 입력하세요.")
            print("  예시: 120분 (2시간) / 720분 (12시간 비행)")
            print("  * 180분 이하: 단일 영화 추천")
            print("  * 180분 초과: 여러 영화 조합 추천")
            
            try:
                available_time = int(input("\n시청 가능 시간(분): ").strip())
            except ValueError:
                print("잘못된 입력입니다.")
                input("\n엔터를 눌러 계속...")
                continue
            
            if available_time <= 0:
                print("시간은 0보다 커야 합니다.")
                input("\n엔터를 눌러 계속...")
                continue
            
            # 180분 기준으로 단일/조합 판단
            THRESHOLD = 180
            
            if available_time <= THRESHOLD:
                # 단일 영화 추천
                print(f"\n{available_time}분 이하 영화를 추천합니다...")
                
                try:
                    top_k = int(input("추천 개수 (기본 5): ").strip() or 5)
                except ValueError:
                    top_k = 5
                
                recommendations = recommender.recommend_single(
                    user_profile, 
                    top_k, 
                    available_time
                )
                
                if recommendations.empty:
                    print("\n조건에 맞는 영화가 없습니다.")
                else:
                    print_header(f"추천 영화 ({len(recommendations)}편)")
                    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                        display_movie_info(movie.to_dict(), i)
            
            else:
                # 조합 추천
                print(f"\n{available_time}분에 맞는 영화 조합을 추천합니다...")
                
                try:
                    max_solutions = int(input("추천 조합 수 (기본 5): ").strip() or 5)
                except ValueError:
                    max_solutions = 5
                
                print(f"\n조합을 찾는 중...")
                combinations = recommender.recommend_for_duration(
                    user_profile,
                    available_time,
                    max_solutions
                )
                
                if not combinations:
                    print("\n조건에 맞는 조합을 찾을 수 없습니다.")
                    print("  - OTT 구독 정보를 확인해주세요.")
                    print("  - 시간을 조정해보세요.")
                    print(f"  - 또는 {THRESHOLD}분 이하로 입력하면 단일 영화를 추천합니다.")
                else:
                    print_header(f"추천 조합 ({len(combinations)}개)")
                    
                    for i, combo in enumerate(combinations, 1):
                        print(f"\n{'='*70}")
                        print(f"조합 #{i}")
                        print(f"총 {len(combo['movies'])}편 / 총 {combo['total_runtime']}분 (목표 대비 {combo['deviation']}분 차이)")
                        print(f"평균 점수: {combo['avg_score']:.4f}")
                        print('='*70)
                        
                        for j, movie in enumerate(combo['movies'], 1):
                            display_movie_info(movie, j)
                        
                        if i < len(combinations):
                            cont = input("\n다음 조합 보기? (y/n): ").strip().lower()
                            if cont != 'y':
                                break
            
            input("\n엔터를 눌러 계속...")
        
        elif choice == '5':
            # 프로필 확인
            print_header("사용자 프로필")
            print(f"\n사용자 ID: {user_profile.user_id}")
            print(f"선호 장르: {', '.join(user_profile.preferred_genres) if user_profile.preferred_genres else '미설정'}")
            print(f"구독 OTT: {', '.join(user_profile.subscribed_otts) if user_profile.subscribed_otts else '미설정'}")
            print(f"좋아하는 영화: {len(user_profile.liked_movies)}편")
            print(f"싫어하는 영화: {len(user_profile.disliked_movies)}편")
            print(f"총 상호작용: {len(user_profile.interaction_history)}회")
            
            input("\n엔터를 눌러 계속...")
        
        elif choice == '6':
            # 시스템 설정
            print_header("시스템 설정")
            print(f"\n현재 설정:")
            print(f"  - Tag 가중치: {config.weights['tag']}")
            print(f"  - Overview 가중치: {config.weights['overview']}")
            print(f"  - Genre 가중치: {config.weights['genre']}")
            print(f"  - Rating 가중치: {config.weights['rating']}")
            print(f"  - 다양성 강도: {config.diversity_lambda}")
            print(f"  - 최소 평점: {config.min_rating}")
            print(f"  - 런타임 허용 오차: ±{config.runtime_tolerance}분")
            
            print("\n변경하시겠습니까? (y/n)")
            if input(">> ").strip().lower() == 'y':
                try:
                    config.weights['tag'] = float(input(f"Tag 가중치 (현재 {config.weights['tag']}): ") or config.weights['tag'])
                    config.weights['overview'] = float(input(f"Overview 가중치 (현재 {config.weights['overview']}): ") or config.weights['overview'])
                    config.weights['genre'] = float(input(f"Genre 가중치 (현재 {config.weights['genre']}): ") or config.weights['genre'])
                    config.weights['rating'] = float(input(f"Rating 가중치 (현재 {config.weights['rating']}): ") or config.weights['rating'])
                    config.diversity_lambda = float(input(f"다양성 강도 (현재 {config.diversity_lambda}): ") or config.diversity_lambda)
                    
                    print("\n✓ 설정 변경 완료")
                except ValueError:
                    print("\n✗ 잘못된 입력입니다.")
            
            input("\n엔터를 눌러 계속...")
        
        elif choice == '0':
            # 종료
            print("\n프로그램을 종료합니다.")
            user_profile.save(config.user_profile_dir)
            break
        
        else:
            print("\n잘못된 선택입니다.")
            input("\n엔터를 눌러 계속...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        print(f"\n오류가 발생했습니다: {e}")