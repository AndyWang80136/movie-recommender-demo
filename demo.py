import os
import random
import threading
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import euclidean_distances, ndcg_score, roc_auc_score
from sqlalchemy import create_engine
from streamlit.runtime.scriptrunner import add_script_run_ctx

MODEL_URL = 'http://localhost:8000/predict/'


def sql_connection():
    return create_engine(
        f'postgresql://{os.environ["DB_USER"]}:{os.environ["DB_PW"]}@{os.environ["DB_DATASET"]}'
    )


class Demo:

    def __init__(self):
        self.user_df = self.load_user_phase_df()
        self.item_df = self.get_movie_rating_df()
        self.item_mapping = dict(
            zip(self.item_df['item_id'], self.item_df.index))
        self.user_mapping = dict(
            zip(self.user_df['user_id'], self.user_df.index))

    @staticmethod
    @st.cache_data
    def load_user_phase_df():
        engine = sql_connection()
        command = f"""
            SELECT 
                users.user_id,
                users.gender,
                ages.age_interval,
                occupations.occupation,
                train_count,
                val_count,
                test_count,
                total_count
            FROM user_phase_count
            JOIN users USING (user_id)
            JOIN occupations USING (occupation_id)
            JOIN ages USING (age_id)
            WHERE test_count != 0
            ORDER BY users.user_id ASC
        """
        df = pd.read_sql(command, engine)
        return df

    @staticmethod
    @st.cache_data
    def get_movie_rating_df():
        engine = sql_connection()
        command = f"""
            SELECT
                items.item_id,
                movie_title,
                movie_genres,
                COALESCE(num_ratings,0) AS num_ratings,
                COALESCE(avg_ratings,0) AS avg_ratings
            FROM items
            LEFT JOIN (
                SELECT 
                    item_id,
                    COUNT(*) AS num_ratings,
                    ROUND(AVG(rating), 2) AS avg_ratings
                FROM ratings
                GROUP BY item_id
            ) item_rating USING (item_id)
        """
        df = pd.read_sql(command, engine)
        return df

    def format_sidebar(self):
        st.markdown(
            """
                <style>
                [data-testid="stSidebar"][aria-expanded="true"]{
                    min-width: 280px;
                    max-width: 280px;
                }
            """,
            unsafe_allow_html=True,
        )

    def format_button(self):
        st.markdown(
            """
                <style>
                div.stButton > button:first-child {
                    height: 160px;
                    width: 180px;
                }
                </style>
            """,
            unsafe_allow_html=True,
        )

    def format_user_info(self, user_id: int):
        current_user_info = self.user_df.iloc[
            self.user_mapping[user_id]].to_dict()
        gender = f'**Gender:** `{current_user_info["gender"]}`'
        age = f'**age:** `{current_user_info["age_interval"]}`'
        occupation = f'**Occupation:** `{current_user_info["occupation"]}`'
        phase = f'**train/val/test:** `{current_user_info["train_count"]}`/`{current_user_info["val_count"]}`/`{current_user_info["test_count"]}`'
        user_display_string = '\n\n'.join([gender, age, occupation, phase])
        return user_display_string

    def infer_df(self, test_df: pd.DataFrame):
        try:
            response = requests.post(MODEL_URL,
                                     json={'data': test_df.to_dict()})
            if response.status_code == 200:
                pred = response.json()
                return {'code': response.status_code, 'prediction': pred}
            else:
                return {'code': response.status_code, 'prediction': None}
        except Exception as e:
            return {'code': e, 'prediction': None}


class LoginDemo(Demo):
    """Define the window when first entering application
    """

    def __init__(self):
        super().format_sidebar()

    def confirm_clicked(self):
        st.session_state['app'] = RealLifeDemo() if st.session_state[
            'function'] == 'Real Life Recommender' else DCNPerformanceDemo()

    def create_window(self):
        with st.sidebar:
            st.header('Function Selections')
            st.radio('function',
                     ['Real Life Recommender', 'Recommender Performance'],
                     key='function')
            st.button('Confirm', key='confirm', on_click=self.confirm_clicked)


class DCNPerformanceDemo(Demo):
    """Define model performance check on test phase
    """
    DISPLAY_COLS = 4
    DISPLAY_ROWS = 2

    def __init__(self):
        super().__init__()
        st.session_state['recommendations'] = {
            'user_id': None,
            'test_statistics': {},
            'status': 'not started',
            'recommendations': [],
            'scores': [],
            'metrics': {},
            'user_rating': []
        }

        self.num_recommendations = self.DISPLAY_COLS * self.DISPLAY_ROWS

    @staticmethod
    def get_test_df_by_user_id(user_id: int):
        engine = sql_connection()
        command = f"""
            WITH user_rating AS(
                SELECT 
                    ratings.*,
                    gender,
                    occupation_id AS occupation,
                    zip_code,
                    age_id as age,
                    movie_title,
                    movie_genres
                FROM ratings
                JOIN users USING (user_id)
                JOIN items USING (item_id)
                WHERE ratings.user_id = {user_id}
            )  
            SELECT
                *,
                rating_timestamp AS timestamp
            FROM rating_phase
            RIGHT JOIN user_rating
            USING (rating_id)
            WHERE phase = 'test'
        """
        df = pd.read_sql(command, engine)
        return df

    def infer_by_user(self, user_id: int):
        if user_id != st.session_state['recommendations']['user_id']:
            st.session_state['recommendations']['user_id'] = user_id
            st.session_state['recommendations']['status'] = 'not started'
            test_df = self.get_test_df_by_user_id(user_id=user_id)
            preds = self.infer_df(test_df)
            if preds['prediction'] is not None:
                test_df['prediction'] = preds['prediction']
                sort_df = test_df.sort_values(
                    by='prediction', ascending=False).reset_index(drop=True)
                ratings = sort_df['rating'].values
                labels = (ratings >= 4).astype(int)
                scores = sort_df['prediction'].values
                st.session_state['recommendations'].update({
                    'status':
                    'successed',
                    'recommendations':
                    sort_df.head(self.num_recommendations)['item_id'].values,
                    'test_statistics': {
                        'num_likes': sum(labels),
                        'num_dislikes': sum(labels == 0)
                    },
                    'scores':
                    scores[:self.num_recommendations],
                    'user_ratings':
                    ratings[:self.num_recommendations],
                    'metrics': {
                        f'NDCG@{self.num_recommendations}':
                        round(
                            ndcg_score(y_true=labels.reshape(1, -1),
                                       y_score=scores.reshape(1, -1),
                                       k=self.num_recommendations), 2),
                        'AUC':
                        round(roc_auc_score(y_true=labels, y_score=scores), 2)
                    }
                })

            else:
                st.session_state['recommendations'] = {
                    'user_id': user_id,
                    'test_statistics': {},
                    'status': preds['code'],
                    'recommendations': [],
                    'scores': [],
                    'metrics': {},
                    'user_rating': []
                }

    def show_recommendations(self):
        items = st.session_state['recommendations']['recommendations']
        scores = st.session_state['recommendations']['scores']
        user_ratings = st.session_state['recommendations']['user_ratings']
        # Title
        st.header('Testing Recommendations', divider='violet')
        # Recommendation button widget
        cols = st.columns([1 for _ in range(self.DISPLAY_COLS)])
        for row in range(self.DISPLAY_ROWS):
            for col in range(self.DISPLAY_COLS):
                index = row * self.DISPLAY_COLS + col
                if index < len(items):
                    item_info = self.item_df.iloc[self.item_mapping[
                        items[index]]].to_dict()
                    cols[col].button(
                        f"{item_info['movie_title']} \n\n " +
                        f"`AVG Rating: {item_info['avg_ratings']}` \n\n " +
                        f"`User Rating: {user_ratings[index]}` \n\n " +
                        f"`Score: {round(scores[index], 2)}`",
                        key=random.randint(0, 1e8))

    def show_test_statistics(self):
        num_likes = st.session_state["recommendations"]['test_statistics'][
            'num_likes']
        num_dislikes = st.session_state["recommendations"]['test_statistics'][
            'num_dislikes']

        st.header('Testing Phase Statistics')
        st.markdown(f'**Like\/DisLike: {num_likes}/{num_dislikes}**')
        st.header('Metrics')
        st.markdown(
            f'**NDCG@{self.num_recommendations}: {st.session_state["recommendations"]["metrics"][f"NDCG@{self.num_recommendations}"]}**\n\n'
            +
            f'**AUC: {st.session_state["recommendations"]["metrics"][f"AUC"]}**'
        )

    def create_window(self):
        self.format_button()
        self.format_sidebar()
        # sidebar view
        with st.sidebar:
            st.header('User ID')
            user_id = st.selectbox('user_id',
                                   self.user_df.user_id,
                                   label_visibility='collapsed')
            # send a thread to get prediction for current user_id
            thread = threading.Thread(target=self.infer_by_user,
                                      kwargs={'user_id': user_id})
            add_script_run_ctx(thread)
            thread.start()

            st.markdown(self.format_user_info(user_id))
            st.session_state['current_user_info'] = self.user_df.iloc[
                self.user_mapping[user_id]].to_dict()

        # wait for inference done
        with st.spinner('Testing Inference ...'):
            while st.session_state['recommendations'][
                    'status'] == 'not started':
                pass

        if st.session_state['recommendations']['status'] == 'successed':
            # sidebar view
            with st.sidebar:
                self.show_test_statistics()
            # main window
            self.show_recommendations()

        else:
            st.text_area(label='Error Message',
                         value=st.session_state['recommendations']['status'],
                         disabled=True)


class RealLifeDemo(Demo):
    DISPLAY_COLS = 4
    DISPLAY_ROWS = 2

    def __init__(self):
        super().__init__()
        st.session_state['history_items'] = []
        st.session_state['history_recommendations'] = {}
        st.session_state['click_item'] = None
        st.session_state['show_statistics'] = [''] * 6
        st.session_state['item_statistics'] = {
            'movie_title': '',
            'movie_genres': ''
        }
        st.session_state['item_like_ratio'] = {}
        st.session_state['item_like_ratio_for_user'] = None
        self.num_recommendations = self.DISPLAY_COLS * self.DISPLAY_ROWS
        self.run_welcome_recommendations()

    @st.cache_data
    def get_movie_rating_genreal_df(_self):
        movie_rating_df = _self.get_movie_rating_df()
        return movie_rating_df.sort_values(by=['num_ratings', 'avg_ratings'],
                                           ascending=[False, False])

    @staticmethod
    def load_item_like_ratio(user_info: dict):
        engine = sql_connection()
        command = f"""
            SELECT 
                likes_table.item_id,
                likes_table.attribute_name,
                likes_table.attribute_value,
                ROUND(num_likes::NUMERIC/NULLIF(num_ratings, 0), 2) AS like_ratio
            FROM num_likes_user_domain likes_table
            JOIN num_ratings_user_domain ratings_table ON 
                ratings_table.attribute_name = likes_table.attribute_name AND
                ratings_table.attribute_value = likes_table.attribute_value AND 
                ratings_table.item_id = likes_table.item_id
            WHERE 	(likes_table.attribute_name = 'gender' AND likes_table.attribute_value = '{user_info["gender"]}') OR 
                (likes_table.attribute_name = 'age_interval' AND likes_table.attribute_value = '{user_info["age_interval"]}') OR
                (likes_table.attribute_name = 'occupation' AND likes_table.attribute_value = '{user_info['occupation']}')
            ORDER BY item_id ASC
        """
        df = pd.read_sql(command, engine)
        return df

    @staticmethod
    def get_candidate_pool_by_genre(item_id: int,
                                    num_candidates: int = 25) -> list:
        engine = sql_connection()
        command = f"""
            WITH cte AS (
                SELECT 
                    item1_id,
                    item2_id,
                    similarity,
                    RANK() OVER (PARTITION BY item1_id ORDER BY similarity DESC) AS sim_rank
                FROM item_genre_cossim
                WHERE item1_id = {item_id}
            ), user_item_ratings AS (
                SELECT
                    items.item_id,
                    movie_title,
                    movie_genres,
                    COALESCE(num_ratings,0) AS num_ratings,
                    COALESCE(avg_ratings,0) AS avg_ratings
                FROM items
                LEFT JOIN (
                    SELECT 
                        item_id,
                        COUNT(*) AS num_ratings,
                        ROUND(AVG(rating), 2) AS avg_ratings
                    FROM ratings
                    GROUP BY item_id
                ) item_rating ON item_rating.item_id = items.item_id
            )
            SELECT 
                item1_id,
                item2_id, 
                similarity,
                sim_rank,
                num_ratings,
                avg_ratings
            FROM cte
            JOIN user_item_ratings ON user_item_ratings.item_id = item2_id
            ORDER BY similarity DESC, num_ratings DESC, avg_ratings DESC
            LIMIT {num_candidates}
        """
        cossim_df = pd.read_sql(command, engine)
        return cossim_df['item2_id'].values.tolist()

    def get_candidate_pool_by_user_interact(self,
                                            item_id: int,
                                            user_info: dict,
                                            num_candidates: int = 25) -> list:
        like_ratio_df = st.session_state['item_like_ratio'][
            user_info['user_id']]
        gender_df = like_ratio_df[
            like_ratio_df['attribute_name'] == 'gender'].reset_index(
                drop=True).rename(columns={'like_ratio': f'gender_like_ratio'})
        age_df = like_ratio_df[like_ratio_df['attribute_name'] ==
                               'age_interval'].reset_index(drop=True).rename(
                                   columns={
                                       'like_ratio': f'age_interval_like_ratio'
                                   }).drop(columns=['item_id'])
        occupation_df = like_ratio_df[
            like_ratio_df['attribute_name'] == 'occupation'].reset_index(
                drop=True).rename(columns={
                    'like_ratio': f'occupation_like_ratio'
                }).drop(columns=['item_id'])
        all_item_df = pd.concat([gender_df, age_df, occupation_df], axis=1)
        item_index = all_item_df[all_item_df['item_id'] ==
                                 item_id].index.values
        user_item_df = all_item_df[[
            'gender_like_ratio', 'age_interval_like_ratio',
            'occupation_like_ratio'
        ]].fillna(-1).values
        distance = np.round(
            euclidean_distances(user_item_df[item_index], user_item_df),
            2).ravel()

        sorted_index = np.argsort(distance)

        sorted_index = sorted_index[sorted_index != item_index[0]]
        df = all_item_df.iloc[sorted_index].reset_index(drop=True)
        df['similarity'] = distance[sorted_index]
        merge_df = df.merge(self.item_df, on='item_id').sort_values(
            by=['similarity', 'num_ratings', 'avg_ratings'])
        sorted_item_ids = merge_df['item_id'].values.tolist()[:num_candidates]
        return sorted_item_ids

    @staticmethod
    def get_test_df_by_user_id(user_id: int, item_ids: List[int]):
        engine = sql_connection()
        command = f"""
            WITH cte AS (
                SELECT 
                    items.*,
                    {user_id} AS user_id
                FROM items
                WHERE items.item_id IN {tuple(item_ids)}
            )

            SELECT 
                *,
                occupation_id AS occupation,
                age_id AS age,
                (SELECT MAX(rating_timestamp) FROM ratings WHERE user_id = {user_id}) AS timestamp
            FROM cte
            JOIN users USING (user_id)
        """
        df = pd.read_sql(command, engine)
        return df

    def recommendations_by_algorithm(self, algorithm: str):
        if algorithm == 'Content-Based Algorithm':
            recommendation_items = self.get_candidate_pool_by_genre(
                item_id=st.session_state['click_item'],
                num_candidates=self.num_recommendations)
            recommendations = {
                'recommendations': recommendation_items,
                'status': 'successed',
                'scores': None
            }
        elif algorithm == 'User-Behavior Algorithm':
            recommendation_items = self.get_candidate_pool_by_user_interact(
                item_id=st.session_state['click_item'],
                user_info=st.session_state['current_user_info'],
                num_candidates=self.num_recommendations)
            recommendations = {
                'recommendations': recommendation_items,
                'status': 'successed',
                'scores': None
            }
        else:
            user_id = st.session_state['current_user_info']['user_id']
            candidates = self.get_candidate_pool_by_genre(
                item_id=st.session_state['click_item'],
                num_candidates=25) + self.get_candidate_pool_by_user_interact(
                    item_id=st.session_state['click_item'],
                    user_info=st.session_state['current_user_info'],
                    num_candidates=25)
            test_df = self.get_test_df_by_user_id(user_id=user_id,
                                                  item_ids=candidates)
            preds = self.infer_df(test_df)
            if preds['prediction'] is not None:
                test_df['prediction'] = preds['prediction']
                sort_df = test_df.sort_values(
                    by='prediction', ascending=False).reset_index(drop=True)
                infer_times = len(st.session_state['history_recommendations']
                                  ['recommendations'])
                if infer_times > 0:
                    prev_candidates = st.session_state[
                        'history_recommendations']['recommendations'][
                            infer_times - 1]
                    prev_candidates = prev_candidates[
                        ~prev_candidates.item_id.isin(sort_df.item_id)].copy()
                    prev_candidates[
                        'prediction'] = prev_candidates['prediction'] * 0.9
                    sort_df = pd.concat([sort_df, prev_candidates],
                                        axis=0).sort_values(by='prediction',
                                                            ascending=False)
                st.session_state['history_recommendations']['recommendations'][
                    infer_times] = sort_df.head(self.num_recommendations)

                candidates = sort_df['item_id'].values[:self.
                                                       num_recommendations]
                scores = sort_df['prediction'].values[:self.
                                                      num_recommendations]
                recommendations = {
                    'recommendations': candidates,
                    'status': 'successed',
                    'scores': scores
                }
            else:
                recommendations = {
                    'recommendations': None,
                    'status': 'failed',
                    'scores': None
                }
        return recommendations

    def run_welcome_recommendations(self):
        movie_rating_df = self.get_movie_rating_genreal_df()
        st.session_state['recommendations'] = {
            'recommendations':
            movie_rating_df.head(
                self.num_recommendations)['item_id'].values.tolist(),
            'status':
            'successed',
            'scores':
            None
        }

    def algorithm_button_selected(self, algorithm):
        st.session_state['algorithm'] = algorithm

    def item_button_click(self, item_id: int):
        st.session_state['click_item'] = item_id
        st.session_state['history_items'].append(item_id)
        st.session_state[
            'recommendations'] = self.recommendations_by_algorithm(
                algorithm=st.session_state['algorithm'])

    def user_selected(self, user_id: int):
        # change user selection
        if user_id != st.session_state['history_recommendations'].get(
                'user_id'):
            st.session_state['history_items'] = []
            st.session_state['current_user_info'] = self.user_df.iloc[
                self.user_mapping[user_id]].to_dict()
            st.session_state['item_like_ratio'] = {
                user_id:
                self.load_item_like_ratio(
                    user_info=st.session_state['current_user_info'])
            }
            st.session_state['history_recommendations'] = {
                'user_id': user_id,
                'recommendations': {}
            }
        if st.session_state['click_item'] is not None:
            df = st.session_state['item_like_ratio'][user_id]
            item_df = df[df.item_id == st.session_state['click_item']]
            item_info = {
                **self.item_df.iloc[self.item_mapping[st.session_state['click_item']]],
                **dict(zip(item_df['attribute_name'], item_df['like_ratio']))
            }
            st.session_state['item_statistics'] = item_info
            st.session_state['show_statistics'] = [
                item_info['item_id'], item_info['num_ratings'],
                item_info['avg_ratings'], item_info['gender'],
                item_info['age_interval'], item_info['occupation']
            ]

    def show_recommendations(self):
        # main window
        st.header(
            f'Movie Information: {st.session_state["item_statistics"]["movie_title"]}',
            divider='violet')
        st.markdown(
            f'**Movie Genre:** {" ".join([f"`{i}`" for i in sorted(st.session_state["item_statistics"]["movie_genres"].split("|")) if i])}'
        )
        # Movie Information
        with st.container():
            cols = st.columns(3)
            for (col, title, value) in zip([*cols, *cols], [
                    'ID', '#Ratings', 'AVG Rating', 'Gender Like Ratio',
                    'Age Like Ratio', 'Occupation Like Ratio'
            ], st.session_state['show_statistics']):
                col.metric(title, value)

        # Recommendation List
        st.header('Guess User may like')
        recommendations = st.session_state['recommendations']
        recommendation_items, status, scores = recommendations[
            'recommendations'], recommendations['status'], recommendations[
                'scores']
        if status == 'successed':
            cols = st.columns([1 for _ in range(self.DISPLAY_COLS)])
            for row in range(self.DISPLAY_ROWS):
                for col in range(self.DISPLAY_COLS):
                    index = row * self.DISPLAY_COLS + col
                    item_info = self.item_df.iloc[self.item_mapping[
                        recommendation_items[index]]].to_dict()
                    title = item_info['movie_title']
                    genres = ' '.join([
                        f'`{genre}`' for genre in sorted(
                            item_info['movie_genres'].split('|'))
                    ])
                    score = f'`Score: {scores[index]:.2f}`' if scores is not None else ''
                    text = '\n\n'.join([title, genres, score])

                    cols[col].button(text,
                                     key=random.randint(0, 1e8),
                                     on_click=self.item_button_click,
                                     kwargs={'item_id': item_info['item_id']})
        else:
            st.text_area(label='Error Message', value=status, disabled=True)

    def create_window(self):
        self.format_button()
        self.format_sidebar()
        # sidebar
        with st.sidebar:
            st.header('Algorithm Information')
            algorithm_selected = st.radio('Recommender', [
                'Content-Based Algorithm', 'User-Behavior Algorithm',
                'Deep Cross Network'
            ])
            self.algorithm_button_selected(algorithm_selected)

            st.header('User ID')
            user_id = st.selectbox('user_id',
                                   self.user_df.user_id,
                                   label_visibility='collapsed')
            self.user_selected(user_id=user_id)
            st.markdown(self.format_user_info(user_id=user_id))
        self.show_recommendations()


if __name__ == '__main__':
    if 'app' not in st.session_state:
        app = LoginDemo()
    else:
        app = st.session_state['app']

    app.create_window()