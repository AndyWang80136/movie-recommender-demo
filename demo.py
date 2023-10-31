import os
import random
from collections import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import euclidean_distances, ndcg_score, roc_auc_score
from sqlalchemy import create_engine

MODEL_URL = 'http://localhost:8000/predict/'


def sql_connection():
    return create_engine(
        f'postgresql://{os.environ["DB_USER"]}:{os.environ["DB_PW"]}@{os.environ["DB_DATASET"]}'
    )


class Layout:

    @staticmethod
    def format_sidebar():
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

    @staticmethod
    def format_button():
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


@dataclass
class Item:
    movie_title: str
    item_id: int
    movie_genres: List[str]
    avg_ratings: int
    num_ratings: int
    user_id: int = field(default=None)
    rating: int = field(default=None)
    score: float = field(default=None)
    statistics: dict = field(default_factory=dict)


@dataclass
class User:
    user_id: int
    gender: str
    age_interval: str
    occupation: str
    train_count: int = field(default=None)
    val_count: int = field(default=None)
    test_count: int = field(default=None)


class ItemEngine:
    """Define a item information engine
    """

    def __init__(self, item_df: pd.DataFrame):
        self.item_df = item_df
        self.item_mapping = dict(
            zip(self.item_df['item_id'], self.item_df.index))

    def create(self, item_id: int, **kwargs) -> Item:
        """create item object from item_id

        Args:
            item_id: item id

        Returns:
            Item: Item object
        """
        item_info = self.item_df.iloc[self.item_mapping[item_id]].to_dict()
        return Item(**item_info, **kwargs)


class UserEngine:
    """Define a user information engine
    """

    def __init__(self, user_df: pd.DataFrame):
        self.user_df = user_df
        self.user_mapping = dict(
            zip(self.user_df['user_id'], self.user_df.index))

    def create(self, user_id: int):
        user_info = self.user_df.iloc[self.user_mapping[user_id]].to_dict()
        return User(**user_info)

    @property
    def users(self):
        return self.user_df.user_id.values.tolist()


@dataclass
class Recommendations:
    """stores information of recommendations
    """
    recommendations: List[Item] = field(default_factory=list)
    user_id: int = field(default=None)
    status: str = field(default='not started')
    statistics: dict = field(default_factory=dict)
    _item_ids: dict = field(default_factory=set)

    def add(self, item: Item):
        self.recommendations.append(item)
        self._item_ids.add(item.item_id)

    @property
    def item_ids(self):
        return [item.item_id for item in self.recommendations]

    def __len__(self):
        return len(self.recommendations)

    def __getitem__(self, index: int):
        if index < len(self.recommendations):
            return self.recommendations[index]
        else:
            raise IndexError

    def __iter__(self):
        index = 0
        while index < len(self.recommendations):
            yield self.recommendations[index]
            index += 1


class RecommendationDemo:
    """
    """
    DISPLAY_COLS = 4
    DISPLAY_ROWS = 2

    def __init__(self):
        self.num_recommendations = self.DISPLAY_COLS * self.DISPLAY_ROWS
        self.user_engine = UserEngine(user_df=self.load_user_phase_df())
        self.item_engine = ItemEngine(item_df=self.load_movie_rating_df())
        st.session_state['recommendations'] = Recommendations()

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
                test_count
            FROM user_phase_count
            JOIN users USING (user_id)
            JOIN occupations USING (occupation_id)
            JOIN ages USING (age_id)
            WHERE test_count != 0
            ORDER BY train_count DESC
        """
        df = pd.read_sql(command, engine)
        return df

    @staticmethod
    @st.cache_data
    def load_movie_rating_df():
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

    @staticmethod
    def format_item(item: Item):
        return (f'{item.movie_title}')

    @staticmethod
    def format_user(user: User):
        return (
            f'**Gender:** `{user.gender}` \n\n' +
            f'**Age:** `{user.age_interval}` \n\n' +
            f'**Occupation:** `{user.occupation}` \n\n' +
            f'**Train/Val/Test:** `{user.train_count}`/`{user.val_count}`/`{user.test_count}`'
        )

    def infer_df_by_model(self, test_df: pd.DataFrame) -> dict:
        """infer dataframe by deployed model

        Args:
            test_df: dataframe

        Returns:
            dict: value with status code and prediction
        """
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

    def show_recommendations(self,
                             title: str = 'Recommendations',
                             on_click: Optional[dict] = None,
                             title_divider: bool = False):
        Layout.format_button()
        recommendations = st.session_state['recommendations']
        assert isinstance(recommendations, Iterable)
        # Title
        st.header(
            title,
            divider=title_divider if not title_divider else title_divider)
        # Recommendation button widget
        cols = st.columns([1 for _ in range(self.DISPLAY_COLS)])
        for row in range(self.DISPLAY_ROWS):
            for col in range(self.DISPLAY_COLS):
                index = row * self.DISPLAY_COLS + col
                if index < len(recommendations):
                    item = recommendations[index]
                    cols[col].button(self.format_item(item),
                                     key=random.randint(0, 1e8),
                                     kwargs={'item': item},
                                     on_click=on_click)


class LoginDemo:
    """Define the window when first entering application
    """

    def __init__(self):
        Layout.format_sidebar()

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


class DCNPerformanceDemo(RecommendationDemo):
    """Define model performance check on test phase
    """

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

    def make_recommendations(self, user_id: int):
        """make recommendations by user id, add recommendations in st.session_state['recommendations']

        Args:
            user_id: user id
        """
        if user_id != st.session_state['recommendations'].user_id:
            test_df = self.get_test_df_by_user_id(user_id=user_id)
            preds = self.infer_df_by_model(test_df)
            if preds['prediction'] is not None:
                test_df['score'] = preds['prediction']
                sorted_df = test_df.sort_values(
                    by='score', ascending=False).reset_index(drop=True)
                sorted_df['label'] = (sorted_df['rating'] >= 4).astype(int)
                recommendations = Recommendations(
                    user_id=user_id,
                    status='successed',
                    statistics={
                        'user': {
                            'num_likes': sum(sorted_df['label']),
                            'num_dislikes': sum(sorted_df['label'] == 0)
                        }
                    })
                for _, rec in sorted_df.head(
                        self.num_recommendations).iterrows():
                    item_info = rec.to_dict()
                    item = self.item_engine.create(
                        item_id=item_info['item_id'],
                        score=item_info['score'],
                        rating=item_info['rating'],
                        user_id=item_info['user_id'])
                    recommendations.add(item)

                recommendations.statistics['metrics'] = {
                    f'NDCG@{self.num_recommendations}':
                    round(
                        ndcg_score(
                            y_true=sorted_df['label'].values.reshape(1, -1),
                            y_score=sorted_df['score'].values.reshape(1, -1),
                            k=self.num_recommendations), 2),
                    'AUC':
                    round(
                        roc_auc_score(y_true=sorted_df['label'].values,
                                      y_score=sorted_df['score'].values), 2)
                }
                st.session_state['recommendations'] = recommendations
            else:
                st.session_state['recommendations'] = Recommendations(
                    user_id=user_id, status=preds['code'])

    @staticmethod
    def format_item(item: Item):
        return (f'{item.movie_title} \n\n' +
                f'`Avg Rating: {item.avg_ratings}` \n\n' +
                f'`User Rating: {item.rating}` \n\n' +
                f'`Score: {round(item.score, 2)}`')

    def show_statistics(self):
        num_likes = st.session_state["recommendations"].statistics['user'][
            'num_likes']
        num_dislikes = st.session_state["recommendations"].statistics['user'][
            'num_dislikes']

        st.header('Testing Phase Statistics')
        st.markdown(f'**Like\/DisLike: {num_likes}/{num_dislikes}**')
        st.header('Metrics')
        st.markdown(
            f'**NDCG@{self.num_recommendations}: {st.session_state["recommendations"].statistics["metrics"][f"NDCG@{self.num_recommendations}"]}**\n\n'
            +
            f'**AUC: {st.session_state["recommendations"].statistics["metrics"][f"AUC"]}**'
        )

    def create_window(self):
        Layout.format_sidebar()
        # sidebar view
        with st.sidebar:
            st.header('User ID')
            user_id = st.selectbox('user_id',
                                   self.user_engine.users,
                                   label_visibility='collapsed')
            self.make_recommendations(user_id=user_id)
            user = self.user_engine.create(user_id=user_id)
            st.session_state['current_user'] = user
            st.markdown(self.format_user(user))

        if st.session_state['recommendations'].status == 'successed':
            # sidebar view
            with st.sidebar:
                self.show_statistics()
            # main window
            self.show_recommendations(title='Test Recommendations',
                                      title_divider='violet')
        else:
            st.text_area(label='Error Message',
                         value=st.session_state['recommendations'].status,
                         disabled=True)


class RealLifeDemo(RecommendationDemo):

    def __init__(self):
        super().__init__()
        st.session_state['recommendations'] = None
        st.session_state['click_item'] = None
        st.session_state['history_recommendations'] = None
        self.make_welcome_recommendations()

    @staticmethod
    def load_like_ratio_statistics(user: User, item: Optional[Item] = None):
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
            WHERE (likes_table.attribute_name = 'gender' AND likes_table.attribute_value = '{user.gender}') OR 
                (likes_table.attribute_name = 'age_interval' AND likes_table.attribute_value = '{user.age_interval}') OR
                (likes_table.attribute_name = 'occupation' AND likes_table.attribute_value = '{user.occupation}')
            ORDER BY item_id ASC
        """
        df = pd.read_sql(command, engine)
        if item is not None:
            df = df[df['item_id'] == item.item_id].reset_index(drop=True)
        df = df[['attribute_name', 'like_ratio',
                 'item_id']].pivot(columns='attribute_name',
                                   values='like_ratio',
                                   index='item_id').reset_index(drop=False)
        return df

    def get_recommendations_by_genre(self,
                                     item: Item,
                                     user: User,
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
                WHERE item1_id = {item.item_id}
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
        recommendations = Recommendations(user_id=user.user_id,
                                          status='successed')
        for item_id in cossim_df['item2_id']:
            item = self.item_engine.create(item_id=item_id)
            recommendations.add(item)
        return recommendations

    def get_recommendations_by_like_ratio(self,
                                          item: Item,
                                          user: User,
                                          num_candidates: int = 25) -> list:
        columns = ['gender', 'occupation', 'age_interval']
        df = self.load_like_ratio_statistics(user=user)
        item_like_ratio = np.nan_to_num(
            df[df['item_id'] == item.item_id][columns].values, nan=-1.)
        all_like_ratio = np.nan_to_num(df[columns].values, nan=-1.)

        df['distance'] = np.round(
            euclidean_distances(item_like_ratio, all_like_ratio), 2).ravel()
        df = df[df.item_id != item.item_id]
        sorted_df = df.sort_values(by='distance')
        sorted_df['rank'] = sorted_df['distance'].rank(method='min')
        candidate_df = sorted_df[sorted_df['rank'] <= sorted_df['rank'].iloc[
            num_candidates - 1]]
        recommendations = Recommendations(user_id=user.user_id,
                                          status='successed')
        for item_id, distance in zip(candidate_df['item_id'],
                                     candidate_df['distance']):
            item = self.item_engine.create(item_id=item_id,
                                           statistics={'distance': distance})
            recommendations.add(item)
        recommendations.recommendations = sorted(
            recommendations,
            key=lambda r:
            (-r.statistics['distance'], r.num_ratings, r.avg_ratings),
            reverse=True)[:num_candidates]
        return recommendations

    def get_recommendations_by_model(self,
                                     df: pd.DataFrame,
                                     user: User,
                                     item: Item,
                                     num_candidates: int = 25):
        df = df.copy()
        preds = self.infer_df_by_model(df)

        if preds['prediction'] is not None:
            df['score'] = preds['prediction']
            sort_df = df.sort_values(
                by='score',
                ascending=False).reset_index(drop=True).head(num_candidates)

            recommendations = Recommendations(
                user_id=user.user_id,
                status='successed',
            )
            for _, row in sort_df.iterrows():
                item = self.item_engine.create(item_id=row['item_id'],
                                               score=row['score'])
                recommendations.add(item)

            # check same user has history recommendations or not
            prev_recommendations = st.session_state[
                'recommendations'] if st.session_state[
                    'recommendations'].user_id == user.user_id else None

            if prev_recommendations is None:
                return recommendations

            # merge prev recommendations and make scores decay 0.99
            for item in prev_recommendations:
                # previous algorithm (genre or like ratio) doesn't contain score in item
                if item.score is None:
                    continue
                if item.item_id not in recommendations.item_ids:
                    copy_item = deepcopy(item)
                    copy_item.score *= 0.99
                    recommendations.add(copy_item)

            recommendations.recommendations = sorted(
                recommendations, key=lambda a: a.score,
                reverse=True)[:num_candidates]

        else:
            recommendations = Recommendations(
                user_id=st.session_state['current_user'].user_id,
                status='failed')
        return recommendations

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

    def recommendations_by_algorithm(self, algorithm: str) -> Recommendations:
        if algorithm == 'Content-Based Algorithm':
            # Get recommendations by movie genre similarity
            recommendations = self.get_recommendations_by_genre(
                item=st.session_state['click_item'],
                user=st.session_state['current_user'],
                num_candidates=self.num_recommendations)
        elif algorithm == 'User-Behavior Algorithm':
            # Get recommendations by like ratio distance
            recommendations = self.get_recommendations_by_like_ratio(
                item=st.session_state['click_item'],
                user=st.session_state['current_user'],
                num_candidates=self.num_recommendations)
        else:
            # Get 50 candidates from recommendations by movie genre and like ratio
            # Use DCN model to rank items
            candidates_id = self.get_recommendations_by_genre(
                item=st.session_state['click_item'],
                user=st.session_state['current_user'],
                num_candidates=25
            ).item_ids + self.get_recommendations_by_like_ratio(
                item=st.session_state['click_item'],
                user=st.session_state['current_user'],
                num_candidates=25).item_ids
            test_df = self.get_test_df_by_user_id(
                user_id=st.session_state['current_user'].user_id,
                item_ids=candidates_id)
            recommendations = self.get_recommendations_by_model(
                df=test_df,
                item=st.session_state['click_item'],
                user=st.session_state['current_user'],
                num_candidates=self.num_recommendations)

        return recommendations

    def make_welcome_recommendations(self):
        sorted_df = self.item_engine.item_df.sort_values(
            by=['num_ratings', 'avg_ratings'], ascending=[False, False])
        recommendations = Recommendations(user_id=-1)
        for _, rec in sorted_df.head(self.num_recommendations).iterrows():
            item_info = rec.to_dict()
            item = self.item_engine.create(item_id=item_info['item_id'])
            recommendations.add(item)
        recommendations.status = 'successed'
        st.session_state['recommendations'] = recommendations

    def algorithm_button_selected(self, algorithm):
        st.session_state['algorithm'] = algorithm

    def item_button_click(self, item: Item):
        st.session_state['click_item'] = item
        st.session_state[
            'recommendations'] = self.recommendations_by_algorithm(
                algorithm=st.session_state['algorithm'])

    def user_selected(self, user_id: int):
        # change user selection
        if user_id != st.session_state['recommendations'].user_id:
            user = self.user_engine.create(user_id=user_id)
            st.session_state['current_user'] = user

        if st.session_state['click_item'] is not None:
            like_ratio_df = self.load_like_ratio_statistics(
                user=st.session_state['current_user'],
                item=st.session_state['click_item'])

            st.session_state['click_item'].statistics['like_ratio'] = dict(
                zip(['gender', 'age_interval', 'occupation'],
                    like_ratio_df[['gender', 'age_interval',
                                   'occupation']].values[0, :]))

    def show_click_item_info(self):
        # Define item if user didn't click item
        item = st.session_state["click_item"] if st.session_state[
            "click_item"] is not None else Item(item_id=0,
                                                movie_title='',
                                                movie_genres='',
                                                avg_ratings=None,
                                                num_ratings=None)
        item_info = [
            item.item_id, item.num_ratings, item.avg_ratings,
            item.statistics['like_ratio']['gender'],
            item.statistics['like_ratio']['age_interval'],
            item.statistics['like_ratio']['occupation']
        ] if st.session_state["click_item"] is not None else [''] * 6

        # Design Layout
        st.header(f'Movie Information: {item.movie_title}', divider='violet')
        st.markdown(
            f'**Movie Genre:** {" ".join([f"`{i}`" for i in sorted(item.movie_genres.split("|")) if i])}'
        )
        # Movie Information
        with st.container():
            cols = st.columns(3)
            for (col, title, value) in zip([*cols, *cols], [
                    'ID', '#Ratings', 'AVG Rating', 'Gender Like Ratio',
                    'Age Like Ratio', 'Occupation Like Ratio'
            ], item_info):
                col.metric(title, value)

    @staticmethod
    def format_item(item: Item):
        text = (f'{item.movie_title} \n\n' + ' '.join(
            [f'`{genre}`' for genre in sorted(item.movie_genres.split('|'))]))
        if item.score is not None:
            text += ('\n\n' + f'`Score: {round(item.score, 2)}`')
        return text

    def create_window(self):
        # sidebar
        Layout.format_sidebar()
        with st.sidebar:
            st.header('Algorithm Information')
            algorithm_selected = st.radio('Recommender', [
                'Content-Based Algorithm', 'User-Behavior Algorithm',
                'Deep Cross Network'
            ])
            self.algorithm_button_selected(algorithm_selected)

            st.header('User ID')
            user_id = st.selectbox('user_id',
                                   self.user_engine.users,
                                   label_visibility='collapsed')
            self.user_selected(user_id=user_id)
            st.markdown(
                self.format_user(user=st.session_state['current_user']))
        self.show_click_item_info()
        self.show_recommendations(title='Guess User May Like',
                                  on_click=self.item_button_click)


if __name__ == '__main__':
    if 'app' not in st.session_state:
        app = LoginDemo()
    else:
        app = st.session_state['app']

    app.create_window()
