import random
from collections.abc import Iterable
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

from movie_recommender.data import (Item, ItemEngine, MovieData,
                                    Recommendations, User, UserEngine)
from movie_recommender.utils import PSQLLoader

RECALL_MODEL = 'http://localhost:8000'
RANK_MODEL = 'http://localhost:8001'
RERANK_MODEL = 'http://localhost:8002'
RECALL_METHODS = [
    'ItemCF-Ratings', 'UserCF-Ratings', 'UserCF-Content', 'ItemCF-Genres',
    'Item-Genres'
]
RECALL_METHOD_INFO = {
    'ItemCF-Ratings': f'{RECALL_MODEL}/recall/itemcf-ratings/users',
    'UserCF-Ratings': f'{RECALL_MODEL}/recall/usercf-ratings/users',
    'UserCF-Content': f'{RECALL_MODEL}/recall/usercf-content/users',
    'ItemCF-Genres': f'{RECALL_MODEL}/recall/itemcf-genres/users',
    'Item-Genres': f'{RECALL_MODEL}/recall/item-genres/items'
}
RANK_METHODS = ['DCN']
RANK_METHOD_INFO = {'DCN': f'{RANK_MODEL}/rank/dcn'}
RERANK_METHODS = ['UCB-MMR']
RERANK_INFO = {'UCB-MMR': f'{RERANK_MODEL}/rerank/ucb-mmr'}

STAGES = ['recall', 'rank', 'rerank']
MODELS = {
    'recall': RECALL_METHODS,
    'rank': RANK_METHODS,
    'rerank': RERANK_METHODS
}
BEST_RECALL_PARAMS = {
    'UserCF-Ratings': {
        'similarity_top_k': 100,
        'output_top_k': 100
    },
    'UserCF-Content': {
        'similarity_top_k': 100,
        'output_top_k': 100
    },
    'ItemCF-Ratings': {
        'similarity_top_k': 10,
        'output_top_k': 100
    },
    'ItemCF-Genres': {
        'similarity_top_k': 10,
        'output_top_k': 100
    },
    'Item-Genres': {
        'similarity_top_k': 50,
        'output_top_k': 100
    }
}


class Layout:

    @staticmethod
    def format_sidebar():
        st.markdown(
            """
                <style>
                [data-testid="stSidebar"][aria-expanded="true"]{
                    min-width: 320px;
                    max-width: 320px;
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
                    height: 200px;
                    width: 180px;
                }
                </style>
            """,
            unsafe_allow_html=True,
        )


class RecommendationDemo:
    DISPLAY_COLS = 4
    DISPLAY_ROWS = 2
    NUM_SIMILARITY_TOP_K: int = 50
    NUM_EACH_RECALL_RECOMMENDATIONS: int = 100
    NUM_RANK_RECOMMENDATIONS: int = 100
    NUM_DISPLAY_RECOMMNEDATIONS: int = DISPLAY_COLS * DISPLAY_ROWS

    def __init__(self):
        self.psql_loader = PSQLLoader()
        self.user_engine = UserEngine(df=self.dataset.users)
        self.item_engine = ItemEngine(df=self.dataset.items)
        st.session_state['recommendations'] = st.session_state.get(
            'recommendations', Recommendations())

    @property
    @st.cache_data
    def dataset(_self):
        dataset = MovieData(users=_self.psql_loader.load('sql/user_df.sql'),
                            items=_self.psql_loader.load('sql/item_df.sql'),
                            ratings=_self.psql_loader.load(
                                'sql/rating_df.sql',
                                query_params={'phase': 'train'}))

        return dataset

    @staticmethod
    def format_item(item: Item):
        return (f'{item.movie_title}')

    @staticmethod
    def format_user(user: User):
        return (
            f'**Gender:** `{user.gender}` \n\n' +
            f'**Age:** `{user.age_interval}` \n\n' +
            f'**Occupation:** `{user.occupation}` \n\n' +
            f'**Train/Val/Test:** `{user.statistics["train_count"]}`/`{user.statistics["val_count"]}`/`{user.statistics["test_count"]}` \n\n'
            +
            f'**Interaction TImes**: `{user.statistics["demo_interaction_times"]}`'
        )

    def format_algorithm(self):
        return

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


class ML1MDemo(RecommendationDemo):
    """Define real life demo on ML1M dataset by applying recall, rank, and rerank
    """

    def __init__(self):
        super().__init__()
        st.session_state['clicked_item'] = st.session_state.get(
            'clicked_item', None)
        st.session_state['clicked_items'] = st.session_state.get(
            'clicked_items', None)
        st.session_state['prev_state'] = st.session_state.get(
            'prev_state', Recommendations())
        st.session_state['prev_recommendations'] = st.session_state.get(
            'prev_recommendations', Recommendations())

    def algorithm_selected(self):
        user = st.session_state['current_user']
        if user.statistics['train_count'] != 0:
            st.session_state['recall_algos'] = RECALL_METHODS
        else:
            st.session_state['recall_algos'] = [
                'UserCF-Content', 'Item-Genres'
            ]

        st.session_state['rank_algos'] = RANK_METHODS
        st.session_state['rerank_algos'] = RERANK_METHODS

    def item_button_click(self, item: Item):
        st.session_state['clicked_item'] = item
        st.session_state['clicked_items'].add(item)

    def user_selected(self, user_id: int):
        # change user selection
        if user_id != st.session_state['recommendations'].user_id:
            user = self.user_engine.search(id=user_id)
            data_count_df = self.psql_loader.load(
                'sql/user_stats.sql', query_params={'user_id': user_id})
            user.statistics.update(data_count_df.iloc[0].to_dict())
            user.statistics.update(demo_interaction_times=0)
            st.session_state['current_user'] = user
            st.session_state['recommendations'] = Recommendations(
                user_id=user_id)
            st.session_state['clicked_items'] = Recommendations(
                user_id=user_id)
            st.session_state['clicked_item'] = None
            st.session_state['prev_state'] = Recommendations(user_id=user_id)
            st.session_state['prev_recommendations'] = Recommendations(
                user_id=user_id)
        else:
            # update demo interaction items
            st.session_state['current_user'].statistics[
                'demo_interaction_times'] += 1

    def show_user_info(self):
        # display user information
        st.markdown(self.format_user(user=st.session_state['current_user']))

    def show_algorithm_info(self):
        # display algorithm information
        st.header('Recall Information')
        st.markdown('\n'.join([
            f"`{idx}:{i}`"
            for idx, i in enumerate(st.session_state["recall_algos"])
        ]))
        st.header('Rank Information')
        st.markdown('\n'.join(
            [f"`{i}`" for i in st.session_state["rank_algos"]]))
        st.header('Rerank Information')
        st.markdown('\n'.join(
            [f"`{i}`" for i in st.session_state["rerank_algos"]]))

    def show_clicked_item_info(self):
        # Define item if user didn't click item
        item = st.session_state["clicked_item"] if st.session_state[
            "clicked_item"] is not None else Item(
                item_id=-1, movie_title='', movie_genres='')
        item_stats = self.psql_loader.load(sql_file='sql/item_stats.sql',
                                           query_params={
                                               'item_id': item.item_id
                                           }).iloc[0].to_dict()
        item_info = [
            item.item_id,
            int(item_stats['num_ratings']), item_stats['avg_ratings']
        ] if st.session_state["clicked_item"] is not None else [''] * 3

        # Design Layout
        st.header(f'Movie Information: {item.movie_title}', divider='violet')
        st.markdown(
            f'**Movie Genre:** {" ".join([f"`{i}`" for i in sorted(item.movie_genres) if i])}'
        )
        # Movie Information
        with st.container():
            cols = st.columns(3)
            for (col, title, value) in zip([*cols, *cols],
                                           ['ID', '#Ratings', 'AVG Rating'],
                                           item_info):
                col.metric(title, value)

    @staticmethod
    def format_item(item: Item):
        # format item information
        text = (f'{item.movie_title} \n\n' + '`' +
                ' | '.join([f'{genre}'
                            for genre in sorted(item.movie_genres)]) + '`')

        if item.algorithms:
            text += (
                '\n\n' + f':blue[Recall: '
                f'{", ".join([str(st.session_state["recall_algos"].index(i)) for i in item.algorithms["recall"]["name"].split("/")])}]'
            )
            text += (
                '\n\n' +
                f':violet[Rank Score: {round(item.get("algorithms/rank/score"), 2)}]'
            )
        if item.statistics:
            text += ('\n\n' +
                     f':red[Impressions: {item.statistics["impressions"]}]')

        return text

    def get_recall_api_request(self):
        query = {}
        user_id = st.session_state['current_user'].user_id
        clicked_item_ids = st.session_state['clicked_items'].item_ids

        for name in st.session_state['recall_algos']:
            if name == 'Item-Genres':
                if not clicked_item_ids:
                    continue
                params = {
                    'item_ids': clicked_item_ids,
                    **BEST_RECALL_PARAMS[name]
                }
                url = f'{RECALL_METHOD_INFO[name]}/'
            else:
                params = BEST_RECALL_PARAMS[name]
                url = f'{RECALL_METHOD_INFO[name]}/{user_id}'
            query[name] = dict(url=url, data=None, params=params)

        return query

    def get_rank_api_request(self):
        user_id = st.session_state['current_user'].user_id
        item_ids = st.session_state['recall_items'].item_ids
        test_df = self.psql_loader.load(sql_file='sql/infer_df.sql',
                                        query_params=dict(
                                            user_id=user_id,
                                            item_ids=tuple(item_ids)))
        query = {}
        for name in st.session_state['rank_algos']:
            url = f'{RANK_METHOD_INFO[name]}/'
            query[name] = dict(url=url,
                               json={'df': test_df.to_dict()},
                               params=None)
        return query

    def get_rerank_api_request(self):

        query = {}

        rank_df = st.session_state['rank_items'].to_df(
            columns=['item_id', 'algorithms/rank/score', 'movie_genres'])
        clicked_df = st.session_state['clicked_items'].to_df(
            columns=['item_id', 'algorithms/rank/score', 'movie_genres'])
        impression_df = st.session_state['recommendations'].to_df(
            columns=['item_id', 'algorithms/rank/score', 'movie_genres'])

        if not clicked_df.empty:
            clicked_df = clicked_df.iloc[[-1]].reset_index(drop=True)

        df_dict = {
            'rank_df': rank_df.to_json(orient='split'),
            'clicked_df': clicked_df.to_json(orient='split'),
            'impression_df': impression_df.to_json(orient='split')
        }
        for name in st.session_state['rerank_algos']:
            params = dict(top_k=self.NUM_DISPLAY_RECOMMNEDATIONS,
                          last_n=1,
                          reset=impression_df.empty)
            query[name] = dict(url=f'{RERANK_INFO[name]}/',
                               params=params,
                               json=df_dict)
        return query

    def apply_recall_models(self):
        return {
            name: requests.get(**query)
            for name, query in self.get_recall_api_request().items()
        }

    def apply_rank_models(self):
        return {
            name: requests.post(**query)
            for name, query in self.get_rank_api_request().items()
        }

    def apply_rerank_models(self):
        return {
            name: requests.post(**query)
            for name, query in self.get_rerank_api_request().items()
        }

    def postprocess_recall_results(self, df_dict: Dict[str, pd.DataFrame]):
        clicked_item_ids = st.session_state['clicked_items'].item_ids
        pred_df = pd.concat(df_dict.values(), ignore_index=True)
        pred_df = pred_df[~pred_df.item_id.isin(clicked_item_ids)]
        sort_df = pred_df.groupby('item_id').agg(
            score=pd.NamedAgg(column='score', aggfunc=sum),
            algorithm=pd.NamedAgg(column='algorithm',
                                  aggfunc=lambda a: '/'.join(a))).sort_values(
                                      by='score',
                                      ascending=False).reset_index(drop=False)
        return sort_df

    def postprocess_rank_results(self, df_dict: Dict[str, pd.DataFrame]):
        pred_df = pd.concat(df_dict.values(), ignore_index=True)
        sort_df = pred_df.sort_values(by='score',
                                      ascending=False).reset_index(drop=True)
        return sort_df

    def postprocess_rerank_results(self, df_dict: Dict[str, pd.DataFrame]):
        pred_df = pd.concat(df_dict.values(), ignore_index=True)
        return pred_df

    def apply_stage_model(self, stage: str, key: Optional[str] = None):
        assert stage in ['recall', 'rank', 'rerank']
        key = stage if key is None else key
        pred_results = getattr(self, f'apply_{stage}_models')()
        df_results = {}
        for algo, response in pred_results.items():
            try:
                result = response.json()
                pred_df = pd.read_json(result, orient='split')
                pred_df['algorithm'] = algo
                df_results[algo] = pred_df
            except:
                df_results[algo] = response.status_code

        processed_df = getattr(self,
                               f'postprocess_{stage}_results')(df_results)

        user_id = st.session_state['current_user'].user_id
        prev_state_recommendations = st.session_state['prev_state']
        recommendations = Recommendations(
            user_id=user_id,
            status='successed',
            algorithms=prev_state_recommendations.algorithms)
        recommendations.algorithms.append(stage)

        for _, item_info in processed_df.iterrows():
            item_dict = item_info.to_dict()
            item_id = item_dict.pop('item_id')
            try:
                item = prev_state_recommendations.get(item_id=item_id)
            except:
                # In recall stage, create from item_engine
                item = self.item_engine.search(id=item_id)
            item.algorithms[stage] = dict(name=item_dict['algorithm'],
                                          score=item_dict['score'])
            recommendations.add(item)

        st.session_state[key] = recommendations
        st.session_state['prev_state'] = recommendations

    def postprocess_items(self, items: Recommendations):
        for item in items:
            try:
                prev_item = st.session_state['prev_recommendations'].get(
                    item_id=item.item_id)
                item.statistics[
                    'impressions'] = prev_item.statistics['impressions'] + 1
                prev_item.statistics['impressions'] = item.statistics[
                    'impressions']
            except:
                item.statistics['impressions'] = 0
                st.session_state['prev_recommendations'].add(item)
        return items

    def infer_recommendations(self):
        for stage, stage_key in zip(STAGES, [f'{s}_items' for s in STAGES]):
            self.apply_stage_model(stage=stage, key=stage_key)
        st.session_state['recommendations'] = self.postprocess_items(
            items=st.session_state['rerank_items'])

    def create_window(self):
        # sidebar
        Layout.format_sidebar()
        with st.sidebar:
            st.header('User ID')
            user_id = st.selectbox('user_id',
                                   self.user_engine.user_id,
                                   label_visibility='collapsed')
            self.user_selected(user_id=user_id)
            self.algorithm_selected()

            self.show_user_info()
            self.show_algorithm_info()

        self.infer_recommendations()
        self.show_clicked_item_info()
        self.show_recommendations(title='Guess User May Like',
                                  on_click=self.item_button_click)


if __name__ == '__main__':
    app = ML1MDemo()
    app.create_window()
