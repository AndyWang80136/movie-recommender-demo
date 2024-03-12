import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from movie_recommender.utils.sql import PSQLLoader


class TestPSQLLoader:

    @pytest.fixture(scope='class')
    def sql_file(self):
        return 'sql/rating_df.sql'

    @pytest.fixture
    def test_sql_file(self, tmp_path):
        with open(tmp_path / 'test.sql', 'w') as fp:
            fp.write('SELECT * FROM users')
        return tmp_path / 'test.sql'

    def test_init(self, psql_loader):
        assert isinstance(psql_loader.connection, Engine)
        assert psql_loader.is_valid_connection

    def test_load(self, psql_loader, sql_file):
        df = psql_loader.load(sql_file=sql_file,
                              query_params={'phase': 'test'})
        assert isinstance(df, pd.DataFrame)

    def test_load_template(self, test_sql_file):
        template = PSQLLoader.load_template(test_sql_file)
        assert template.render() == 'SELECT * FROM users'
