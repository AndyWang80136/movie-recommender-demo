import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Template
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

__all__ = ['PSQLLoader']

DB_USER: str = os.environ.get("DB_USER")
DB_PASSWORD: str = os.environ.get('DB_PW')
DB_DATABASE: str = os.environ.get('DB_DATABASE')


@dataclass
class PSQLLoader:
    user: str = DB_USER
    password: str = DB_PASSWORD
    database: str = DB_DATABASE

    def __post_init__(self):
        self.connection = create_engine(
            f'postgresql://{self.user}:{self.password}@{self.database}')

    def is_valid_connection(self) -> bool:
        """check connection

        Returns:
            bool: connection valid or not
        """
        try:
            self.connection.connect()
            return True
        except OperationalError:
            return False

    def load(self,
             sql_file: Path,
             query_params: Optional[dict] = None) -> pd.DataFrame:
        """execute sql file and return dataframe

        Args:
            sql_file: sql file
            query_params: parameters in sql file

        Returns:
            pd.DataFrame: output dataframe
        """
        if query_params is None:
            query_params = {}
        query_template = self.load_template(sql_file=sql_file).render(
            **query_params)
        return pd.read_sql(query_template, self.connection)

    @staticmethod
    def load_template(sql_file: Path) -> Template:
        """read sql as Template

        Args:
            sql_file: sql file

        Returns:
            Template: jinja Template
        """
        with open(sql_file, 'r') as fp:
            sql_query = Template(fp.read())
        return sql_query
