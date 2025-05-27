import connectorx as cx


# session = mysqlx.get_session(
#     {
#         "host": "localhost",
#         "port": 33060,
#         "user": "root",
#         "password": "95gzagedmrjzP0nv3xq!",
#     }
# )

# session.sql("USE sakila").execute()
#uri = "mysql://root:95gzagedmrjzP0nv3xq!@localhost:3306"

# config = {
#     "dbms": "mysql",
#     "port": 3306,
#     "user": "root",
#     "password": "95gzagedmrjzP0nv3xq!",
#     "server": "localhost"
# }

def create_uri(dbms, user, password, server, port, database = None):

    uri = f"{dbms}://{user}:{password}@{server}:{port}"

    return uri if database is None else f"{uri}/{database}"

# uri = create_uri(**config)

df = cx.read_sql(conn = uri, query = "SELECT * FROM sys.sys_config", protocol= "text", return_type = "polars", partition_num=4)
# print(df.head(20))

import yaml
import os

os.chdir("./tsuro/sql/")
with open("db_creds.yaml", "r") as stream:
    yaml_stream = yaml.safe_load(stream)

print(yaml_stream)



