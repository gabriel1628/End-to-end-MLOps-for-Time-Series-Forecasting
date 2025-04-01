import pymysql
from dotenv import dotenv_values

env_vars = dotenv_values(".env")

connection = pymysql.connect(
    host="localhost", user=env_vars["DB_USER"], password=env_vars["DB_PASSWORD"]
)

with connection.cursor() as cursor:
    # conn.cursor().execute(f"create database {db_name}")
    cursor.execute("SHOW DATABASES")
    list_databases = cursor.fetchall()

if "study_1" not in list_databases:
    db_name = "study_1"
else:
    i = 1
    while f"study_{i}" in l:
        i += 1
    db_name = f"study_{i}"
print(db_name)

# https://pymysql.readthedocs.io/en/latest/user/examples.html
# https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls
