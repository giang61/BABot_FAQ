import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

db = SQLDatabase.from_uri('postgresql://postgres:P0seidon@localhost:5432/dvdrental')

print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM film LIMIT 10;")