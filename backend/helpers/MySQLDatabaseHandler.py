import os
import sqlalchemy as db

class MySQLDatabaseHandler(object):
    
    IS_DOCKER = True if 'DB_NAME' in os.environ else False

    def __init__(self,MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE,MYSQL_HOST = "localhost"):
        
        self.MYSQL_HOST = os.environ['DB_NAME'] if MySQLDatabaseHandler.IS_DOCKER else MYSQL_HOST
        self.MYSQL_USER = "admin" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_USER
        self.MYSQL_USER_PASSWORD = "admin" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_USER_PASSWORD
        self.MYSQL_PORT = 3306 if MySQLDatabaseHandler.IS_DOCKER else MYSQL_PORT
        self.MYSQL_DATABASE = "kardashiandb" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_DATABASE
        self.engine = self.validate_connection()

    def validate_connection(self):
        print(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")
        return db.create_engine(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")

    def lease_connection(self):
        return self.engine.connect()
    
    def query_executor(self,query):
        conn = self.lease_connection()
        if type(query) == list:
            for i in query:
                conn.execute(i)
        else:
            conn.execute(query)
        

    def query_selector(self,query):
        conn = self.lease_connection()
        data = conn.execute(query)
        return data

    def load_file_into_db(self,file_path  = None):
        if MySQLDatabaseHandler.IS_DOCKER:
            return
        if file_path is None:
            file_path = os.path.join(os.environ['ROOT_PATH'],'init.sql')
        sql_file = open(file_path,"r")
        sql_file_data = list(filter(lambda x:x != '',sql_file.read().split(";\n")))
        self.query_executor(sql_file_data)
        sql_file.close()

# MODELS START HERE
class Book(db.Model):
    __tablename__ = "webtoons"
    title = db.Column(db.String(150), nullable=False)
    descript = db.Column(db.String(1900), nullable=False)
    authors = db.Column(db.String(100), nullable=False)
    publisher = db.Column(db.String(50), nullable=False)
    categories = db.Column(db.String(50), nullable=False)
    review_score = db.Column(db.String(15), nullable=False)
    review_count = db.Column(db.String(15), nullable=False)


    def __init__(self, **kwargs):
        """
        Initializes a Book object
        """

        self.webtoon_id = kwargs.get(webtoon_id)
        self.title = kwargs.get(title)
        self.genre = kwargs.get(genre)
        self.thumbnail = kwargs.get(thumbnail)
        self.summary = kwargs.get(summary)
        self.episodes = kwargs.get(episodes)
        self.created_by = kwargs.get(created_by)
        self.view = kwargs.get(view)
        self.subscribe = kwargs.get(subscribe)
        self.grade = kwargs.get(grade)
        self.released_date = kwargs.get(released_date)
        self.url = kwargs.get(url)
        self.cover = kwargs.get(cover)
        self.likes = kwargs.get(likes)
        self.written_by = kwargs.get(written_by)
        self.art_by = kwargs.get(art_by)
        self.adapted_by = kwargs.get(adapted_by)
        self.original_work_by = kwargs.get(original_work_by)
        self.assisted_by = kwargs.get(assisted_by)
