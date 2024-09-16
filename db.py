import sqlite3
import logging

DATABASE_NAME = "housing.db"

def get_db():
    conn = sqlite3.connect(DATABASE_NAME, timeout=5)
    return conn

def create_tables():
    tables = [
        """
        CREATE TABLE IF NOT EXISTS property_listings 
        (
            property_id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            price REAL NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            area_sqft INTEGER NOT NULL,
            description TEXT NOT NULL,
            dt_posted timestamp,
            dt_updated timestamp,
            status TEXT NOT NULL DEFAULT 'available',
            seller_id TEXT NOT NULL,
            seller_contact TEXT NOT NULL,
            seller_email TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tbl_train_data 
        (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            RERA INTEGER,
            num_rooms INTEGER,
            square_ft INTEGER,
            ready_to_move FLOAT,
            longitude FLOAT,
            latitude FLOAT,
            city_code INTEGER,
            avg_sqft_per_room FLOAT,
            posted_by_builder INTEGER, 
            posted_by_dealer INTEGER, 
            posted_by_owner INTEGER,
            target FLOAT
        );

        CREATE TABLE IF NOT EXISTS tbl_prediction 
        (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            RERA INTEGER,
            num_rooms INTEGER,
            square_ft INTEGER,
            ready_to_move FLOAT,
            longitude FLOAT,
            latitude FLOAT,
            city_code INTEGER,
            avg_sqft_per_room FLOAT,
            posted_by_builder INTEGER, 
            posted_by_dealer INTEGER, 
            posted_by_owner INTEGER,
            Price FLOAT,
            dt_pred timestamp,
            pred_type TEXT,
            user_id varchar(100)
        );

        CREATE TABLE IF NOT EXISTS  register_user 
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name varchar(100),
            email varchar(100),
            pass_word varchar(100),
            token varchar(500),
            user_status Text,
            user_level Text,
            dt_time timestamp,
            UNIQUE (id ,email)
        );

        CREATE TABLE IF NOT EXISTS tbl_login_user 
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email varchar(100),
            pass_word varchar(100),
            token varchar(500),
            user_status TEXT,
            user_level TEXT,
            dt_time timestamp,
            UNIQUE (id, email)
        );

        CREATE TABLE IF NOT EXISTS tbl_model_artifacts 
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_train_score FLOAT,
            model_test_score FLOAT,
            MAE FLOAT,
            RMSE FLOAT,
            R2 FLOAT,
            Adjusted_R2 FLOAT, 
            model_timestamp timestamp,
            model_status TEXT,
            model_location varchar(1000)
        );
        """
    ]
    
    db = get_db()
    cursor = db.cursor()
    for table in tables:
        try:
            cursor.executescript(table)
            db.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
        finally:
            cursor.close()
            db.close()

if __name__ == '__main__':
    create_tables()
