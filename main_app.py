import seaborn as sns
import os
import uuid
import ast
import os.path
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from flask import * 
from flask import Flask, request, flash, render_template, session, redirect, request, send_from_directory, send_file
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta,datetime
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask_mail import Mail, Message
from passlib.context import CryptContext
import re
from db import get_db, create_tables
import sqlite3
import shutil
from scipy import stats
from geopy import Nominatim
from send_email import send_notification

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=5)

UPLOAD_FOLDER = 'Dataset'
UPLOAD_MODEL='heart_model'
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_MODEL'] = UPLOAD_MODEL
MAIL_USERNAME = "abhyasttechnosolution@gmail.com"
os.makedirs(os.path.abspath(UPLOAD_FOLDER), exist_ok=True)

password_manager = CryptContext(
    schemes=["pbkdf2_sha256"],
    default="pbkdf2_sha256",
    pbkdf2_sha256__default_rounds=30000
)

mail = Mail(app)

DATA_FILE = 'data.json'

create_tables()

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('city_encoder.pkl', 'rb') as f:
    city_encoder = joblib.load(f)
city_list = city_encoder.classes_.tolist()

label_encoder = LabelEncoder()

# result = {
#             'rera': "select",
#             'num_of_rooms': 0,
#             'square_ft': 0,
#             'ready_to_move': 0,
#             'neighbourhood_region': "neighbourhood",
#             'city': "select",
#             'posted_by': "select",
#             'price': 0
#         }

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load the machine learning model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# model = joblib.load('random_forest_model.pkl')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Geolocator object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
geolocator = Nominatim(user_agent="my_geocoder")


@app.route('/forgot', methods=["POST", "GET"])
def forgot():
    if request.method == "POST":
        email = request.form["email"]
        token = str(uuid.uuid4())
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "select email,pass_word,token from register_user where email=?", (email,))
        items = cursor.fetchall()

        if items:
            data = cursor.fetchone()
            send_notification(email, MAIL_USERNAME, token)
            
            statement = "Update register_user set token=? where email=?"
            cursor.execute(statement, [token, email])
            db.commit()
            cursor.close()

            msg = "A token was sent to your email. Kindly check your Email"
            return render_template('reset.html', message = msg)
        else:

            msg = "This email is not registered with us"
            return render_template('forgot.html', message = msg)

    return render_template('forgot.html')


@app.route('/reset', methods=["POST", "GET"])
def reset():
    if request.method == "POST":
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        token = request.form["token"]

        if password != confirm_password:
            flash("Password do not match", 'danger')
            return redirect('/reset')
        passencr = password_manager.hash(password)
        token1 = str(uuid.uuid4())
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "select email,pass_word,token from register_user where token=?", (token,))
        items = cursor.fetchall()

        if items:

            cursor.execute(
                'UPDATE register_user set token=?, pass_word=? WHERE token = ?', (token1, passencr, token))
            db.commit()
            cursor.close()

            msg = "Your password successfully updated."
            return render_template('login.html',message = msg)

        else:
            msg = "Your token is invalid"
            return render_template('reset.html', message = msg)

    return render_template('reset.html')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Admin Part modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functionalities for admin
@app.route("/admin")
def admin_dashboard():
    return render_template("admin_dashboard.html")

# View Users Route
@app.route("/admin/view_users")
def view_users():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, email, user_status, user_level, dt_time FROM register_user")
    users = cursor.fetchall()   
    return render_template("view_users.html", users=users)

# following features to be updated depending on the accessibility of the users: approval_status, user_type
@app.route("/admin/user_approval", methods=["GET", "POST"])
def user_approval():
    if request.method == "POST":
        # Print the entire form data
        print("Form Data:", request.form)
        
        # Get the list of selected users for approval
        selected_users = request.form.getlist("selected_users")
        approval = request.form["approval"]
        user_type = request.form["user_type"]

        # Print selected user IDs to ensure they are being received correctly
        print("Selected User IDs:", selected_users)
        print("Approval Status:", approval)
        print("User Type:", user_type)

        # Update each selected user's details
        db = get_db()
        cursor = db.cursor()
        for user_id in selected_users:
            update_query = "UPDATE register_user SET user_status = ?, user_level = ? WHERE id = ?"
            cursor.execute(update_query, (approval, user_type, user_id))
        db.commit()

        # Fetch updated user list
        cursor.execute("SELECT id, email, user_status, user_level, dt_time FROM register_user")
        users = cursor.fetchall()

        # Provide feedback to the user
        message = "Users approved successfully"
        return render_template("view_users.html", message=message, users=users)

    # If it's a GET request or after processing the POST request, redirect to view_users page
    return redirect("/admin/view_users")

# Download Users Data Route
@app.route("/admin/download_users_data")
def download_users_data():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM register_user")
    users = cursor.fetchall()

    # Convert user data to DataFrame
    df = pd.DataFrame(users)
    
    # Save DataFrame as CSV file
    df.to_csv("users_data.csv", index=False)
    
    # Return the CSV file as a downloadable attachment
    return send_file("users_data.csv", as_attachment=True)

# View Logged-in Users Data Route
@app.route("/admin/view_logged_in_users")
def view_logged_in_users():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM user_logins")
    logged_in_users = cursor.fetchall()
    return render_template("view_logged_in_users.html", logged_in_users=logged_in_users)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Activating a model for making predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route('/activate', methods=["GET", "POST"])
def activate():
    if 'user_id' in session:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "SELECT id, model_name, model_train_score, model_test_score, MAE, RMSE, R2, Adjusted_R2, model_timestamp, model_status FROM tbl_model_artifacts")
        items = cursor.fetchall()
        
        payload = []
        for item in items:
            payload.append({
                'id': item[0], 
                'model_name': item[1], 
                "model_train_score": item[2], 
                "model_test_score": item[3], 
                "MAE": item[4], 
                "RMSE": item[5], 
                "R2": item[6], 
                "Adjusted_R2": item[7], 
                "model_timestamp": item[8], 
                "model_status": item[9]
            })
        
        message = ""
        if request.method == 'POST':
            ids_checked = request.form.getlist('check')
            if ids_checked:
                if len(ids_checked) > 1:
                    message = "Kindly select only 1 model at a time for activation."
                else:
                    id = int(ids_checked[0])
                    cursor.execute(
                        "UPDATE tbl_model_artifacts SET model_status='Active' WHERE id=?", (id,))
                    cursor.execute(
                        "UPDATE tbl_model_artifacts SET model_status='Inactive' WHERE id<>?", (id,))
                    db.commit()
                    message = "Selected model activated successfully."
            else:
                message = "No Model selected for Activation !!"
        
        return render_template('model_history.html', payload=payload, message=message)
    else:
        return redirect('/')

    
@app.route('/upload_dataset')
def upload_dataset():
    if 'user_id' in session:
        return render_template('upload_data.html', level=session['user_role'])
    else:
        return redirect('/')

@app.route('/upload_data', methods=["GET", "POST"])
def upload_data():
    if 'user_id' in session:
        if request.method == 'POST':
            f = request.files['file']
            fname = request.files['file'].filename
            if fname == "":
                flash("Please upload a file.")
                return redirect('/upload_dataset')

            extension = fname.split('.')[-1]
            if extension not in ["csv", "xlsx"]:
                flash("Please upload a CSV or Excel file.")
                return redirect('/upload_dataset')

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(file_path)

            if extension == "csv":
                orig_data = pd.read_csv(file_path)
            else:
                orig_data = pd.read_excel(file_path)

            # Data cleaning and preprocessing steps
            orig_data.columns = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'SQUARE_FT', 'READY_TO_MOVE', 
                                 'RESALE', 'ADDRESS', 'LONGITUDE', 'LATITUDE', 'TARGET(PRICE_IN_LACS)']

            orig_data = orig_data.drop_duplicates()
            orig_data.rename(columns={"BHK_NO.": "NUM_ROOMS", "TARGET(PRICE_IN_LACS)": "PRICE_IN_LACS"}, inplace=True)
            orig_data = orig_data[((orig_data["LATITUDE"] > 60) & (orig_data["LATITUDE"] < 90)) & 
                                  ((orig_data["LONGITUDE"] > 5) & (orig_data["LONGITUDE"] < 36))]

            orig_data["CITY"] = orig_data["ADDRESS"].str.split(',').str[-1]
            city_count = orig_data["CITY"].value_counts()
            cities_to_keep = city_count[city_count >= 10].index
            orig_data = orig_data[orig_data["CITY"].isin(cities_to_keep)]
            orig_data = orig_data[orig_data['SQUARE_FT'] > 200]

            one_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 1) & (orig_data['SQUARE_FT'] > 200)]
            two_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 2) & (orig_data['SQUARE_FT'] > 400)]
            three_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 3) & (orig_data['SQUARE_FT'] > 800)]
            four_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 4) & (orig_data['SQUARE_FT'] > 1200)]
            five_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 5) & (orig_data['SQUARE_FT'] > 1600)]
            six_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 6) & (orig_data['SQUARE_FT'] > 2000)]
            seven_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 7) & (orig_data['SQUARE_FT'] > 2500)]
            eight_bhk_data = orig_data[(orig_data["NUM_ROOMS"] == 8) & (orig_data['SQUARE_FT'] > 3000)]

            new_data = pd.concat([one_bhk_data, two_bhk_data, three_bhk_data, four_bhk_data, five_bhk_data, six_bhk_data, 
                                  seven_bhk_data, eight_bhk_data], axis=0)
            new_data.reset_index(drop=True, inplace=True)

            city_count = new_data["CITY"].value_counts()
            cities_to_keep = city_count[city_count >= 10].index
            new_data = new_data[new_data["CITY"].isin(cities_to_keep)]
            new_data["NEIGHBOURHOOD"] = new_data["ADDRESS"].str.split(',').str[-2]

            orig_data["NEIGHBOURHOOD"] = orig_data["ADDRESS"].str.split(',').str[-2]
            orig_data = orig_data[orig_data["CITY"] != "Maharashtra"]

            neighbourhood = {}
            for index, row in orig_data.iterrows():
                city = row["CITY"]
                neighbour = row["NEIGHBOURHOOD"]
                if city in neighbourhood:
                    neighbourhood[city].append(neighbour)
                else:
                    neighbourhood[city] = [neighbour]

            state_rows = new_data[new_data['CITY'].str.lower() == "maharashtra"]
            corrected_cities = []
            for index, row in state_rows.iterrows():
                neighbour_hood = row['NEIGHBOURHOOD'].lower()
                valid_cities = []
                for city, neighbours in neighbourhood.items():
                    if neighbour_hood in [neighbour.lower() for neighbour in neighbours]:
                        valid_cities.append(city)
                if valid_cities:
                    corrected_cities.append((index, valid_cities[0]))

            for index, city in corrected_cities:
                new_data.at[index, 'CITY'] = city

            new_data = new_data[new_data["CITY"] != "Maharashtra"]
            new_data = new_data[new_data["CITY"] != "Goa"]

            new_data = new_data[new_data["PRICE_IN_LACS"] < 15000]
            new_data = new_data[new_data["SQUARE_FT"] < 100000]

            new_data[['SQUARE_FT', 'PRICE_IN_LACS']] = np.log(new_data[['SQUARE_FT', 'PRICE_IN_LACS']])
            new_data["AVG_SQFT_PER_ROOM"] = new_data["SQUARE_FT"] / new_data["NUM_ROOMS"]

            cleaned_data = new_data[new_data["BHK_OR_RK"] != "RK"]
            cleaned_data = cleaned_data.drop(columns=["ADDRESS", "BHK_OR_RK", "NEIGHBOURHOOD", "UNDER_CONSTRUCTION", "RESALE"], axis=1)

            city_encode = LabelEncoder()
            cleaned_data["CITY"] = city_encode.fit_transform(cleaned_data["CITY"])

            with open('city_encoder.pkl', 'wb') as f:
                joblib.dump(city_encode, f)

            cleaned_data = pd.get_dummies(data=cleaned_data, columns=["POSTED_BY"], dtype="int64")

            # cleaned_data.columns = ['RERA', 'NUM_ROOMS', 'SQUARE_FT', 'READY_TO_MOVE', 'LONGITUDE', 'LATITUDE', 'CITY', 
            #                         'AVG_SQFT_PER_ROOM', 'POSTED_BY_Builder', 'POSTED_BY_Dealer', 'POSTED_BY_Owner', 
            #                         'PRICE_IN_LACS']

            conn = sqlite3.connect('housing.db')

            df_check = pd.read_sql_query("SELECT * FROM tbl_train_data", conn)
            if len(df_check) > 0:
                db = get_db()
                cursor = db.cursor()
                cursor.execute("DELETE FROM tbl_train_data")
                db.commit()

            db = get_db()
            cursor = db.cursor()
            for index, row in cleaned_data.iterrows():
                statement = "INSERT INTO tbl_train_data(RERA, num_rooms, square_ft, ready_to_move, longitude, latitude, city_code, avg_sqft_per_room, posted_by_builder, posted_by_dealer, posted_by_owner, target) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                cursor.execute(statement, [row['RERA'], row['NUM_ROOMS'], row['SQUARE_FT'], row['READY_TO_MOVE'], row['LONGITUDE'], 
                                           row['LATITUDE'], row['CITY'], row['AVG_SQFT_PER_ROOM'], row['POSTED_BY_Builder'], 
                                           row['POSTED_BY_Dealer'], row['POSTED_BY_Owner'], row['PRICE_IN_LACS']])
            db.commit()
            conn.close()

            message = ("File uploaded and data processed successfully.")
            return render_template('train_data.html', tables=[cleaned_data.to_html(classes='data')], titles=cleaned_data.columns.values, message = message)
        return redirect('/upload_dataset')
    else:
        return redirect('/')

# train the machine learning models
@app.route("/train_models", methods=["GET", "POST"])
def train_model():
    if request.method == 'POST':
        target = request.form['target']
        test_dist = float(request.form['test_dist'])

        db = get_db()
        cursor = db.cursor()

        statement = "SELECT RERA, num_rooms, square_ft, ready_to_move, longitude, latitude, city_code, avg_sqft_per_room, posted_by_builder, posted_by_dealer, posted_by_owner, target FROM tbl_train_data"
        conn = sqlite3.connect('housing.db')

        df = pd.read_sql_query(statement, conn)

        x = df.drop(columns=["target"], axis=1)
        y = df["target"]

        min_max_scaler = MinMaxScaler()
        x = min_max_scaler.fit_transform(x)

        with open('scaler.pkl', 'wb') as f:
            joblib.dump(min_max_scaler, f)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_dist, random_state=1)
        model_results = model_comparison(x_train, y_train, x_test, y_test)

        model_results = model_results.reset_index(drop=True)
        
        db = get_db()
        cursor = db.cursor()
        """
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT, model_train_score REAL, model_test_score REAL, MAE REAL, RMSE REAL, R2 REAL, Adjusted_R2 REAL, model_timestamp TEXT, model_status TEXT, model_location TEXT
        """
        for index, row in model_results.iterrows():
            statement = "INSERT INTO tbl_model_artifacts(model_name, model_train_score, model_test_score, MAE, RMSE, R2, Adjusted_R2, model_timestamp, model_status, model_location) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            cursor.execute(statement, [row['model_name'], row['model_train_score'], row['model_test_score'], row['MAE'], row['RMSE'], row['R2'], row['Adjusted_R2'], datetime.now(), 'Inactive', row['Location']])
        db.commit()

        model_results = model_results.drop(['Location'], axis=1)

        return render_template('train_result.html', tables=[model_results.to_html(classes='result')], titles=model_results.columns.values)
    return redirect('/')

def model_comparison(x_train, y_train, x_test, y_test):
    tmstp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S_%f')
    path_loc = os.path.abspath(os.path.join('house_price_model', tmstp))
    os.makedirs(path_loc, exist_ok=True)
    date_time = datetime.now()
    models = []

    # Linear Regression
    linear = LinearRegression()
    linear.fit(x_train, y_train)

    y_pred_linear = linear.predict(x_test)
    model_name = 'LinearRegression'
    model_path = generate_and_upload_pickle(linear, model_name, path_loc)
    train_score_lr, test_score_lr, mae_lr, rmse_lr, r2_lr, adj_r2_lr = evaluate_model(linear, x_train, y_train, x_test, y_test, y_pred_linear)
    models.append((model_name, train_score_lr, test_score_lr, mae_lr, rmse_lr, r2_lr, adj_r2_lr, model_path))

    # Decision tree regressor
    decision_tree = DecisionTreeRegressor(min_samples_split=2)
    decision_tree.fit(x_train, y_train)

    y_pred_decision_tree = decision_tree.predict(x_test)
    model_name = 'DecisionTreeRegressor'
    model_path = generate_and_upload_pickle(decision_tree, model_name, path_loc)
    train_score_dt, test_score_dt, mae_dt, rmse_dt, r2_dt, adj_r2_dt = evaluate_model(decision_tree, x_train, y_train, x_test, y_test, y_pred_decision_tree)
    models.append((model_name, train_score_dt, test_score_dt, mae_dt, rmse_dt, r2_dt, adj_r2_dt, model_path))

    # Random Forest regressor
    random_forest_reg = RandomForestRegressor()
    random_forest_reg.fit(x_train, y_train)

    y_pred_random_forest = random_forest_reg.predict(x_test)
    model_name = 'RandomForestRegressor'
    model_path = generate_and_upload_pickle(random_forest_reg, model_name, path_loc)
    train_score_rf, test_score_rf, mae_rf, rmse_rf, r2_rf, adj_r2_rf = evaluate_model(random_forest_reg, x_train, y_train, x_test, y_test, y_pred_random_forest)
    models.append((model_name, train_score_rf, test_score_rf, mae_rf, rmse_rf, r2_rf, adj_r2_rf, model_path))

    # Gradient Boosting Regressor
    gboost_regressor = GradientBoostingRegressor(n_estimators=400, max_depth=10, min_samples_split=2, learning_rate=0.01, loss="absolute_error")
    gboost_regressor.fit(x_train, y_train)

    y_pred_gboost = gboost_regressor.predict(x_test)
    model_name = 'GradientBoostingRegressor'
    model_path = generate_and_upload_pickle(gboost_regressor, model_name, path_loc)
    train_score_gb, test_score_gb, mae_gb, rmse_gb, r2_gb, adj_r2_gb = evaluate_model(gboost_regressor, x_train, y_train, x_test, y_test, y_pred_gboost)
    models.append((model_name, train_score_gb, test_score_gb, mae_gb, rmse_gb, r2_gb, adj_r2_gb, model_path))

    model_results = pd.DataFrame(models, columns=['model_name', 'model_train_score', 'model_test_score', 'MAE', 'RMSE', 'R2', 'Adjusted_R2', 'Location'])
    return model_results

def generate_and_upload_pickle(model: object, modelname: str, path_loc: str) -> str:
    model_path_loc = os.path.join(path_loc, modelname + '.pkl')
    joblib.dump(model, model_path_loc)
    return model_path_loc

def evaluate_model(model, x_train, y_train, x_test, y_test, y_pred):
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = x_test.shape[1]
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    return train_score, test_score, mae, rmse, r2, adj_r2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Admin Part modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route("/agent")
def agent_dashboard():
    return render_template("agent_dashboard.html")


# View Property Listings Route
@app.route("/agent/property_listings")
def view_property_listings():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT property_id, address, city, state, price, bedrooms, bathrooms, area_sqft, status, seller_id, seller_contact, seller_email FROM property_listings")
    listings = cursor.fetchall()
    print(listings)
    return render_template("property_listings.html", listings=listings)


@app.route("/agent/add_listing", methods=["GET", "POST"])
def add_property_listing():
    if 'user_id' in session:
        user_id = session.get("user_name")
        if request.method == "POST":
            address = request.form["address"]
            price = request.form["price"]
            bedrooms = request.form["bedrooms"]
            bathrooms = request.form["bathrooms"]
            area_sqft = request.form["area_sqft"]
            description = request.form["description"]
            status = request.form["status"]
            contact_number = request.form["contact_number"]
            contact_email = request.form["contact_email"]
            City = request.form["City"]
            State = request.form["State"]
            date_posted = datetime.now()

            db = get_db()
            cursor = db.cursor()
            sql = """INSERT INTO property_listings (address, city, state, price, bedrooms, bathrooms, area_sqft, description, dt_posted, dt_updated, status, seller_id, seller_contact, seller_email)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            cursor.execute(sql, (address, City, State, price, bedrooms, bathrooms, area_sqft, description, date_posted, date_posted, status, user_id, contact_number, contact_email))
            db.commit()
            flash("New property listing added successfully", "success")
            return redirect(url_for("agent_dashboard"))
        else:
            return render_template("add_listing.html")
    else:
        return redirect('/')

# Update Property Listing Route
@app.route("/agent/update_listing/<int:listing_id>", methods=["GET", "POST"])
def update_property_listing(listing_id):
    if request.method == "POST":
        # Retrieve form data
        address = request.form["address"]
        price = float(request.form["price"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        area_sqft = int(request.form["area_sqft"])
        description = request.form["description"]
        status = request.form.get("status", "available")
        City = request.form["City"]
        State = request.form["State"]
        date_updated = datetime.now()

        # Update the property listing in the database
        db = get_db()
        cursor = db.cursor()
        update_query = """
        UPDATE property_listings
        SET address=?, city=?, state=?, price=?, bedrooms=?, bathrooms=?, area_sqft=?, description=?, dt_updated=?, status=?
        WHERE property_id = ?
        """
        cursor.execute(update_query, (address, City, State, price, bedrooms, bathrooms, area_sqft, description, date_updated, status, listing_id))
        db.commit()
        flash("Property listing updated successfully", "success")
        return redirect(url_for("view_property_listings"))
    else:
        # Retrieve the details of the listing from the database
        db = get_db()
        cursor = db.cursor()
        select_query = "SELECT property_id, address, city, state, price, bedrooms, bathrooms, area_sqft, description, status, seller_contact, seller_email FROM property_listings WHERE property_id = ?"
        cursor.execute(select_query, (listing_id,))
        listing = cursor.fetchone()

        # Render the update form pre-populated with the listing details
        return render_template("update_listing.html", listing=listing)

@app.route("/agent/download_property_data")
def download_property_data():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM property_listings")
    property = cursor.fetchall()
    
    # Convert property data to DataFrame
    df = pd.DataFrame(property)
    
    # Save DataFrame as CSV file
    df.to_csv("property_listings.csv", index=False)
    
    # Return the CSV file as a downloadable attachment
    return send_file("property_listings.csv", as_attachment=True)


@app.route("/agent/delete_listing/<int:listing_id>", methods=["POST", "GET"])
def delete_property_listing(listing_id):
    db = get_db()
    cursor = db.cursor()
    # Delete the property listing from the database based on listing_id
    delete_query = "DELETE FROM property_listings WHERE property_id = ?"
    cursor.execute(delete_query, (listing_id,))
    db.commit()
    flash("Property listing deleted successfully", "success")
    return redirect(url_for('view_property_listings'))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Admin Part modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route("/client")
def client_dashboard():
    return render_template("client_dashboard.html")

# Route for viewing properties
@app.route('/properties')
def view_properties():
    db = get_db()
    cursor = db.cursor()
    select_query = "SELECT property_id, address, city, state, price, bedrooms, bathrooms, area_sqft, status, seller_contact, seller_email FROM property_listings"
    cursor.execute(select_query)
    properties = cursor.fetchall()
    return render_template('view_properties.html', properties=properties)


# Route for buying a property
@app.route('/buy_property/<int:property_id>', methods=['POST'])
def buy_property(property_id):
    user_id = session.get("user_name")
    if 'user_id' in session:
        db = get_db()
        cursor = db.cursor()
        # Check if the property exists and is available
        select_query = "SELECT property_id, address, city, state, price, bedrooms, bathrooms, area_sqft, description, status, DATE(dt_posted), DATE(dt_updated), seller_contact, seller_email FROM property_listings WHERE property_id = ? AND status = 'available'"
        cursor.execute(select_query, (property_id,))
        listing = cursor.fetchone()
        if listing:
            return render_template('confirm_purchase.html', property=listing)
        else:
            # Property not available or does not exist 
            message = "Property not available"
            select_query = "SELECT property_id, address, city, state, price, bedrooms, bathrooms, area_sqft, description, status, seller_contact, seller_email FROM property_listings"
            cursor.execute(select_query)
            properties = cursor.fetchall()
            return render_template('view_properties.html', properties=properties, message = message)
    else:
        return redirect(url_for('login'))


# # Confirm purchase route
@app.route('/confirm_purchase/<int:property_id>', methods=['POST'])
def confirm_purchase(property_id):
    user_id = session.get("user_name")
    if 'user_id' in session:
        # Get the contact details from the form
        buyer_name = request.form['buyer_name']
        buyer_email = request.form['buyer_email']
        buyer_contact_number = request.form['buyer_contact_number']
        contact_message = request.form['buyer_message']
        print(buyer_name, buyer_email, contact_message, buyer_contact_number)
        message = "Your interest in the property has been sent the the seller"
        # For now, let's redirect back to the property listing page
        db = get_db()
        cursor = db.cursor()
        select_query = "SELECT property_id, address, city, state, price, bedrooms, bathrooms, area_sqft, description, status, seller_contact, seller_email FROM property_listings"
        cursor.execute(select_query)
        properties = cursor.fetchall()
        return render_template('view_properties.html', properties=properties, message = message)
    else:
        return redirect(url_for('login'))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Predict house price ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route('/prediction_form')
def prediction_form():
    if 'user_id' in session:
        return render_template('prediction_form.html', cities = city_list, level=session['user_role'])
    else:
        return redirect('/')

@app.route("/predict_individual", methods = ["GET", "POST"])
def predict_individual():
    if 'user_id' in session:
        user_id = session.get("user_name")
        dt = datetime.now()

        # city_list = ['Bangalore', 'Chandigarh', 'Chennai', 'Faridabad', 'Ghaziabad', 'Gurgaon', 'Jaipur', 'Kolkata', 'Lalitpur', 'Lucknow', 'Mohali', 'Mumbai', 'Nagpur', 'Noida', 'Pune', 'Surat', 'Vadodara']
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "select model_location from tbl_model_artifacts where model_status='Active'")
        items = cursor.fetchall()
        if items:
            loc_path = items[0][0]
        else:
            message = "No active model found."
            return render_template("product.html", cities = city_list, message = message)
        l_pth=r""+loc_path
        
        path_folder=os.path.dirname(l_pth)
        
        model = joblib.load(l_pth)
        
        if request.method == "POST":
            rera = int(request.form["rera"])
            print("RERA: ", rera)
            num_of_rooms = int(request.form["num_of_rooms"])
            print("rooms: ", num_of_rooms)
            square_ft = float(request.form["square_ft"])
            sq_ft = np.log(square_ft)
            print("Square_ft:", sq_ft)
            ready_to_move = int(request.form["ready_to_move"])
            print("Ready to move: ", ready_to_move)
            neighbourhood = request.form["neighbourhood_region"]
            city = request.form["city"]
            if city == "--Select--":
                message = "please select a city"
                return render_template("product.html", cities = city_list, message = message)
                
            else:
                city_code = city_list.index(city)
                print("city code: ", city_code)
            posted_by = request.form["posted_by"]
            try:
                location = geolocator.geocode(neighbourhood)
            except:
                message = "Internal error: Location could not be found"
                return render_template("product.html", cities = city_list, message = message)
            
            avg_sqft_per_room = float(sq_ft / num_of_rooms)
            print("average sq. ft per room: ", avg_sqft_per_room)

            if posted_by == 'Builder':
                posted_by_builder = 1
                posted_by_dealer = 0
                posted_by_owner = 0
            elif posted_by == 'Dealer':
                posted_by_builder = 0
                posted_by_dealer = 1
                posted_by_owner = 0
            else:
                posted_by_builder = 0
                posted_by_dealer = 0
                posted_by_owner = 1
                print("posted_by_builder: ", posted_by_builder)
                print("posted_by_dealer: ", posted_by_dealer)
                print("posted_by_owner: ", posted_by_owner)
            if location:
                latitude = location.latitude
                longitude = location.longitude
                input_data = [rera, num_of_rooms, sq_ft, ready_to_move, longitude, latitude, city_code, avg_sqft_per_room, posted_by_builder, posted_by_dealer, posted_by_owner]
                input_array = np.array(input_data).reshape(1, -1)
                x = scaler.transform(input_array)
                print(input_array)
                print(x)
                prediction = 0
                # Make predictions
                try:
                    prediction = model.predict(x)
                    print(prediction[0])
                    prediction = np.exp(prediction)
                    print(prediction[0])
                    Price = f"{prediction[0]:.2f}"
                except:
                    message = "Model could not predict the price. Pease ensure all the values are entered correctly"
                    return render_template('prediction_form.html', cities = city_list, message=message)
                result = {
                    'rera': rera,
                    'num_of_rooms': num_of_rooms,
                    'square_ft': square_ft,
                    'ready_to_move': ready_to_move,
                    'neighbourhood_region': neighbourhood,
                    'city': city,
                    'posted_by': posted_by,
                    'price': Price,
                    'datetime': datetime.now()
                }
                # store the prediction in the database
                statement = "INSERT INTO tbl_prediction(RERA, num_rooms, square_ft, ready_to_move, longitude, latitude, city_code, avg_sqft_per_room, posted_by_builder, posted_by_dealer, posted_by_owner, Price, dt_pred, pred_type, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                cursor.execute(statement, [rera, num_of_rooms, square_ft, ready_to_move, longitude, latitude, city_code, avg_sqft_per_room, posted_by_builder, posted_by_dealer, posted_by_owner, Price, dt, "individual", user_id])
                db.commit()

                return render_template("prediction_results.html", result=result)
            else:
                message = "Location not found"
                return render_template("product.html", cities = city_list, message = message)
        return render_template("product.html", cities = city_list, message = message)
    else:
        return redirect('/')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bulk prediction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route('/bulk')
def bulk():
    if 'user_id' in session:
        return render_template('upload_bulk.html', level=session['user_role'])
    else:
        return redirect('/')

@app.route("/predict_bulk", methods=["GET", "POST"])
def predict_bulk():
    if "user_id" in session:
        if request.method == 'POST':
            tmstp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S_%f')
            path_loc = os.path.abspath(os.path.join('prediction', tmstp))
            os.makedirs(path_loc, exist_ok=True)
            
            f = request.files['file']
            fname = f.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(file_path)
            
            if fname == "":
                flash("Please Upload File !!")
                return render_template('upload_bulk.html')

            if fname.split('.')[-1] != "csv":
                flash("Please Upload CSV format !!")
                return render_template('upload_bulk.html')

            data = pd.read_csv(file_path)
            data.columns = ["rera", "num_of_rooms", "sq_ft", "ready_to_move", "city", "address", "posted_by"]

            data[["rera", "ready_to_move"]] = data[["rera", "ready_to_move"]].apply(label_encoder.fit_transform)
            data["sq_ft"] = np.log(data["sq_ft"])
            data['city'] = data['city'].apply(lambda x: city_encoder.transform([x])[0] if x in city_encoder.classes_ else -1)

            # Remove rows where city encoding resulted in -1
            data = data[data['city'] != -1]

            # Handling location coordinates, we initially set the coordinates to -1 and 
            # then loop over the address in the data rows to extrct the location coordinates of the address
            data["latitude"] = -1.0
            data["longitude"] = -1.0
            for idx, row in data.iterrows():
                location = geolocator.geocode(f"{row['address']}, {row['city']}")
                if location:
                    data.at[idx, "latitude"] = location.latitude
                    data.at[idx, "longitude"] = location.longitude

            # If there still exist any coordinates that are -1 we remove those data points from the dataset
            data = data[(data["latitude"] != -1.0) & (data["longitude"] != -1.0)]
            data["avg_sqft_per_room"] = data["sq_ft"] / data["num_of_rooms"]

            data["posted_by_builder"] = data["posted_by"].str.lower().apply(lambda x: 1 if x == 'builder' else 0)
            data["posted_by_dealer"] = data["posted_by"].str.lower().apply(lambda x: 1 if x == 'dealer' else 0)
            data["posted_by_owner"] = data["posted_by"].str.lower().apply(lambda x: 1 if x == 'owner' else 0)

            cols_drop = ["address", "posted_by"]
            df = data.drop(cols_drop, axis=1)

            final_columns = ["rera", "num_of_rooms", "sq_ft", "ready_to_move", "longitude", "latitude", "city", "avg_sqft_per_room", 
                             "posted_by_builder", "posted_by_dealer", "posted_by_owner"]
            df = df[final_columns]

            db = get_db()
            cursor = db.cursor()
            cursor.execute("select model_location from tbl_model_artifacts where model_status='Active'")
            loc_path = cursor.fetchone()[0]
            
            model = joblib.load(loc_path)
            df_scaled = scaler.transform(df)

            data['prediction'] = model.predict(df_scaled)
            data['prediction'] = round(data['prediction'], 2)
            data['datetime'] = datetime.now()

            for index, row in data.iterrows():
                statement = """INSERT INTO tbl_prediction
                               (RERA, num_rooms, square_ft, ready_to_move, longitude, latitude, city_code, avg_sqft_per_room, 
                               posted_by_builder, posted_by_dealer, posted_by_owner, Price, dt_pred, pred_type, user_id)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                cursor.execute(statement, [row["rera"], row["num_of_rooms"], np.exp(row["sq_ft"]), row["ready_to_move"], row["longitude"], 
                                           row["latitude"], row["city"], row["avg_sqft_per_room"], row["posted_by_builder"], 
                                           row["posted_by_dealer"], row["posted_by_owner"], row["prediction"], datetime.now(), 
                                           'bulk', session['user_id']])
            db.commit()

            cursor.execute("select * FROM tbl_prediction where pred_type='bulk' order by dt_pred desc")
            items = cursor.fetchall()
            payload = []
            for item in items:
                content = {'ID': item[0], 'RERA': item[1], 'NUM_OF_ROOMS': item[2], 'AREA(SQ.FT)': item[3], 'READY_TO_MOVE': item[4], 
                           'LONGITUDE': item[5], 'LATITUDE': item[6], 'CITY': item[7], 'AVG_SQ_FT_PER_ROOM': item[8], 
                           'POSTED_BY_BUILDER': item[9], 'POSTED_BY_DEALER': item[10], 'POSTED_BY_OWNER': item[11], 
                           'PREDICTIONS': item[12], 'dt_pred': item[21]}
                payload.append(content)

            return render_template('bulk_prediction.html', payload=payload, level=session['user_role'])
    else:
        return redirect('/')

# Prediction history
@app.route("/prediction_history")
def prediction_history():
    if "user_id" in session:
        user_id = session.get("user_name")
        db = get_db()
        cursor = db.cursor()
        sql = """
            SELECT RERA, num_rooms, square_ft, ready_to_move, longitude, latitude, city_code, 
                   posted_by_builder, posted_by_dealer, posted_by_owner, Price
            FROM tbl_prediction 
            WHERE user_id = ? AND pred_type = 'individual'
            ORDER BY dt_pred DESC 
            LIMIT 5
        """
        cursor.execute(sql, (user_id,))
        predictions = cursor.fetchall()
        
        # Pass the predictions to the template
        return render_template('prediction_history.html', predictions=predictions)
    else:
        return redirect('/login')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Registration module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route("/register")
def register():
    return render_template("register.html")

# Registering the users and adding their details into the database 
@app.route("/add_user", methods = ["POST"])
def add_user():
    if request.method == "POST":
        user_email = request.form["user_email"]
        user_password = request.form["user_password"]
        confirm_password = request.form["confirm_password"]
        user_role = request.form["user_role"]

        encrypt_password = password_manager.hash(user_password)
        dt = datetime.now()
        token = "default"

        if user_password == confirm_password:
            db = get_db()
            cursor = db.cursor()
            sql = "select email, pass_word from register_user where email = ?"
            cursor.execute(sql, (user_email,))
            users = cursor.fetchall()
            
            if users:
                message = "User already exists. Please Login"
                return render_template("register.html", message = message)
            else:
                query = "insert into register_user (email, pass_word, token, user_status, user_level, dt_time) values (?, ?, ?, ?, ?, ?)"
                cursor.execute(query, (user_email, encrypt_password, token, "registered", user_role, dt))
                db.commit()
                cursor.close()
                message = "User registered successfully. Kindly login."
                return render_template("register.html", message = message)
        else:
            message = "Password mismatch"
            return render_template("register.html", message = message)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Login module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route("/login")
def login():
    return render_template("login.html")


# Login validation for users
@app.route('/login_validation', methods = ["POST", "GET"])
def login_validation():
    if request.method == "POST":
        login_email = request.form["email"]
        login_password = request.form["password"]

        db = get_db()
        cursor = db.cursor()
        sql = "select name, email, pass_word, user_level, user_status from register_user where email = ?"
        cursor.execute(sql, (login_email,))
        user = cursor.fetchone()

        if user:
            user_name = user[0]
            hashed_password = user[2]
            user_role = user[3]
            user_approve_status = user[4]
            dt = datetime.now()

            if password_manager.verify(login_password, hashed_password):
                flash("Logged in successfully")
                session.permanent = True
                session["user_name"] = user_name
                session["user_id"] = login_email
                session["user_role"] = user_role

                session["user_approve_status"] = user_approve_status
                statement = "INSERT into user_logins (email, user_level, dt_time) VALUES (?, ?, ?)"
                cursor.execute(statement, (login_email, user_role, dt))
                db.commit()
                cursor.close()
                return redirect((url_for("dashboard")))
            else:
                message = "Invalid credentials. Please enter a valid passeord"
                return render_template("login.html", message = message)
        else:
            message = "User not found. Please enter valid credentials or register"
            return render_template("login.html", message = message)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Navigate to appropriate dashboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route('/dashboard')
def dashboard():
    user_role = session.get('user_role')
    user_approval_status = session.get("user_approve_status")
    if user_role == 'admin':
        if user_approval_status == "approved":
            return redirect(url_for('admin_dashboard'))
        else:
            message = "Your approval is pending. Kindly contact administrator"
    elif user_role == 'agent':
        if user_approval_status == "approved":
            return redirect(url_for('agent_dashboard'))
        else:
            message = "Your approval is pending. Kindly contact administrator"
    elif user_role == 'client':
        if user_approval_status == "approved":
            return redirect(url_for('client_dashboard'))
        else:
            message = "Your approval is pending. Kindly contact administrator"
            return render_template("login.html", message=message)
    else:
        message = "Unauthorized access. Please log in."
        # Handle other user roles or unauthorized access
    return render_template("login.html", message=message)

@app.route('/product')
def product():
    if 'user_id' in session:
        # city_list = ['Bangalore', 'Chandigarh', 'Chennai', 'Faridabad', 'Ghaziabad', 'Gurgaon', 'Jaipur', 'Kolkata', 'Lalitpur', 'Lucknow', 'Mohali', 'Mumbai', 'Nagpur', 'Noida', 'Pune', 'Surat', 'Vadodara']
        return render_template('product.html', level=session['user_role'], cities = city_list)
    else:
        return redirect('/')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logout module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route("/logout")
def logout():
    session.clear()
    return redirect('/')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Index page/root page ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route('/')
def index():
    return render_template('index.html')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Running the flask app ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__== "__main__":
    app.run(debug = True)
