# Real Estate Management System

## Overview

The Real Estate Management System is a comprehensive platform designed to manage property listings, user roles (Admin, Agent, Client), and predictive analytics for property prices. The application enables agents to list properties, clients to view and predict property prices, and admins to manage users and monitor model training and prediction processes.

## Features

- **User Roles**: Admin, Agent, and Client dashboards, each with specific functionality.
- **Property Listings**: Agents can add, update, and view property listings.
- **Predictive Analytics**: Integrated machine learning models to predict property prices based on user input.
- **Model Management**: Admins can view, activate, and manage trained machine learning models.
- **User Management**: Admins can approve or disapprove users and assign roles (Admin, Agent, Client).
- **Prediction History**: View recent individual predictions.
  
## Screenshots
Include screenshots here to give users a visual of how the application looks. You can upload images to the repository and reference them in this section.

## Project Structure

```bash
├── app.py                # Main Flask Application 
├── db.py                 # Database setup and management 
├── models/               # Trained machine learning models stored here 
├── static/               # Static files (CSS, JS, images) 
├── templates/            # HTML templates for the application 
├── README.md             # Documentation 
├── requirements.txt      # Required dependencies 
└── .env                  # Environment variables (e.g., for Flask secret key) 
```

## Prerequisites
Before running the project, ensure that you have the following installed:

Python 3.x
Flask
SQLite (for database)
scikit-learn
pandas
numpy
joblib (for model serialization)

## Setup and Installation
**1. Clone the repository**
```sh
git clone https://github.com/yourusername/Real-Estate-Management-System.git
cd Real-Estate-Management-System
```
**2. Set up a virtual environment (optional but recommended)**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
**3. Install the required dependencies**
```sh
pip install -r requirements.txt
```
**4. Database setup**
Ensure SQLite is installed and run the following commands to create the required database tables:
```sh
python db.py
```
This will set up tables for users, property listings, and model artifacts.

### 5. Model Training

- Admins can upload datasets to train machine learning models.
- Once a model is trained, it can be activated for predictions.
- Use the `/train_models` route for training new models.

### 6. Running the Application

Start the Flask application by running:

```bash
python app.py
```

The app will be accessible at `http://127.0.0.1:5000/` by default.

## Usage

### Admin Dashboard

- View and manage user accounts.
- Train, activate, and manage machine learning models for property price predictions.

### Agent Dashboard

- Add, update, and manage property listings.
- View property listings in the system.
- Check prediction history for properties.

### Client Dashboard

- View available properties.
- Predict property prices using the integrated model.
- View recent predictions.

## Model Training and Predictions

- **Admin Role**: Admins can upload datasets, train models, and activate specific models for prediction.
- **Prediction Process**: Agents and clients can predict property prices based on user inputs such as RERA, number of rooms, area (sq. ft.), and city.

## API Endpoints

Here are some of the key routes:

| Route                     | Method | Role  | Description                                    |
|----------------------------|--------|-------|------------------------------------------------|
| `/login`                   | GET    | All   | User login                                    |
| `/dashboard`               | GET    | All   | Redirects to the appropriate user dashboard    |
| `/admin/user_approval`     | POST   | Admin | Approve/disapprove users                      |
| `/agent/property_listings` | GET    | Agent | View all property listings                    |
| `/client/prediction`       | POST   | Client| Make individual property price predictions    |
| `/admin/train_models`      | POST   | Admin | Train machine learning models                 |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/new-feature`.
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
