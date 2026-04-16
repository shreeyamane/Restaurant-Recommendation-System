##Project Overview
This project is a FastAPI-based restaurant recommendation system powered by a content-based filtering model. Users can search for a restaurant and refine recommendations using budget, rating, and cuisine preferences. The project also includes data visualizations that summarize restaurant trends from the dataset.

##Features
Restaurant recommendations based on similarity
Filtered results using budget, minimum rating, and cuisine
Friendly error messages when a restaurant is not found or no filtered matches are available
Login and registration with SQLite-backed user storage
Insights page for dataset visualizations
Popular cuisines section on the homepage
Tech Stack
Python
FastAPI
Scikit-learn
Pickle
HTML
CSS
Jupyter Notebook
Screenshots
Home Page
Home Page

Insights Page
Insights

Dataset Visualizations
Rating Distribution Popular Cuisines

How to Run
Create and activate a virtual environment.
Install the required packages.
Run the FastAPI app from the project root.
pip install -r requirements.txt
uvicorn app:app --reload
Add your default login credentials in .env:
APP_USER_EMAIL=shreeyash2573@gmail.com
APP_USER_PASSWORD=Shreeyash@2516
FLASK_SECRET_KEY=restaurant-recommender-secret-key
Open the local FastAPI URL in your browser, usually http://127.0.0.1:8000.
Log in with the seeded user or create a new account from the register page.
Visit /insights to view the saved visualizations in the web app.
Project Structure
app/
  static/
    css/
    images/
  templates/
    index.html
    result.html
    insights.html
app.py
model/
  build_model.ipynb
  recommend.py
  restaurant.pkl
  tfidf.pkl
  tfidf_matrix.pkl
Notes
Visualization assets are served from app/static/images/.
The notebook remains available for model building and analysis.
The web app now exposes the charts through the /insights route.
