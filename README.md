# Restaurant Recommendation System

## Project Overview

This project is a FastAPI-based restaurant recommendation system powered by a content-based filtering model. Users can search for a restaurant and refine recommendations using budget, rating, and cuisine preferences. The project also includes data visualizations that summarize restaurant trends from the dataset.

## Features

- Restaurant recommendations based on similarity
- Filtered results using budget, minimum rating, and cuisine
- Friendly error messages when a restaurant is not found or no filtered matches are available
- Login and registration with SQLite-backed user storage
- Insights page for dataset visualizations
- Popular cuisines section on the homepage

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pickle
- HTML
- CSS
- Jupyter Notebook

## Screenshots

### Home Page

![Home Page](<img width="1919" height="1009" alt="Dashbord" src="https://github.com/user-attachments/assets/e93d5d55-f354-44da-946b-f2ac2740b03a" />
)

### Insights Page

![Insights](<img width="1915" height="1019" alt="User Input" src="https://github.com/user-attachments/assets/eff5484f-609b-44c4-951d-1b59820a6f91" />
)

### Dataset Visualizations

![Rating Distribution](<img width="1000" height="500" alt="rating_distribution" src="https://github.com/user-attachments/assets/0d959d1f-c519-4f0e-b2d2-cc47391a2785" />
)
![Popular Cuisines](<img width="1000" height="600" alt="cuisine_freq" src="https://github.com/user-attachments/assets/34b530f9-68e1-4970-a5e7-3a3630c90055" />
)

## How to Run

1. Create and activate a virtual environment.
2. Install the required packages.
3. Run the FastAPI app from the project root.

pip install -r requirements.txt
uvicorn app:app --reload

4. Open the local FastAPI URL in your browser, usually `http://127.0.0.1:8000`.
5. Log in with the seeded user or create a new account from the register page.
6. Visit `/insights` to view the saved visualizations in the web app.

## 📂 Project Structure

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

## Notes

- Visualization assets are served from `app/static/images/`.
- The notebook remains available for model building and analysis.
- The web app now exposes the charts through the `/insights` route.
