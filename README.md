# Movie Recommender System

A Python-based movie recommendation system built on the MovieLens 100K dataset. Given a movie ID, it finds the most similar titles by comparing genres, average user ratings, and normalized popularity scores using a K-Nearest Neighbors (KNN) distance approach.

## How It Works

Instead of relying purely on raw ratings (where a movie rated 5.0 by 2 people would outrank a classic rated 4.3 by 500 people), the notebook builds a combined feature vector for each movie:

1. **Data Loading & Cleaning:** Reads user ratings (`u.data`) and movie metadata (`u.item`, using `latin` encoding to handle special characters). Drops unused metadata columns like IMDb URLs and release dates.

2. **Feature Engineering:**

   * **Genres:** Uses the 18 one-hot encoded genre columns (`Action`, `Adventure`, `Comedy`, `Drama`, `Sci-Fi`, etc.).

   * **Average Rating:** Aggregates user ratings per movie (`avg_rating`).

   * **Popularity Score:** Computes the rating count per movie and applies Min-Max scaling so values fall cleanly between `0` and `1`:

     $\text{popularity} = \frac{x - \min(x)}{\max(x) - \min(x)}$

3. **Exploratory Analysis:** Visualizes user activity and rating-count distributions (showing the long-tail / right-skewed pattern where most movies have relatively few ratings).

4. **Similarity Search (KNN):** Merges genre vectors with `avg_rating` and `popularity`, then runs `similarMovies(mid, k)` to calculate Euclidean distances (`scipy.spatial.distance.euclidean`) against all other movies and return the top $k$ closest matches.

## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) from GroupLens Research, included in the `ml-100k/` folder:

* **100,000** ratings (scale of 1–5, overall average ~3.53)

* **943** unique users

* **1,682** unique movies across 18 genres

## Tech Stack

* **Python 3**

* **Pandas & NumPy** — data manipulation and aggregation

* **SciPy** — Euclidean and Cosine distance calculations

* **Matplotlib & Seaborn** — exploratory data visualization

* **Jupyter Notebook / Google Colab** — development environment

## Project Structure

```text
Movie Recommender System/
├── ml-100k/                     # MovieLens 100K dataset files (u.data, u.item, etc.)
├── Movie Recommendation.ipynb   # Main notebook with EDA, preprocessing, and KNN recommender
├── Project Report.pdf           # Detailed project documentation
├── requirements.txt             # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

Make sure you have **Python 3** installed.

### Setup & Running the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/Disha-Saxenaa/Movie-Recommender-System.git
   cd "Movie-Recommender-System"
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook "Movie Recommendation.ipynb"
   ```

4. Run all cells from top to bottom. Make sure the `ml-100k/` folder stays in the same directory as the notebook so the relative paths (`ml-100k/u.data` and `ml-100k/u.item`) resolve properly.

### Getting Recommendations

Call `similarMovies(mid, k)` with any valid MovieLens `movie_id` and the number of results you want:

```python
similarMovies(50, k=10)   # Star Wars (1977)
similarMovies(127, k=10)  # Godfather, The (1972)
similarMovies(313, k=10)  # Titanic (1997)
```

## Example Output

Running `similarMovies(50, k=10)` for **Star Wars (1977)** returns:

| Movie ID | Movie Title |
| ----- | ----- |
| 50 | Star Wars (1977) |
| 181 | Return of the Jedi (1983) |
| 172 | Empire Strikes Back, The (1980) |
| 498 | African Queen, The (1951) |
| 222 | Star Trek: First Contact (1996) |
| 228 | Star Trek: The Wrath of Khan (1982) |
| 271 | Starship Troopers (1997) |
| 82 | Jurassic Park (1993) |
| 121 | Independence Day (ID4) (1996) |
| 174 | Raiders of the Lost Ark (1981) |

Running `similarMovies(127, k=5)` for **Godfather, The (1972)** returns:

| Movie ID | Movie Title |
| ----- | ----- |
| 127 | Godfather, The (1972) |
| 187 | Godfather: Part II, The (1974) |
| 56 | Pulp Fiction (1994) |
| 806 | Menace II Society (1993) |
| 182 | GoodFellas (1990) |

## Future Improvements

* Allow searching by movie title string instead of numeric `movie_id` (and exclude the queried movie itself from the top-$k$ list).

* Incorporate user demographic data from `u.user` (age, occupation) and build a user-item rating matrix for true user-based collaborative filtering or SVD.

* Vectorize distance calculations with Scikit-Learn's `NearestNeighbors` for faster lookup on larger datasets like MovieLens 1M/25M.

* Add offline evaluation metrics (such as Precision@K or RMSE on the `u1.base`/`u1.test` splits).

## Author

**Disha Saxena** — Developed as a B.Tech Computer Science & Engineering project at KIIT Deemed to be University.
