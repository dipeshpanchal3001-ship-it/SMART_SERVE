# 🍽️ SmartServe — Food Waste Risk Prediction

> A machine-learning project that classifies the **food-waste risk** (Low / Medium / High) of a planned food-service operation *before* the food is prepared.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Type-Academic%20Project-green)

---

## 📌 Overview

Food waste happens when the quantity of food prepared is greater than the quantity demanded. **SmartServe** is a notebook-based machine-learning workflow that uses planning information — expected customers, previous-day sales, prices, discounts, meal type, weather, special events and more — to predict whether a planned preparation is likely to have **Low**, **Medium** or **High** waste risk.

It is aimed at establishments such as restaurants, hotels, college canteens, hostel messes and catering services.

The project covers the complete ML workflow in a single Jupyter Notebook:

1. Synthetic dataset generation
2. Data inspection and cleaning
3. Exploratory data analysis (EDA)
4. Preprocessing and pipeline building
5. Model training and comparison
6. Prediction on an unseen record

> **Note:** The dataset is **synthetic** and was generated for learning and experimentation. It is not made up of real restaurant or canteen records.

---

## 🎯 Objectives

- Create a structured synthetic dataset for food-service planning.
- Check for missing values, duplicates, invalid values and outliers, then save a cleaned dataset.
- Perform EDA with statistical summaries and visualizations.
- Train and compare a **Baseline**, **Decision Tree**, **K-Nearest Neighbors** and **Random Forest** classifier.
- Use **K-Means** as an unsupervised technique for exploratory grouping.
- Evaluate models using **Accuracy, Precision, Recall, F1-score and ROC-AUC**.
- Use the final **Random Forest pipeline** to predict risk for a new, unseen record.

---

## 📊 Dataset

| Property | Value |
| --- | --- |
| File | `SmartServe_food_waste_dataset.csv` |
| Records | 13,248 |
| Target column | `waste_risk` |
| Classes | Low, Medium, High |

**Class distribution**

| Class | Records | Share |
| --- | ---: | ---: |
| Low | 4,080 | 30.8% |
| Medium | 4,203 | 31.7% |
| High | 4,965 | 37.5% |
| **Total** | **13,248** | **100%** |

### Input features

| Feature | Description |
| --- | --- |
| `establishment_type` | Restaurant, canteen, hotel, hostel mess, caterer, etc. |
| `city` | City of the establishment |
| `cuisine_type` | Cuisine category |
| `food_item` | Name of the food item being planned |
| `item_category` | Category of the item (e.g. main course) |
| `meal_type` | Breakfast, lunch, dinner, etc. |
| `day_of_week` | Day on which the food is planned |
| `weather_condition` | Sunny, cold, rainy, etc. |
| `special_event` | Whether a special event may affect demand |
| `previous_day_sales` | Sales information from the previous day |
| `price_per_plate_inr` | Selling price per plate (₹) |
| `cost_per_plate_inr` | Cost per plate (₹) |
| `discount_applied_pct` | Discount applied (%) |
| `expected_customers` | Estimated number of customers |
| `quantity_prepared` | Planned quantity of food to be prepared |

### 🚫 Data leakage prevention

The dataset also contains outcome columns such as `quantity_sold`, `leftover_quantity` and `waste_percentage`. These are only known **after** service, so they are **excluded from the model inputs**. This keeps the model realistic for pre-service prediction.

---

## 🧠 Models Used

| Model | Type | Role |
| --- | --- | --- |
| Baseline classifier | Supervised | Reference point for comparison |
| Decision Tree | Supervised | Interpretable tree-based classifier |
| K-Nearest Neighbors (KNN) | Supervised | Distance-based classifier |
| Random Forest | Supervised (ensemble) | **Final selected model** |
| K-Means | Unsupervised | Exploratory grouping of records |

The final prediction workflow uses a Scikit-learn **Pipeline** that combines preprocessing and `RandomForestClassifier(random_state=42)`, so the same transformations are applied consistently during training and prediction.

---

## 📈 Results

Evaluation on the held-out test set (synthetic data):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.4080 | 0.1665 | 0.4080 | 0.2365 | — |
| Decision Tree | 0.9018 | 0.9025 | 0.9018 | 0.9021 | 0.9579 |
| KNN | 0.6237 | 0.6347 | 0.6237 | 0.6241 | 0.8023 |
| **Random Forest** | **0.9304** | **0.9314** | **0.9304** | **0.9307** | **0.9906** |

**Key takeaways**

- Random Forest achieved the best results across all metrics.
- Decision Tree also performed strongly; KNN performed noticeably lower.
- A tuned Random Forest was also tried but reached a lower accuracy (**0.8477**), so the **untuned Random Forest** was kept for the final prediction workflow.

> ⚠️ These metrics describe performance on **synthetic data** and should not be treated as validated real-world performance.

---

## 📁 Project Structure

```
SmartServe/
│
├── Phases.ipynb                          # Main notebook (full workflow)
├── SmartServe_food_waste_dataset.csv     # Original synthetic dataset
├── smartserve_cleaned_data_phase02.csv   # Cleaned dataset (output of data cleaning phase)
├── .ipynb_checkpoints/                   # Jupyter auto-generated checkpoints
└── .jupyter/                             # Jupyter supporting folder
```

---

## 🛠️ Tech Stack

- **Python** — core language
- **Jupyter Notebook** — development environment
- **Pandas** & **NumPy** — data handling and numerical operations
- **Matplotlib** & **Seaborn** — visualization and EDA
- **Scikit-learn** — preprocessing, modelling, clustering and evaluation

### System requirements

| Component | Requirement |
| --- | --- |
| OS | Windows, Linux or macOS |
| Processor | Modern dual-core or better |
| RAM | 4 GB minimum (8 GB+ recommended) |
| Storage | ~1 GB free space |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/dipeshpanchal3001-ship-it/SMART_SERVE.git
cd SMART_SERVE
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook
```

Open `Phases.ipynb` and run the cells **in order** from top to bottom.

---

## 🔮 Making a Prediction

The final section of the notebook predicts the risk category for one unseen record:

1. Load all the cell, the dataset and separate `X` (input features) and `y` (`waste_risk`).
2. Split into training and testing sets.
3. Build the preprocessing + Random Forest pipeline.
4. Fit the pipeline on the training data.
5. Create a single-row DataFrame with the same input columns used in training.
6. Call `predict()` and read the returned class: **Low**, **Medium** or **High**.

Illustrative example (edit the values in the notebook; categorical values must match categories present in the dataset):

```pythonunseen_data.loc[0, "establishment_type"] = "Hotel"
unseen_data.loc[0, "city"] = "Ahmedabad"
unseen_data.loc[0, "cuisine_type"] = "Indian"
unseen_data.loc[0, "food_item"] = "Pizza"
unseen_data.loc[0, "item_category"] = "Main Course"
unseen_data.loc[0, "meal_type"] = "Lunch"
unseen_data.loc[0, "day_of_week"] = "Sunday"
unseen_data.loc[0, "weather_condition"] = "Sunny"
unseen_data.loc[0, "special_event"] = "Yes"
unseen_data.loc[0, "previous_day_sales"] = 100
unseen_data.loc[0, "expected_customers"] = 180
unseen_data.loc[0, "price_per_plate_inr"] = 180
unseen_data.loc[0, "cost_per_plate_inr"] = 130
unseen_data.loc[0, "discount_applied_pct"] = 0
unseen_data.loc[0, "planned_buffer"] = 20
```

The output is a **risk category**, not an exact quantity of waste. Use it as a decision-support indication when reviewing planned preparation quantities.

---

## ⚠️ Limitations

- **Synthetic data** — strong scores here do not guarantee similar performance on real food-service data.
- **Risk category, not exact waste** — the model predicts Low / Medium / High, not an amount or percentage.
- **Input quality matters** — unrealistic values for customers, quantity, prices or sales will affect predictions.
- **Class imbalance** — classes are not perfectly balanced, so accuracy alone can hide weaker per-class performance.
- **No real-world validation** — the model has not been trialled in a live food-service business.
- **Notebook-only** — there is no dedicated user interface or deployed application.
- **Changing conditions** — unexpected events, supply issues and local factors may not be captured.

---

## 🌱 Future Enhancements

- Collect authorised, anonymised **real-world** data from food-service establishments.
- Expand the data across more cities, seasons, cuisines and establishment sizes.
- Re-train and validate the model regularly on new records.
- Report **class-wise** precision, recall and F1 (especially for High-risk cases).
- Test more algorithms with cross-validation, tuned carefully to avoid data leakage.
- Add **explainability** (feature importance) to show which factors drive a prediction.
- Build a separate **user-facing interface** for restaurant staff.
- Run a controlled **pilot** with a participating establishment.

---

## 👨‍🎓 Author

**Dipesh Panchal**
Enrollment No.: 202312101920
Bachelor of Computer Applications (Honours) — Semester VII
Asia Pacific BCA College, Gujarat University, Ahmedabad, Gujarat
Academic Year: 2026–2027

**Project Guide:** Mrs. Naina Asnani

---

## 📚 References

- [Python Documentation](https://docs.python.org/3/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 📝 Disclaimer

SmartServe is an academic project built on a synthetic dataset. Its predictions are for learning and experimentation and should not be used as the sole basis for real operational decisions.
