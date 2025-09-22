# Car Price Prediction with Images + Tabular Data

This project explores building a neural network that predicts car prices using both **visual features (images of cars)** and **structured/tabular data (e.g., mileage, year, gearbox, body type)**.  
The final model was deployed using [Streamlit](https://carpricepredictiondemo-rzlfjybe2shspfzl4sr9az.streamlit.app/).

---

## 🚀 Project Overview
- Built a **hybrid prediction system** combining:
  - A **ResNet34 CNN** (transfer learning) for extracting features from car images.
  - A set of **engineered tabular features** such as car age, mileage per year, registration year, fuel type, gearbox type, etc.
- Tackled challenges in **skewed data distribution** (many cheap cars, few expensive cars).
- Experimented with:
  - Log transformation & standardization of prices.
  - Stratified splits by price ranges.
  - Upsampling rare expensive cars.
  - Weighted losses and robust loss functions.
  - Feature engineering & categorical encoding for tabular data.
- Evaluated different model combinations:  
  - **Image-only**  
  - **Tabular-only**  
  - **Image + Tabular embeddings**

---

## 📊 Key Results

| Model Type                      | RMSE     | MAE     | MAPE   | R²    |
|---------------------------------|----------|---------|--------|-------|
| **Image Only (ResNet34)**       | 22,746.9 | 3,645.3 | 19.65% | 0.562 |
| Tabular Only                    | 31,773.5 | 10,231.2| 64.40% | 0.146 |
| Tabular + Image Embeddings      | 25,130.0 | 3,207.1 | 16.00% | 0.466 |

- **Best performance** was achieved with the **image-only CNN**, explaining ~56% of variance (R²) with ~20% MAPE.  
- Adding tabular data did not consistently improve results, and often led to overfitting.

---

## 🧪 Methodology & Experiments

1. **Baseline CNN**: Started with ResNet34 + MSE loss → poor performance due to price skew.  
2. **Log Transform**: Applied log + standardization → reduced skew, improved stability.  
3. **Transfer Learning**: Freezing/unfreezing layers with tuned learning rate → best CNN baseline (~19% MAPE).  
4. **Stratified Splits**: Ensured both cheap and expensive cars were in validation → more realistic (but worse) results.  
5. **Upsampling & Weighted Losses**: Tried to balance rare expensive cars → led to overfitting or collapse.  
6. **Huber Loss**: Tested for robustness against outliers → no improvement.  
7. **Tabular Features**: Engineered continuous + categorical features (car age, gearbox type, fuel type, body type, etc.) and combined with CNN embeddings → often worse due to noise.  

---

## 🌐 Deployment

The final demo is available on **Streamlit**:  
👉 [Car Price Prediction App](https://carpricepredictiondemo-rzlfjybe2shspfzl4sr9az.streamlit.app/)

Features:
- Upload a **car image**.
- Input **structured attributes** (mileage, year, gearbox, etc.).
- Get a **predicted price**.

---

## ⚙️ Tech Stack
- **Python** (fastai, PyTorch, scikit-learn, XGBoost)
- **Streamlit** (for deployment)
- **Pandas, NumPy** (data preprocessing)
- **Matplotlib** (visualizations)

---

## 📈 Lessons Learned
- CNNs trained on **images alone** captured a lot of useful signal, explaining ~55% of variance.  
- Adding structured/tabular data **did not improve performance** in this dataset due to noise and possible overfitting.  
- Handling **skewed target distributions** is critical for regression tasks.  
- Validation strategy matters: stratified splits gave a more **realistic but harsher** estimate of performance.  

---

## 🔮 Future Work
- Use **permutation importance** or SHAP values to identify truly useful tabular features.  
- Explore **transformer-based multimodal models** for better fusion of image + structured data.  
- Acquire more balanced data across **different price ranges**.  
- Try **contrastive pretraining** on car images to improve embeddings before regression.  

---

## 📜 License
This project is released under the MIT License.
