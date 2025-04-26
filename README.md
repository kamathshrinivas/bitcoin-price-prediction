# Bitcoin Price Prediction Using LSTM

This project predicts Bitcoin price using LSTM (Long Short-Term Memory) model trained on historical price data from Yahoo Finance.

## Libraries
- TensorFlow / Keras
- Pandas
- Scikit-learn
- Matplotlib

## How to Run

1. Install dependencies:  
```bash
pip install -r requirements.txt
```

2. Add `btc_dataset.csv` from Yahoo Finance.

3. Run the script:
```bash
python bitcoin_lstm.py
```

---
## 📈 Output

You’ll see a plot of Actual vs Predicted Bitcoin price using LSTM.

---
### 🔍 Graph Explanation

- **Blue Line**: Actual Bitcoin prices over time (from test dataset).
- **Orange Dashed Line**: Prices predicted by the LSTM model.

The graph above visualizes how closely the model's predictions follow the real prices.  
It shows the effectiveness of LSTM in learning patterns in time-series data like cryptocurrency prices.

The better the predicted line aligns with the actual price, the more accurate the model is.  
This model achieved **high prediction performance** as reflected by visual accuracy and RMSE (Root Mean Squared Error) score.

---

## 🛠 How It Works

1. Historical data is fetched from Yahoo Finance (`btc_dataset.csv`)
2. Data is normalized using `MinMaxScaler`
3. A deep learning model using **TensorFlow/Keras** is built:
   - 2 LSTM layers
   - 1 Dense output layer
4. The model is trained and then tested on unseen data.
5. A plot is generated comparing actual vs predicted prices.

---



