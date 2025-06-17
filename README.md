![image](https://github.com/user-attachments/assets/f348334e-f36e-4be0-91ab-cdc32e9a68bb)


CodeForces Efficiency Predictor and Contest Standings Tool  

Overview:
             The CodeForces Efficiency Predictor & Contest Standings Tool is a Flask-based web application that offers both rating prediction analytics and contest standing filters using the Codeforces public API
                 
                      
Key Features
Efficiency-Based Rating Prediction: Computes rating-per-contest metrics (overall, recent, peak) to forecast future performance using a blend of polynomial regression and Random Forest models.

Pattern Recognition: Automatically classifies user progression into patterns such as accelerating, stable, converged, and declining based on statistical and efficiency metrics.

Real-Time Contest Standings Filter: Retrieves full contest standings and filters participants by organization keywords in batches to respect API rate limits.

Professional Visualizations: Generates thread-safe non-GUI matplotlib charts with confidence intervals, color-coded rating bands, and an unobtrusive statistics box.

Responsive UI: Clean, mobile-friendly HTML/CSS design ensuring a seamless user experience across devices.


   Installation
Clone the repository:

bash
git clone https://github.com/yourusername/codeforces-efficiency-predictor.git  
cd codeforces-efficiency-predictor  
Install required packages:
  python app.py  
  Open your browser at http://localhost:5000.

Run the application:


![image](https://github.com/user-attachments/assets/ca56d784-ff67-481b-a95d-de2df4a41935)
![image](https://github.com/user-attachments/assets/fc71ec77-aa1b-47e3-b140-62d21b8c7624)

 Usage
   
   Contest Standings

   
   Go to the Home page.
   Enter a Contest ID (e.g., 1567).
   Enter comma-separated Organizations (e.g., university, college).
    Click Get Standings to view filtered results in a table.

Rating Analytics


   Navigate to the Analytics page.
   Enter your Codeforces handle (case-sensitive).
   Click Predict My Rating to generate a forecast chart and efficiency metrics.

![image](https://github.com/user-attachments/assets/fa0ea53a-ed09-48a5-9c5f-d0c9fccb83ce)
![image](https://github.com/user-attachments/assets/e19c43d6-6999-49a1-95a1-e12aedfc3b1b)

Prediction Methodology

   Data Collection: Fetches user rating history and contest standings via the Codeforces API.  

 Efficiency Calculation: Derives overall, growth, recent, and peak efficiency metrics (rating gain per contest) to quantify learning rate.
 
  Pattern Classification: Assigns a progression pattern (e.g., high potential, plateauing expert) based on efficiency thresholds.
  
   Threshold Modeling: Estimates an eventual rating ceiling using sigmoid and logarithmic convergence models tailored to the user’s pattern.
 
  Ensemble Forecasting: Combines polynomial regression and Random Forest predictions with convergence constraints to produce realistic future ratings



