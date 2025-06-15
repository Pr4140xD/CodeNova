from flask import Flask, request, render_template
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import base64
from io import BytesIO
import datetime
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class RatingPredictor:
    def __init__(self):
        self.models = {
            'polynomial': PolynomialFeatures(degree=3),
            'forest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
    
    def detect_pattern(self, ratings):
        """Simplified pattern detection"""
        x = np.arange(len(ratings))
        
        # Test different patterns
        patterns = {}
        
        # Linear trend
        slope = np.polyfit(x, ratings, 1)[0]
        patterns['linear'] = abs(slope)
        
        # Exponential-like growth
        if len(ratings) > 5:
            recent_growth = np.mean(np.diff(ratings[-5:]))
            early_growth = np.mean(np.diff(ratings[:5])) if len(ratings) > 5 else 0
            patterns['exponential'] = max(0, recent_growth - early_growth)
        
        # Determine best pattern
        if patterns.get('exponential', 0) > patterns.get('linear', 0) * 2:
            return 'exponential'
        elif slope > 10:
            return 'linear_growth'
        elif slope < -10:
            return 'declining'
        else:
            return 'stable'
    
    def predict_ratings(self, ratings, dates, future_contests=20):
        """Simplified prediction using ensemble of polynomial and ML"""
        if len(ratings) < 5:
            return None, None, 0.5
        
        x = np.arange(len(ratings)).reshape(-1, 1)
        y = np.array(ratings)
        
        # Polynomial prediction
        poly_features = PolynomialFeatures(degree=min(3, len(ratings)//2))
        x_poly = poly_features.fit_transform(x)
        poly_model = LinearRegression().fit(x_poly, y)
        
        # Random Forest prediction 
        features = []
        for i in range(len(ratings)):
            feature = [
                ratings[i],  # Current rating
                i,  # Contest number
                np.mean(ratings[max(0, i-3):i+1]),  # Recent average
                ratings[i] - ratings[max(0, i-1)] if i > 0 else 0  # Recent change
            ]
            features.append(feature)
        
        if len(features) > 5:
            X_ml = np.array(features[:-1])
            y_ml = np.array(ratings[1:])
            rf_model = RandomForestRegressor(n_estimators=30, random_state=42)
            rf_model.fit(X_ml, y_ml)
        else:
            rf_model = None
        
        # Generate predictions
        future_x = np.arange(len(ratings), len(ratings) + future_contests).reshape(-1, 1)
        
        # Polynomial predictions
        future_x_poly = poly_features.transform(future_x)
        poly_pred = poly_model.predict(future_x_poly)
        
        # ML predictions
        if rf_model:
            ml_predictions = []
            last_features = features[-1].copy()
            
            for i in range(future_contests):
                pred_rating = rf_model.predict([last_features])[0]
                ml_predictions.append(pred_rating)
                
                # Update features for next prediction
                last_features = [
                    pred_rating,
                    len(ratings) + i,
                    np.mean([last_features[0], pred_rating]),
                    pred_rating - last_features[0]
                ]
            
            ml_pred = np.array(ml_predictions)
            # Combine predictions
            ensemble_pred = 0.7 * poly_pred + 0.3 * ml_pred
        else:
            ensemble_pred = poly_pred
        
        # Pattern detection
        pattern = self.detect_pattern(ratings)
        
        # Confidence based on recent stability
        recent_std = np.std(ratings[-min(10, len(ratings)):])
        confidence = max(0.3, min(0.9, 1.0 - recent_std / 200))
        
        return ensemble_pred, pattern, confidence

# Initialize predictor
predictor = RatingPredictor()

def get_data(api):
    """Fetch data from API"""
    try:
        response = requests.get(api, timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def string_matching(a, b):
    """Check if b is contained in a"""
    return b in a

def updated_standings(contest_id, inp):
    """Get filtered contest standings based on organizations"""
    contest_url = f"https://codeforces.com/api/contest.standings?contestId={contest_id}&from=1&count=9000&showUnofficial=false"
    contest_standings = get_data(contest_url)
    if not contest_standings:
        return None

    contestants = contest_standings['result']['rows']
    filtered_contestants = []
    
    # Process in batches to avoid API limits
    for i in range(0, len(contestants), 700):
        batch = contestants[i:i+700]
        handles = [c['party']['members'][0]['handle'] for c in batch]
        handle_url = f"https://codeforces.com/api/user.info?handles={';'.join(handles)}"
        
        user_data = get_data(handle_url)
        if not user_data:
            continue
            
        for j, user in enumerate(user_data['result']):
            if user.get('organization'):
                org = user['organization'].lower().replace(" ", "")
                for search_org in inp:
                    if string_matching(org, search_org):
                        filtered_contestants.append(batch[j])
                        break
    
    return filtered_contestants

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page for contest standings"""
    if request.method == 'POST':
        contest_id = request.form['contest_id']
        organizations = [org.lower().replace(" ", "") for org in request.form['organizations'].split(',')]
        standings = updated_standings(contest_id, organizations)
        
        if standings:
            return render_template('results.html', standings=standings)
        else:
            return render_template('index.html', error="Error fetching standings or no results found.")
    
    return render_template('index.html')

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    """Analytics page for rating prediction"""
    if request.method == 'POST':
        handle = request.form['handle'].strip()
        
        if not handle:
            return render_template('analytics.html', error="Please enter a valid handle.")
        
        # Fetch user rating data
        data = get_data(f"https://codeforces.com/api/user.rating?handle={handle}")
        
        if not data or data.get('status') != 'OK':
            return render_template('analytics.html', error=f"Could not fetch rating data for '{handle}'. Please check the handle.")

        result = data['result']
        if len(result) < 5:
            return render_template('analytics.html', error="Not enough contests to analyze. Minimum 5 contests required.")

        # Process data
        dates = [datetime.datetime.fromtimestamp(x['ratingUpdateTimeSeconds']) for x in result]
        ratings = [x['newRating'] for x in result]

        # Get predictions
        predictions, pattern, confidence = predictor.predict_ratings(ratings, dates, 20)
        
        if predictions is None:
            return render_template('analytics.html', error="Unable to generate predictions for this user.")

        
        future_dates = [dates[-1] + datetime.timedelta(days=7*i) for i in range(1, len(predictions) + 1)]

        
        plt.figure(figsize=(14, 8))
        
        
        plt.plot(dates, ratings, 'o-', linewidth=2, markersize=5, label='Actual Rating', color='#1f77b4')
        
        
        plt.plot(future_dates, predictions, '--', linewidth=2, 
                label=f'Predicted ({pattern})', color='#ff7f0e')
        
        
        margin = np.std(ratings[-min(5, len(ratings)):]) * (1 - confidence)
        plt.fill_between(future_dates, predictions - margin, predictions + margin, 
                        alpha=0.2, color='#ff7f0e', label=f'Confidence ({confidence:.0%})')
        
        
        categories = [
            (1200, 'Pupil', '#008000'),
            (1400, 'Specialist', '#03A89E'), 
            (1600, 'Expert', '#0000FF'),
            (1800, 'Candidate Master', '#AA00AA'), 
            (2100, 'Master', '#FF8C00'),
            (2400, 'Grandmaster', '#FF0000')
        ]
        
        y_min, y_max = plt.ylim()
        for threshold, title, color in categories:
            if y_min < threshold < y_max:
                plt.axhline(y=threshold, color=color, linestyle=':', alpha=0.6, linewidth=1)
                plt.text(dates[0], threshold + 20, title, fontsize=8, color=color, alpha=0.8)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.title(f'Codeforces Rating Analysis: {handle}\nPattern: {pattern.replace("_", " ").title()}, Confidence: {confidence:.0%}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        
        current_rating = ratings[-1]
        predicted_peak = max(predictions)
        potential_gain = predicted_peak - current_rating
        avg_contest_gain = np.mean(np.diff(ratings[-10:])) if len(ratings) > 10 else 0
        
        stats_text = f"Current Rating: {current_rating}\n"
        stats_text += f"Predicted Peak: {predicted_peak:.0f}\n"
        stats_text += f"Potential Gain: {potential_gain:+.0f}\n"
        stats_text += f"Avg. Recent Gain: {avg_contest_gain:+.1f}/contest\n"
        stats_text += f"Total Contests: {len(ratings)}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=10, fontfamily='monospace')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return render_template('analytics.html', 
                             img_data=img_base64, 
                             handle=handle,
                             pattern_type=pattern.replace("_", " ").title(),
                             confidence=f"{confidence:.0%}",
                             current_rating=current_rating,
                             predicted_peak=f"{predicted_peak:.0f}",
                             potential_gain=f"{potential_gain:+.0f}",
                             total_contests=len(ratings),
                             avg_gain=f"{avg_contest_gain:+.1f}")

    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
