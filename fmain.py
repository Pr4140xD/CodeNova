from flask import Flask, request, render_template
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
from io import BytesIO
import datetime

app = Flask(__name__)

def string_matching(a, b):
    return b in a

def get_data(api):
    response = requests.get(api)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def contest_data(url):
    return get_data(url)

def updated_standings(contest_id, inp):
    contest_url = f"https://codeforces.com/api/contest.standings?contestId={contest_id}&from=1&count=9000&showUnofficial=false"
    contest_standings = contest_data(contest_url)
    if contest_standings is None:
        return None

    contestant = contest_standings['result']['rows']
    no_of_contestant = len(contestant)
    st = 0
    cont_lis = []

    while st < no_of_contestant:
        handle_url = "https://codeforces.com/api/user.info?handles="
        handles = []
        for i in range(st, min(st + 700, no_of_contestant)):
            handle = contestant[i]['party']['members'][0]['handle']
            handles.append(handle)
        handle_url += ';'.join(handles)

        cont_data = get_data(handle_url)
        if cont_data is None:
            break

        for i, cont_org in enumerate(cont_data['result']):
            if cont_org.get('organization') is None:
                continue

            cont_org1 = cont_org['organization'].lower().replace(" ", "")
            for j in range(len(inp)):
                if string_matching(cont_org1, inp[j]):
                    cont_lis.append(contestant[st + i])
                    break

        st += 700

    return cont_lis

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        contest_id = request.form['contest_id']
        organizations = request.form['organizations'].split(',')
        organizations = [org.lower().replace(" ", "") for org in organizations]
        standings = updated_standings(contest_id, organizations)
        if standings:
            return render_template('results.html', standings=standings)
        else:
            return "Error fetching standings or no results."
    return render_template('index.html')

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    if request.method == 'POST':
        handle = request.form['handle']
        url = f"https://codeforces.com/api/user.rating?handle={handle}"
        data = get_data(url)
        if not data or data['status'] != 'OK':
            return f"Could not fetch rating data for {handle}."

        result = data['result']
        if len(result) < 3:
            return "Not enough contests to analyze."

        dates = [datetime.datetime.fromtimestamp(x['ratingUpdateTimeSeconds']) for x in result]
        ratings = [x['newRating'] for x in result]

        x_vals = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
        y_vals = np.array(ratings)

        poly = PolynomialFeatures(degree=3)
        x_poly = poly.fit_transform(x_vals)
        model = LinearRegression()
        model.fit(x_poly, y_vals)

        future_days = np.array([x_vals[-1][0] + i for i in range(1, 121)]).reshape(-1, 1)
        future_x_poly = poly.transform(future_days)
        future_ratings = model.predict(future_x_poly)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(dates, ratings, marker='o', label='Actual Rating')
        future_dates = [dates[-1] + datetime.timedelta(days=i) for i in range(1, 121)]
        plt.plot(future_dates, future_ratings, '--', label='Predicted Rating (Poly Regression)')
        plt.xlabel('Date')
        plt.ylabel('Rating')
        plt.title(f'Codeforces Rating Trend for {handle}')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return render_template('analytics.html', img_data=img_base64, handle=handle)

    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True)
