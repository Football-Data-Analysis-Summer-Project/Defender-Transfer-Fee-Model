from flask import Flask,render_template,request
import pandas as pd 
import pickle 

app = Flask(__name__)

with open("defender.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

@app.route('/',methods=['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            club_level = int(request.form['club_level'])
            minutes_played = int(request.form['minutes_played'])
            goals = int(request.form['goals'])
            tackles = int(request.form['tackles'])
            tackles_won = int(request.form['tackles_won'])
            challenges_lost = int(request.form['challenges_lost'])
            blocks = int(request.form['blocks'])
            interceptions = int(request.form['interceptions'])
            errors = int(request.form['errors'])
            games_missed = int(request.form['games_missed'])

            newPlayer = pd.DataFrame([{
                'Age': age,
                'Club Level': club_level,
                'Minutes played': minutes_played,
                'Goals': goals,
                'Tackles': tackles,
                'Tackles Won': tackles_won,
                'Challenges Lost': challenges_lost,
                'Blocks': blocks,
                'Interceptions': interceptions,
                'Errors': errors,
                'Games Missed': games_missed
            }])

            scaled_input = scaler.transform(newPlayer)
            predicted_value = model.predict(scaled_input)[0]
            prediction = f"{predicted_value:.2f} million euros"
        except Exception as e:
            prediction = f"Error : {str(e)}"
    return render_template('index.html',prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True,port = 8000)