from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['has_profile_pic']),
        int(request.form['username_length']),
        int(request.form['num_digits_in_username']),
        int(request.form['bio_length']),
        float(request.form['follower_following_ratio'])
    ]
    prediction = model.predict([features])[0]
    result = "Likely Fake" if prediction == 1 else "Likely Genuine"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)