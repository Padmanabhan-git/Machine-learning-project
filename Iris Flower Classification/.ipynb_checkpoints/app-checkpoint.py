from flask import Flask, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('SVM.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form (user input)
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Use the loaded SVM model to make a prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        # Iris class mapping (depending on how you trained your model)
        iris_classes = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
        result = prediction[0]

        # Validation check to prevent negative values
        if any(value < 0 for value in [sepal_length, sepal_width, petal_length, petal_width]):
             return render_template('result.html', prediction="Error: Negative values are not allowed.",show_image=False)

        if any(value == 0 for value in [sepal_length, sepal_width, petal_length, petal_width]):
             return render_template('result.html', prediction="Zeros are not allowed.",show_image=False)

        # Image paths corresponding to the Iris classes
        image_paths = {
            'Iris-setosa': '/static/images/iris_setosa.jpg',
            'Iris-versicolor': '/static/images/iris_versicolor.jpg',
            'Iris-virginica': '/static/images/iris_virginica.jpg'
        }
        image_path = image_paths[result]
        
        # Return prediction result to be displayed on the webpage
        return render_template('result.html',prediction=result,image_path=image_path,show_image=True)
    except Exception as e:
        return f"Error occured {e} here is error"

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except SystemExit:
        pass