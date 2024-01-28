

'''
@app.route('/')
def index():
    return render_template('index.html')  # Renders an HTML file located in the 'templates' folder


@app.route('/process', methods=['POST'])
def process_data():
    # Get data from the request
    data = request.json['data']
    processed_data = stuff.main1(data)
    # Return processed data to the frontend
    return render_template('new.html',test=data)

if __name__ == '__main__':
    app.run(debug=True)



'''

from flask import Flask, render_template, request, redirect, url_for
import stuff
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        # Retrieve data from form
        user_input = request.form['user_input']
        user_input = str(user_input)
        # Redirect to the processing page and pass the user input
        return redirect(url_for('process_page', user_input=user_input))
    return render_template('input_page.html')

@app.route('/process/<user_input>')
def process_page(user_input):
    # Process the user input (example function)
    if stuff.main1(user_input) == 0:
        processed_data = "Chances are your stock's price will drop in the coming days"
    else:
        processed_data = "Invest now! Your stock is predicted to be on the rise"
    #processed_data = stuff.main1(user_input)  # Example processing


    return render_template('process_page.html', processed_data=processed_data)

if __name__ == '__main__':
    app.run(debug=True)


