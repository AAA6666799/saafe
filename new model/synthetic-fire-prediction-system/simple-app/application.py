import os
from flask import Flask

# Create Flask app
application = Flask(__name__)

@application.route('/')
def hello():
    return "Hello from Flask on Elastic Beanstalk!"

@application.route('/health')
def health():
    return "OK", 200

# For local development
if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 5001
    port = int(os.environ.get('PORT', 5001))
    application.run(host='0.0.0.0', port=port, debug=True)