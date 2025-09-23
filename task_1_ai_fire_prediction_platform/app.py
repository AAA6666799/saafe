from flask import Flask
import os

# Create the Flask application
application = Flask(__name__)

@application.route('/')
def hello():
    return "Hello, Saafe Fire Detection Dashboard!"

@application.route('/health')
def health():
    return {"status": "healthy"}, 200

# This is the WSGI entry point that Elastic Beanstalk will use
if __name__ == '__main__':
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run the application
    application.run(host='0.0.0.0', port=port)