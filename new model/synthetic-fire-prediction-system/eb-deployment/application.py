from flask import Flask

# Create a Flask application
application = Flask(__name__)

@application.route('/')
def home():
    return "<h1>Fire Detection Dashboard</h1><p>Application is deployed successfully!</p>"

@application.route('/health')
def health():
    return {"status": "healthy", "message": "Application is running"}

# This is the main entry point for Elastic Beanstalk
if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8080)