from flask import Flask

application = Flask(__name__)

@application.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

@application.route('/')
def home():
    return {'message': 'Fire Detection Dashboard Health Check'}, 200

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080)