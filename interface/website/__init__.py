from flask import Flask

def run_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Gotta catch em all'

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app

