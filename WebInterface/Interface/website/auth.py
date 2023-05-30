from flask import Blueprint, render_template

auth = Blueprint('auth', __name__)

@auth.route('/home')
def home():
    return render_template("home.html")

@auth.route('/npr')
def npr():
    return render_template("npr.html")

@auth.route('/designsdownhill')
def designsdownhill():
    return render_template("designsdownhill.html")

@auth.route('/create')
def create():
    return render_template("create.html")
