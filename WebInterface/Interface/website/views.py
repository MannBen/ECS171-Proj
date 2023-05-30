from flask import Blueprint, render_template

views = Blueprint('views', __name__)

@views.route('/')

def home():
    condition = True
    return render_template("home.html")