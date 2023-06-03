from flask import Flask, Blueprint, render_template, request, redirect, url_for
from bs4 import BeautifulSoup
import subprocess
import sys
from website.static.pokemonmodel.activate_model import generate_image

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

@auth.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        checkbox_values = request.form.get('checkbox_name')
        return render_template('pokemongeneration.html', newVal=checkbox_values)
    return render_template('create.html')

@auth.route('/pokemongeneration', methods=['GET','POST'])
def pokemongeneration():
    chosenVal = request.args.get('newVal')
    return render_template('pokemongeneration.html', values=3)

@auth.route('/bug')
def bug():
    generate_image(0)
    return render_template("bug.html")

@auth.route('/dark')
def dark():
    generate_image(1)
    return render_template("dark.html")

@auth.route('/dragon')
def dragon():
    generate_image(2)
    return render_template("dragon.html")

@auth.route('/electric')
def electric():
    generate_image(3)
    return render_template("electric.html")

@auth.route('/fairy')
def fairy():
    generate_image(4)
    return render_template("fairy.html")

@auth.route('/fighting')
def fighting():
    generate_image(5)
    return render_template("fighting.html")

@auth.route('/fire')
def fire():
    generate_image(6)
    return render_template("fire.html")

@auth.route('/flying')
def flying():
    generate_image(7)
    return render_template("flying.html")

@auth.route('/ghost')
def grass():
    generate_image(8)
    return render_template("ghost.html")

@auth.route('/grass')
def ghost():
    generate_image(9)
    return render_template("grass.html")

@auth.route('/ground')
def ground():
    generate_image(10)
    return render_template("ground.html")

@auth.route('/ice')
def ice():
    generate_image(11)
    return render_template("ice.html")

@auth.route('/normal')
def normal():
    generate_image(12)
    return render_template("normal.html")

@auth.route('/poison')
def poison():
    generate_image(13)
    return render_template("poison.html")

@auth.route('/psychic')
def psychic():
    generate_image(14)
    return render_template("psychic.html")

@auth.route('/rock')
def rock():
    generate_image(15)
    return render_template("rock.html")

@auth.route('/steel')
def steel():
    generate_image(16)
    return render_template("steel.html")

@auth.route('/water')
def water():
    generate_image(17)
    return render_template("water.html")


