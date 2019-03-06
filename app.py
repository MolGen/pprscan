import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from keras.models import load_model

from src.utils import parse_fasta, scan_protein

MAXLEN = 35
ALPHABET = "ACDEFGHIKLMNPQRSTVWY*"
CLASSES = np.array(['P1', 'L1', 'S1', 'P2', 'L2', 'S2',
                    'E1', 'E2', 'SS', 'P', 'TPR', 'B'])
WEIGHT = "model/mlp_PPR_n10_01w5_rnd3M_B146K_EV4.h5"

model = load_model(WEIGHT)
model._make_predict_function()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logo')
def logo():
    return render_template('logo.html')


@app.route('/anno', methods=['POST'])
def anno():
    raw = request.form['fasta']
    entries = pd.DataFrame()
    for accession, sequence in parse_fasta(raw.splitlines()):
        entries_temp = scan_protein(
            accession, sequence, model, classes=CLASSES, w=35, bg="B", flatten=True)
        entries = entries.append(entries_temp, ignore_index=False)
    entries = entries.to_html(
        index=False, classes='display table table-striped table-hover" id="annotable', border=1)

    return render_template('tables.html', entries=entries)


@app.route('/curated')
def curated():
    cureated_entries = pd.read_csv(
        "datasets/Ath447_curated_Wang2019.bed", sep="\t", header=0)
    cureated_entries = cureated_entries.to_html(
        index=False, classes='display table table-striped table-hover" id="annotable', border=1)

    return render_template('tables.html', entries=cureated_entries)


if __name__ == '__main__':
    app.run()
