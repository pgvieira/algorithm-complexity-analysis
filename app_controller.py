import os

import pydevd_pycharm
from flask import request, jsonify, Flask
from werkzeug.utils import secure_filename

from app_service import AppService

"pydevd_pycharm.settrace('127.0.0.1', port=5001, stdoutToServer=True, stderrToServer=True)"

app_service = AppService()

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'java'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/calculatecomplexity', methods=['POST'])
def calculate_complexity():
    analysis_type = request.form.get('type')

    if analysis_type is None or analysis_type == '':
        resp = jsonify({'mensagem': 'Não foi selecionada nenhum tipo de análise!'})
        resp.status_code = 400
        return resp

    if 'file' not in request.files:
        resp = jsonify({'mensagem': 'Não foi selecionado nenhum arquivo!'})
        resp.status_code = 400
        return resp

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        complexidade = app_service.calculate_complexity(analysis_type, filename)
        resp = jsonify({'mensagem': 'Complexidade calculada com sucesso!', 'complexidade': complexidade})
        resp.status_code = 200
        return resp
    else:
        resp = jsonify({'mensagem': 'Apenas arquivos do tipo Java são permitidos!'})
        resp.status_code = 400
        return resp


app.run()
