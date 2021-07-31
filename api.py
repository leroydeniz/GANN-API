from flask import Flask, jsonify, request, redirect, url_for, json, abort, send_file
from flask_cors import CORS  # Para que se permita la política CORS

from utils import *
from datetime import datetime

from io import BytesIO

app = Flask(__name__)
# Para aumentar el tamaño máximo de mensaje de solicitud
app.config['MAX_CONTENT_LENGTH'] = 35 * 1000 * 1000
CORS(app)  # Aplica la política de CORS sobre esta aplicación

# Definición de las funciones por caso de uso


@app.route('/')
def index():
    return jsonify({'Autor': 'Leroy Deniz',
                    'Nombre': 'GANN',
                    'Descripción': 'Servicio web para entrenamiento y optimización de redes neuronales con algoritmos genéticos y backpropagation'}
                   )


############################################################################################


@app.route("/train", methods=["POST"])
def api_train():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"\n[{current_time}]: Request(train)[{request.form}]")
    if request.method == "POST":

        # Recibir los datos desde el request
        pEmail = request.form.get('email')
        pServicio = request.form.get('servicio')
        pTerminos = request.form.get('terms')
        pTrain = request.files['train']
        pTrain_extension = request.form.get('train_extension')
        pTrain_filename = f"{pTrain.filename}.{pTrain_extension}"

        # verificar que acepta los términos del servicio
        if pTerminos != 'Acepto':
            return jsonify({'Error': 'Términos y condiciones no aceptados'})

        # Verificar que la extensión es csv
        elif not verificar_extension_csv(pTrain_filename):
            return jsonify({'Error': 'Formato no CSV'})

        # Deserializar el archivo recibido desde formulario
        pTrain = pTrain.read()
        print(f" -> Request(train)[{pTrain_filename}: {len(pTrain)}  bytes]")

        # Verificar que el archivo no está vacío
        if verificar_vacio_csv(pTrain):
            return jsonify({'Error': 'Archivo vacío'})

        # Si llega a este punto, está todo correcto
        else:
            try:
                pTrain = pTrain.decode('UTF-8')
                modelo, tst_scores, size_entrada = optimizar(pTrain)
                fichero = exportar_onnx(modelo, size_entrada)
                response = {
                    "file": fichero.decode('latin1'),
                    "error_perc": tst_scores[0],
                    "num_neuronas": tst_scores[1],
                    "num_capas": tst_scores[2],
                    "avg_loss": tst_scores[3]
                    }
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}]: Response(train)[{tst_scores} alongside File[{len(fichero)} bytes]]\n")
                return response
            except Exception as e:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}]: Exception(train)[{str(e)}]\n")
                return {"Error": str(e)}


############################################################################################


@app.route("/test", methods=["POST"], endpoint="test")
def api_test():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"\n[{current_time}]: Request(test)[{request.form}]")
    if request.method == "POST":
        # Recibir los datos desde el request
        pEmail = request.form.get('email')
        pServicio = request.form.get('servicio')
        pTerminos = request.form.get('terms')
        pTrain = request.files['train']
        pTest = request.files['test']
        pTrain_extension = request.form.get('train_extension')
        pTest_extension = request.form.get('test_extension')
        pTrain_filename = f"{pTrain.filename}.{pTrain_extension}"
        pTest_filename = f"{pTest.filename}.{pTest_extension}"

        # verificar que acepta los términos del servicio
        if pTerminos != 'Acepto':
            return jsonify({'Error': 'Términos y condiciones no aceptados'})

        # Verificar que la extensión es csv u onnx
        elif not verificar_extension_csv(pTrain_filename) and not verificar_extension_onnx(pTrain_filename):
            return jsonify({'Error': 'Modelo o conjunto de datos - Formato no CSV ni ONNX'})

        # Verificar que la extensión es csv
        elif not verificar_extension_csv(pTest_filename):
            return jsonify({'Error': 'Conjunto a clasificar - Formato no CSV'})

        pTrain = pTrain.read()
        pTest = pTest.read()
        print(f" -> Request(test)[{pTrain_filename}: {len(pTrain)} bytes]")
        print(f" -> Request(test)[{pTest_filename}: {len(pTest)} bytes]")

        # Verificar que el archivo no está vacío
        if verificar_vacio_csv(pTrain):
            return jsonify({'Error': 'Conjunto a clasificar - Archivo vacío'})

        if verificar_extension_csv(pTrain_filename) and verificar_vacio_csv(pTrain):
            return jsonify({'Error': 'Modelo o conjunto de datos - Archivo vacío'})

        # Si llega a este punto, está todo correcto
        else:
            try:
                pTrain = pTrain.decode('UTF-8')
                pTest = pTest.decode('UTF-8')
                if verificar_extension_csv(pTrain_filename):
                    # Aquí tiene que entrenar
                    modelo, _, size_entrada = optimizar(pTrain)
                    modelo = importar_onnx(exportar_onnx(modelo, size_entrada))
                else:
                    # Aquí sólo tiene que probar (detecta ONNX)
                    pTrain = pTrain.encode('latin1')
                    modelo = importar_onnx(pTrain)

                response = evaluar(modelo, pTest)
                
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}]: Response(test)[{response}]\n")
                return response
            except Exception as e:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}]: Exception(test)[{str(e)}]\n")
                return {"Error": str(e)}


############################################################################################


@app.route("/classify", methods=["POST"], endpoint="classify")
def api_classify():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"\n[{current_time}]: Request(classify)[{request.form}, {request.files}]")
    if request.method == "POST":
        # Recibir los datos desde el request
        pEmail = request.form.get('email')
        pServicio = request.form.get('servicio')
        pTerminos = request.form.get('terms')
        pTrain = request.files['train']
        pTest = request.files['test']
        pTrain_extension = request.form.get('train_extension')
        pTest_extension = request.form.get('test_extension')
        pTrain_filename = f"{pTrain.filename}.{pTrain_extension}"
        pTest_filename = f"{pTest.filename}.{pTest_extension}"

        # verificar que acepta los términos del servicio
        if pTerminos != 'Acepto':
            return jsonify({'Error': 'Términos y condiciones no aceptados'})

        # Verificar que la extensión es csv u onnx
        elif not verificar_extension_csv(pTrain_filename) and not verificar_extension_onnx(pTrain_filename):
            return jsonify({'Error': 'Modelo o conjunto de datos - Formato no CSV ni ONNX'})

        # Verificar que la extensión es csv
        elif not verificar_extension_csv(pTest_filename):
            return jsonify({'Error': 'Conjunto a clasificar - Formato no CSV'})

        pTrain = pTrain.read()
        pTest = pTest.read()
        print(f" -> Request(classify)[{pTrain_filename}: {len(pTrain)} bytes]")
        print(f" -> Request(classify)[{pTest_filename}: {len(pTest)} bytes]")

        # Verificar que el archivo no está vacío
        if verificar_vacio_csv(pTest):
            return jsonify({'Error': 'Conjunto a clasificar - Archivo vacío'})

        if verificar_extension_csv(pTrain_filename) and verificar_vacio_csv(pTrain):
            return jsonify({'Error': 'Modelo o conjunto de datos - Archivo vacío'})

        # Si llega a este punto, está todo correcto
        else:
            try:
                pTrain = pTrain.decode('UTF-8')
                pTest = pTest.decode('UTF-8')
                if verificar_extension_csv(pTrain_filename):
                    # Aquí tiene que entrenar
                    modelo, _, size_entrada = optimizar(pTrain)
                    modelo = importar_onnx(exportar_onnx(modelo, size_entrada))
                else:
                    # Aquí sólo tiene que probar (detecta ONNX)
                    pTrain = pTrain.encode('latin1')
                    modelo = importar_onnx(pTrain)
                pred = clasificar(modelo, pTest, return_csv=True)
                response = {"file" : pred}
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}]: Response(classify)[{response}]\n")
                return response
            except Exception as e:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}]: Exception(classify)[{str(e)}]\n")
                return {"Error": str(e)}
