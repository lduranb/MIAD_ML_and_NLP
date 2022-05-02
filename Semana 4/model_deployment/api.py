#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from m09_model_deployment import predict_price


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Car Price Predictor API',
    description='Car Price Predictor API')

ns = api.namespace('predict', 
     description='Price Predictor')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='Data to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


# Definición de la clase para disponibilización
@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price(args['URL'])
        }, 200
  
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
