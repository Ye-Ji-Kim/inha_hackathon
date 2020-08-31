# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:28:21 2020

@author: kyeoj
"""

from flask import Flask, render_template, jsonify, request
import serial

app = Flask(__name__)
server = "http://127.0.0.1:5000/"

PORT = 'COM3'
BaudRate = 9600

ARD=serial.Serial(PORT, BaudRate)

@app.route('/arduino')
def register():
    result = request.args.get('result')
    print(result)
    result = result.encode('utf-8')
    ARD.write(result)
    
    return 'success!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)