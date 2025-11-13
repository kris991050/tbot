from flask import Flask, request, abort
import ngrok, json, subprocess, datetime
from ib_insync import *
from sanic import Sanic
# sys.path.append("/Volumes/untitled/Trading/trading_app/scripts.path_setup")
# import path_setup

# Related video wit source code: https://www.youtube.com/watch?v=HQLRPWi2SeA



# Also start "ngrok http 8080"
# voir https://github.com/EconLQ/TWS-orders-placement-via-Tradinview-webhooks/blob/master/TradingViewInteractiveBrokers/app.py


app = Flask(__name__)
# app = Sanic(__name__)

# Create root to easily let us know its on/working.
# @app.route('/')
# async def root(request):
#     print(request.json)
#     return request.json


@app.route('/webhook', methods=['POST', 'GET'])
# async def webhook(request):
def webhook():
    if request.method == 'POST':

        webhook = request.json
        data = json.loads(webhook['data'].replace("'", "\""))

        print(f"POST REUEST RECEIVED @ {datetime.datetime.now()}")
        print(f"Data:  {str(data)}")
        # subprocess.run(["python3", "orders.py", str(data)])

        return webhook
    else:
        print("Only POST method supported")
        print(request)

        # print("request = ", request)

        return "Hello World! GET"

# @app.route('/webhook', methods=['POST'])
# def webhook():
#     if request.method == 'POST':
#         print(request.json)
#         return 'success', 200
#     else:
#         abort(400)


# ib = IB()
# ib.connect('127.0.0.1', 7497, clientId=1)


if __name__ == '__main__':

    # listener = ngrok.forward("localhost:"+str(port), authtoken="2iebQBJi1ETtq0SS4Dux4BtgHUC_ZEtXxSrhTkin7YPs8TF8")#authtoken_from_env=True)
    app.run(debug=True, port=8080)
    # app.run(debug=True, port=5000)

    print("\n\n")







# {
#     "ticker": "{{ticker}}",
#     "currency": "{{syminfo.currency}}",
#     "action": "{{strategy.order.action}}",
#     "direction": "{{strategy.market_position}}",
#     "quantity": "{{strategy.order.contracts}}",
#     "price": "{{close}}",
#     "timeframe": "{{interval}}",
#     "data": "{{strategy.order.alert_message}}"
# }
