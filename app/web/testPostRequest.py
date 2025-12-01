import requests, json
# sys.path.append("/Volumes/untitled/Trading/trading_app/scripts.path_setup")
# import path_setup




if __name__ == '__main__':

    # api-endpoint
    url = "http://127.0.0.1:8080/webhook"
    url = "https://b578bd55fcd0.ngrok-free.app/webhook"
    # url = "https://2eb8-149-22-81-21.ngrok-free.app/webhook"
    # url = "https://cd7e-149-22-81-24.ngrok-free.app/webhook"
    # url = "https://webhook.site/9831a0f8-3346-4534-91ed-b0d08e3b366d"
    # url = "https://bc940c06c092.ngrok-free.app/webhook"
    # url = "http://localhost:5000"

    payload = {
        "ticker": "TSLA",
        "currency": "USD",
        "action": "BUY",
        "direction": "long",
        "quantity": "10",
        "price": "129.0",
        "timeframe": "1M",
        "data": {"PT": "140.0", "SL": "125.0"}}

    payload = {'data': "{'ticker': 'TSLA', 'currency': 'USD', 'action': 'BUY',  'quantity': '10', 'price': '1.36349', 'timeframe': '1M', 'profit_target': '1.4998587627', 'stop_loss': '1.0226309746'}"}
    payload = {'data': "{'ticker': 'QQQ', 'currency': 'USD', 'action': 'SELL',  'quantity': '8', 'price': '', 'timeframe': '1M', 'profit_target': '485', 'stop_loss': '501'}"}
    # payload = {'data': "{'ticker': 'SPY', 'currency': 'USD', 'action': 'SELL',  'quantity': '10', 'price': '', 'timeframe': '1M', 'profit_target': '545', 'stop_loss': '560'}"}
    # payload = {'data': "{'ticker': 'FX:USDCAD', 'currency': 'CAD', 'action': 'BUY', 'quantity': '12', 'price': '1.36349', 'timeframe': '10S', 'profit_target': '1.36202', 'stop_loss': '1.35930'}"}
    # payload = {"data": "{\"ticker\": \"FX:USDCAD\", \"currency\": \"CAD\", \"action\": \"buy\",  \"price\": \"1.36349\", \"timeframe\": \"10S\", \"profit_ttarget\": \"1.4998587627\", \"stop_loss\": \"1.0226309746\"}"}
    # payload = {"userID": "1", "id": "1"}

    # sending get request and saving the response as response object
    r = requests.post(url=url, json=payload, headers={"Content-Type": "application/json"})

    # extracting data in json format
    # data = r.json()
    # print("Post request: ", r)







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
