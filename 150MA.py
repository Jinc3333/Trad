import pandas as pd
import pyupbit
import time

access_key = 'your key'
secret_key = 'your key'
upbit = pyupbit.Upbit(access_key, secret_key)

market = "KRW-BTC"

while True:
    df = pyupbit.get_ohlcv(market, count=150)

    df['ma150'] = df['close'].rolling(window=150).mean()

    current_price = pyupbit.get_current_price(market)

    if current_price is not None:

        if current_price < df['ma150'].iloc[-1]:
   
            krw_balance = upbit.get_balance("KRW")
            sell_amount = krw_balance * 0.995  
            upbit.sell_market_order(market, sell_amount)
            print("매도 주문이 전송되었습니다.")
        else:
    
            krw_balance = upbit.get_balance("KRW")
            buy_amount = krw_balance * 0.995  
            upbit.buy_market_order(market, buy_amount)
            print("매수 주문이 전송되었습니다.")
    else:
        print("현재 가격을 가져올 수 없습니다.")

   
    time.sleep(21600)