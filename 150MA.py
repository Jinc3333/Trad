import pandas as pd
import pyupbit
import time

# Upbit API 접근을 위한 Access Key와 Secret Key 설정
access_key = 'your key'
secret_key = 'your key'
upbit = pyupbit.Upbit(access_key, secret_key)

# 비트코인 시장 코드 설정 (BTC 마켓)
market = "KRW-BTC"

while True:
    # 150 이동 평균선 계산을 위한 데이터 조회
    df = pyupbit.get_ohlcv(market, count=150)

    # 150 이동 평균선 계산
    df['ma150'] = df['close'].rolling(window=150).mean()

    # 최신 데이터로부터 현재 종가를 가져옴
    current_price = pyupbit.get_current_price(market)

    if current_price is not None:
        # 최근 종가가 150 이동 평균선 아래에 있는지 확인
        if current_price < df['ma150'].iloc[-1]:
            # 매도 조건을 충족하면 매도 수행
            krw_balance = upbit.get_balance("KRW")
            sell_amount = krw_balance * 0.99  # 수수료를 고려하여 잔고의 99%를 매도
            upbit.sell_market_order(market, sell_amount)
            print("매도 주문이 전송되었습니다.")
        else:
            # 매수 조건을 충족하면 매수 수행
            krw_balance = upbit.get_balance("KRW")
            buy_amount = krw_balance * 0.99  # 수수료를 고려하여 잔고의 99%를 매수
            upbit.buy_market_order(market, buy_amount)
            print("매수 주문이 전송되었습니다.")
    else:
        print("현재 가격을 가져올 수 없습니다.")

    # 10초 대기
    time.sleep(21600)