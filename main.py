import MetaTrader5 as mt5

# Connect to MT5
if not mt5.initialize():
    print("MT5 initialization failed, moron")
    quit()

def ai_fundamental_recommendation():
    # Your Sonar/AI NLP function here—stubbed out, dummy!
    # Let's mock: returns ('SELL', 90)
    return 'SELL', 90

def technical_analysis(symbol, timeframe=mt5.TIMEFRAME_M15):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    # Example: If close < moving average, return SELL
    closes = [x['close'] for x in rates]
    ma = sum(closes[-20:]) / 20
    if closes[-1] < ma:
        return 'SELL'
    elif closes[-1] > ma:
        return 'BUY'
    else:
        return 'WAIT'

def consensus_decision(ai_rec, ai_conf, tech_rec):
    if ai_conf < 60:
        return 'WAIT'
    if ai_rec == tech_rec and ai_rec in ['SELL', 'BUY']:
        return ai_rec
    else:
        return 'WAIT'
        
def open_trade(symbol, action='SELL', lot=0.1):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL if action == 'SELL' else mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(symbol).bid if action == 'SELL' else mt5.symbol_info_tick(symbol).ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "IQ 1 Million trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print("Trade sent:", result)

# --- Example Day Trading Bot Run ---
symbol = "EURUSD"
ai_rec, ai_conf = ai_fundamental_recommendation()
tech_rec = technical_analysis(symbol)
final_action = consensus_decision(ai_rec, ai_conf, tech_rec)

print(f"Final consensus: {final_action}")
if final_action in ['SELL', 'BUY']:
    open_trade(symbol, final_action)
else:
    print("No consensus, you lazy potato! No trade today.")

# Always shutdown!
mt5.shutdown()
