import sys
sys.path.append("./FinRL-Library")

import pandas as pd
import numpy as np
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from datetime import datetime,timedelta


if __name__=='__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
 
    import os
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
 
    #Prepareing stock data
    # Download and save the data in a pandas DataFrame:
    col_names=['open','high','low','close']
    data_df = pd.read_csv(args.training,names=col_names)
    data_df['tic']='IBM'
    base=datetime.strptime(config.START_DATE,"%Y-%m-%d")
    date=[base + timedelta(days=x)for x in range(len(data_df))]
    data_df['date']=date

    ## we store the stockstats technical indicator column names in config.py
    tech_indicator_list=['macd','macds','macdh','kdjk','kdjd','close_5_sma','close_10_sma','close_20_sma','close_60_sma']  
    print(tech_indicator_list)

    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = tech_indicator_list,
                    use_turbulence=False,
                    user_defined_feature = False)

    data_df = fe.preprocess_data(data_df)

    #Spliting training and testing data
    train = data_df

    #change stock dimension when more than one stock for trading
    stock_dimension = 1
    state_space = 1 + 2*stock_dimension + len(tech_indicator_list)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 1, 
        "initial_amount": 100000, 
        "buy_cost_pct": 0, 
        "sell_cost_pct": 0, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": tech_indicator_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-5
    }
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env = env_train)

    #A2C
    '''
    A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
    model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
    trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=50000)
    '''
    #PPO
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=60000)
    
    #Trading
    ## make a prediction and get the account value change
    trade = pd.read_csv(args.testing,names=col_names)
    trade['tic']='IBM'
    base=datetime.strptime(config.START_TRADE_DATE,"%Y-%m-%d")
    date=[base + timedelta(days=x)for x in range(len(trade))]
    trade['date']=date
    trade=fe.preprocess_data(trade)

    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)

    actions=None
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ppo,environment = e_trade_gym)
    actions=pd.DataFrame(np.array(df_actions['actions'],dtype='int'))
    actions.to_csv(args.output,index=False,header=False)
