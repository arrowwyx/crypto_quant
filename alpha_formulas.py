"""
研报里没有指定的话，就取过去三天的数据做聚合
data字典格式储存到本地，包含所有字段2023.7的数据
1.last_close少一根线，用stock_close_5min?
2.数据量纲是否要做调整？
参数自由发挥一下？
"""

import pandas as pd
import pickle
import math
import numpy as np



# 全局变量df_std,用于规范输出格式(5min)
df_standard = pd.read_hdf('df_standard.h5')
# 全局变量freq,用于声明是用5min还是1min频率的数据来计算
freq = 5

def price_volume_corr(data, win):
    """
    高频因子八-量价相关性因子
    cov没有实现rolling，所以只能全算出来再隔30分钟取一条。
    检验完成
    """
    ind = df_standard.index
    return data['Volume_sum_'+str(freq)+'min'].rolling(window=int(win/freq)).corr(data['stock_close_'+str(freq)+'min']).loc[ind]

def high_price_deals(data, win):
    """
    高频因子八-高位成交因子
    个股在高位成交密集水平的刻画
    检验完成
    """
    VOL = data['Volume_sum_'+str(freq)+'min'].rolling(window=int(win/freq), step=int(30/freq)).sum()
    close_mean = data['stock_close_'+str(freq)+'min'].rolling(window=int(win/freq), step=int(30/freq)).mean()
    vol_mul_close = (data['Volume_sum_'+str(freq)+'min']*data['stock_close_'+str(freq)+'min']).rolling(window=int(win/freq), step=int(30/freq)).sum()
    #vol_mul_close = vol_mul_close.loc[close_mean.index]

    return (vol_mul_close/(close_mean*VOL)).loc[df_standard.index]


# def volume_weighted_skewness(data,freq):
#     """
#     高频因子八-加权偏度因子
#     """
#     df_volume = data['Volume_sum_'+str(freq)+'min']
#     df_close = data['stock_close_' + str(freq) + 'min']
#     # 生成新的df,其中的元素以元组形式保存
#     new_data = {}
#     for col in df_volume:
#         new_data[col] = [(a,b) for a,b in zip(df_volume[col], df_close[col])]
#     df = pd.DataFrame(new_data)
#     def calc_skewness(series):
#         """对于一个窗口内的数据，计算加权偏度"""
#         df_close = series.apply(lambda x:x[1])
#         df_volume = series.apply(lambda x:x[0])
#         close_sigma = df_close.std()
#         w = df_volume/df_volume.sum()
#
#         return (w*(df_close-df_close.mean())**3).sum()/(close_sigma**3)
#     # close_sigma = data['stock_close_'+str(freq)+'min'].rolling(window=int(720/freq), step=int(30/freq)).std()
#     # w = data['Volume_sum_'+str(freq)+'min']/data['Volume_sum_'+str(freq)+'min'].rolling(int(720/freq)).sum()
#     # close_3t = (w*(data['stock_close_'+str(freq)+'min']-data['stock_close_'+str(freq)+'min'].rolling(window=int(720/freq)).mean())**3).rolling(window=int(720/freq), step=int(30/freq)).sum()
#     # df = pd.concat([data['Volume_sum_'+str(freq)+'min'], data['stock_close_'+str(freq)+'min']],axis=1)
#     # l1 = data['Volume_sum_'+str(freq)+'min'].shape[1]
#     # l2 = data['stock_close_'+str(freq)+'min'].shape[1]
#     return df.rolling(window=int(720/freq), step=int(30/freq)).apply(lambda x: calc_skewness(x))


def volume_weighted_skewness(data, win):
    """
    高频因子八-加权偏度因子
    检验完成
    """
    df_volume = data['Volume_sum_'+str(freq)+'min']
    df_close = data['stock_close_' + str(freq) + 'min']

    def calc_skewness(df_vol, df_clo):
        """传入两个窗口Dataframe，计算加权偏度"""
        close_sigma = df_close.std()
        w = df_vol/df_vol.sum()

        return (w*(df_clo-df_clo.mean())**3).sum()/(close_sigma**3)

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    # 循环，手动取rolling的dataframe窗口
    for i in range(df_volume.shape[0]-1):
        if i%(30/freq)!=0:
            continue
        i_res = int(i/(30/freq))
        if i < win/freq:
            result.iloc[i_res,:] = np.nan
        else:
            result.iloc[i_res,:] = calc_skewness(df_volume.iloc[int(i-win/freq):i], df_close.iloc[int(i-win/freq):i])
    return result


def volume_entropy(data, win):
    """
    高频因子八-成交额熵因子
    检验完成，win取720对这个因子似乎太大了
    """
    df_volume = data['Volume_sum_' + str(freq) + 'min']
    df_close = data['stock_close_' + str(freq) + 'min']

    def calc_entropy(df_vol, df_clo):
        """传入两个窗口df，计算熵"""
        df_vol = df_vol/df_vol.sum()
        df_clo = df_clo/df_clo.sum()
        p = df_vol*df_clo
        return (-p*np.log(p)).sum()

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_entropy(df_volume.iloc[int(i - win / freq):i],
                                                  df_close.iloc[int(i - win / freq):i])

    return result

def amount_entropy(data, win):
    """
    高频因子八-成交额占比熵
    检验完成
    """
    df_volume = data['Volume_sum_' + str(freq) + 'min']
    df_close = data['stock_close_' + str(freq) + 'min']

    def calc_amount_entropy(df_vol, df_clo):
        """传入两个窗口df，计算熵"""
        df_amount = df_vol*df_clo
        p = df_amount/df_amount.sum()
        return (-p * np.log(p)).sum()

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_amount_entropy(df_volume.iloc[int(i - win / freq):i],
                                                 df_close.iloc[int(i - win / freq):i])

    return result

def combine_factor_highdeal(data, win):
    """
    高频因子八-合成因子
    检验完成
    """
    return (price_volume_corr(data, win)+volume_entropy(data, win)+volume_weighted_skewness(data,win))/3


def volume_volatility_factor(data, win, win2=20):
    """
    高频因子九-高频成交量波动因子
    检验完成
    """

    df_volume = data['Volume_sum_' + str(freq) + 'min']
    # 每30分钟计算一次过去30分钟成交量的标准差
    sigma_30min = df_volume.rolling(window=int(win/freq), step=int(30/freq)).std()
    # 取过去20个30min的数据进行标准化
    return sigma_30min.rolling(win2).std()/sigma_30min.rolling(win2).mean()


def count_volatility_factor(data, win, win2=20):
    """
    高频因子九-高频成交笔数波动因子
    检验完成
    """

    df_count = data['Count_sum_'+str(freq)+'min']
    # 每30分钟计算一次过去30分钟成交笔数的标准差
    sigma_30min = df_count.rolling(window=int(win / freq), step=int(30 / freq)).std()
    # 取过去20个30min的数据进行标准化
    return sigma_30min.rolling(win2).std() / sigma_30min.rolling(win2).mean()


def highlow_volatility_factor(data, win=30, win2=20):
    """
    高频因子九-高频振幅波动因子
    检验完成
    """

    df_highlow = data['stock_high_'+str(freq)+'min'] - data['stock_low_'+str(freq)+'min']
    # 每30分钟计算一次过去30分钟振幅的标准差
    sigma_30min = df_highlow.rolling(window=int(win / freq), step=int(30 / freq)).std()
    # 取过去20个30min的数据进行标准化
    return sigma_30min.rolling(win2).std() / sigma_30min.rolling(win2).mean()


def diffed_volume_percnt_volatility_factor(data, win=30, win2=20):
    """
    高频因子九-每笔成交量差分标准差因子
    检验完成
    """

    volb = data['Volume_sum_' + str(freq) + 'min']/data['Count_sum_'+str(freq)+'min']
    diff_volb = volb.diff()
    # 注意这里是先去量纲
    diff_volb_std30 = diff_volb.rolling(window=int(win / freq), step=int(30 / freq)).std()/volb.rolling(window=int(win / freq), step=int(30 / freq)).mean()
    return diff_volb_std30.rolling(win2).mean()


def diffed_abs_volume_percnt_volatility_factor(data, win=30, win2=20):
    """
    高频因子九-每笔成交量差分绝对值均值因子
    检验完成
    """
    volb = data['Volume_sum_' + str(freq) + 'min'] / data['Count_sum_' + str(freq) + 'min']
    diff_volb = volb.diff()
    diff_volb_mean30 = abs(diff_volb).rolling(window=int(win / freq), step=int(30 / freq)).mean() / volb.rolling(
        window=int(win / freq), step=int(30 / freq)).mean()
    return diff_volb_mean30.rolling(win2).mean()


def diffed_abs_volume_volatility_factor(data, win=30, win2=20):
    """
    高频因子九-成交量差分绝对值均值因子
    检验完成
    """
    vol = data['Volume_sum_'+str(freq)+'min']
    diff_vol = vol.diff()
    diff_vol_mean30 = abs(diff_vol).rolling(window=int(win / freq), step=int(30 / freq)).mean() / vol.rolling(
        window=int(win / freq), step=int(30 / freq)).mean()
    return diff_vol_mean30.rolling(win2).mean()

def diffed_abs_count_volatility_factor(data, win=30, win2=20):
    """
    高频因子九-成交笔数差分绝对值均值因子
    检验完成
    """
    count = data['Count_sum_'+str(freq)+'min']
    diff_count = count.diff()
    diff_count_mean30 = abs(diff_count).rolling(window=int(win / freq), step=int(30 / freq)).mean()/ count.rolling(
        window=int(win / freq), step=int(30 / freq)).mean()
    return diff_count_mean30.rolling(win2).mean()

def diffed_abs_highlow_volatility_factor(data, win=30, win2=20):
    """
    高频因子九-振幅差分绝对值均值因子
    检验完成
    """
    highlow = data['stock_high_' + str(freq) + 'min'] - data['stock_low_' + str(freq) + 'min']
    diff_highlow = highlow.diff()
    diff_highlow_mean30 = abs(diff_highlow).rolling(window=int(win / freq), step=int(30 / freq)).mean()/ highlow.rolling(
        window=int(win / freq), step=int(30 / freq)).mean()
    return diff_highlow_mean30.rolling(win2).mean()

def volume_peak_count_factor(data, win=60):
    """
    高频因子九-成交量波峰计数因子
    里面的60是参数，后期可以调一下
    运行很慢...
    检验完成
    """
    df_volume = data['Volume_sum_'+str(freq)+'min']
    def calc_peak_count(df_vol):
        """输入一个窗口成交量df，输出每个股票波峰数目"""
        threshold = df_vol.mean()+df_vol.std()
        df_peak = (df_vol>threshold).astype(int)
        # 相邻波峰不重复计算
        df_peak = df_peak - df_peak.shift(1)
        # 把以上计算可能出现的-1替换掉
        df_peak[df_peak==-1] = 0

        return df_peak.sum()

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_peak_count(df_volume.iloc[int(i - win / freq):i])

    return result


def high_freq_rev_factor(data, win=60):
    """
    高频因子十-高频反转因子（成交量加权收益率）
    检验通过
    """
    df_return = data['stock_close_'+str(freq)+'min']/data['stock_open_'+str(freq)+'min'] -1
    df_volume = data['Volume_sum_'+str(freq)+'min']
    def calc_vol_weighted_ret(df_ret, df_vol):
        vol_sum = df_vol.sum()
        return (df_ret*df_vol).sum()/vol_sum

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_vol_weighted_ret(df_return.iloc[int(i - win / freq):i],df_volume.iloc[int(i - win / freq):i])

    return result


def max_volume_per_cnt_rev(data, win=60):
    """
    高频因子十-每笔成交量筛选的局部反转因子,最大值组
    检验完成
    """
    df_return = data['stock_close_' + str(freq) + 'min'] / data['stock_open_' + str(freq) + 'min'] - 1
    df_volume_percnt = data['Volume_sum_' + str(freq) + 'min']/data['Count_sum_' + str(freq) +'min']

    def calc_max_rev(df_ret, df_vol):

        top_10 = df_vol.quantile(0.8)

        return df_ret[df_vol>=top_10].fillna(0).sum()

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume_percnt.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_max_rev(df_return.iloc[int(i - win / freq):i],
                                                          df_volume_percnt.iloc[int(i - win / freq):i])

    return result


def min_volume_per_cnt_rev(data, win=60):
    """
    高频因子十-每笔成交量筛选的局部反转因子,最小值组
    检验完成
    """
    df_return = data['stock_close_' + str(freq) + 'min'] / data['stock_open_' + str(freq) + 'min'] - 1
    df_volume_percnt = data['Volume_sum_' + str(freq) + 'min']/data['Count_sum_' + str(freq) +'min']

    def calc_max_rev(df_ret, df_vol):

        bottom_10 = df_vol.quantile(0.2)

        return df_ret[df_vol<=bottom_10].fillna(0).sum()

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume_percnt.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_max_rev(df_return.iloc[int(i - win / freq):i],
                                                          df_volume_percnt.iloc[int(i - win / freq):i])

    return result

def pvol_ret_corr_factor(data, win=60):
    """
    高频因子十-每笔成交量收益率相关性因子
    检验完成
    """
    volb = data['Volume_sum_' + str(freq) + 'min'] / data['Count_sum_' + str(freq) + 'min']
    df_return = data['stock_close_' + str(freq) + 'min'] / data['stock_open_' + str(freq) + 'min'] - 1

    return df_return.rolling(window= int(win/freq)).corr(volb).loc[df_standard.index]


def volume_cut_rev_factor(data, win=720):
    """
    高频因子十一-成交量划分下的反转因子-异常数据集
    检验完成
    """
    df_logreturn = (data['stock_close_' + str(freq) + 'min'] / data['stock_open_' + str(freq) + 'min']) - 1
    df_volume = data['Volume_sum_'+str(freq)+'min']
    def calc_abnormal_rev(df_lgret, df_vol):
        vol_threshold = df_vol.mean() + df_vol.std()
        abnor_data = df_vol >= vol_threshold
        return (df_vol*df_lgret*abnor_data).sum()/((df_vol*abnor_data).sum()+0.001)

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_abnormal_rev(df_logreturn.iloc[int(i - win / freq):i],
                                                 df_volume.iloc[int(i - win / freq):i])

    return result

def volume_cut_volatility_factor(data, win=720):
    """
    高频因子十一-成交量划分下的波动因子-异常数据集
    检验完成
    """
    df_return = data['stock_close_' + str(freq) + 'min'] / data['stock_open_' + str(freq) + 'min'] - 1
    df_volume = data['Volume_sum_'+str(freq)+'min']
    def calc_abnormal_volatility(df_ret, df_vol):
        vol_threshold = df_vol.mean() + df_vol.std()
        abnor_data = df_vol >= vol_threshold
        return (df_ret*abnor_data).std()

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_abnormal_volatility(df_return.iloc[int(i - win / freq):i],
                                                 df_volume.iloc[int(i - win / freq):i])

    return result

def volume_cut_liquidity_factor(data, win=720):
    """
    高频因子十一-成交量划分下的流动性因子-异常数据集
    检验完成
    """
    df_return = data['stock_close_' + str(freq) + 'min'] / data['stock_open_' + str(freq) + 'min'] - 1
    df_volume = data['Volume_sum_'+str(freq)+'min']
    df_amount = data['Amount_sum_'+str(freq)+'min']
    def calc_abnormal_liquidity(df_ret, df_vol, df_amt):
        vol_threshold = df_vol.mean() + df_vol.std()
        abnor_data = df_vol >= vol_threshold
        return (1+abs(df_ret[abnor_data].fillna(0))).cumprod()/((df_amt*abnor_data).sum()+0.001)

    result = pd.DataFrame(0, index=df_standard.index, columns=df_standard.columns)
    for i in range(df_volume.shape[0]-1):
        if i % (30 / freq) != 0:
            continue
        i_res = int(i / (30 / freq))
        if i < win / freq:
            result.iloc[i_res, :] = np.nan
        else:
            result.iloc[i_res, :] = calc_abnormal_liquidity(df_return.iloc[int(i - win / freq):i],
                                                 df_volume.iloc[int(i - win / freq):i],df_amount[int(i - win / freq):i])

    return result

if __name__ == '__main__':
    # 指定.pkl文件的路径
    file_path = 'data_5min.pkl'
    # 使用pickle.load()加载.pkl文件中的字典
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    # 补齐第一天index九点半的线，为了生成正确的索引。
    for f in [f'stock_close_{freq}min',f'stock_open_{freq}min',f'stock_high_{freq}min',f'stock_low_{freq}min']:
        data[f].loc[data[f].index[0]-pd.Timedelta(f'{freq}m')] = np.nan
        data[f] = data[f].sort_index()
    temp = high_freq_rev_factor(data)
    #price_vol_corr = price_volume_corr(data,720)
    #high_price_deal = high_price_deals(data)
    #weighted_skewness = volume_weighted_skewness(data)
    #entropy = volume_entropy(data)
    #peak_count = volume_peak_count_factor(data, freq=5)
    #max_volume_per_cnt = max_volume_per_cnt_rev(data, freq=5)
