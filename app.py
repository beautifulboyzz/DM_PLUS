import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区 (每日详细版)
# ==========================================
DATA_FOLDER = r"D:\SAR日频\全部品种日线"
OUTPUT_FOLDER = r"D:\SAR日频\Dual Momentum（双重动量）"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# --- 核心策略参数 ---
LOOKBACK_SHORT = 5  # 短期动量
LOOKBACK_LONG = 20  # 长期动量
FILTER_MA = 60  # 趋势过滤
ATR_WINDOW = 20  # 波动率窗口

# --- 仓位与缓冲 ---
HOLD_NUM = 5  # 目标持仓数
BUFFER_RANK = 8  # 排名缓冲 (前8名不换股)

# --- 成本与风控 ---
STOP_LOSS_PCT = 0.04  # 4% 止损
COMMISSION = 0.0000  # 万3
SLIPPAGE = 0.0000  # 万5
RISK_FREE_RATE = 0.0


# ==========================================
# 2. 辅助类 (日志增强)
# ==========================================
class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        folder = os.path.dirname(filepath)
        if not os.path.exists(folder): os.makedirs(folder)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"====== 实战双重动量 (每日详细日志版) ======\n")
            f.write(f"参数: Top{HOLD_NUM} (Buffer{BUFFER_RANK}), StopLoss={STOP_LOSS_PCT * 100}%\n")
            f.write("==================================================\n\n")

    def log(self, content, print_to_console=True):
        if print_to_console: print(content)
        with open(self.filepath, 'a', encoding='utf-8') as f: f.write(content + "\n")


# ==========================================
# 3. 数据加载
# ==========================================
def load_data(folder):
    print("正在加载数据...")
    price_dict, vol_dict, low_dict, open_dict = {}, {}, {}, {}
    if not os.path.exists(folder): return None, None, None, None

    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    for file in files:
        path = os.path.join(folder, file)
        try:
            try:
                df = pd.read_csv(path, encoding='gbk')
            except:
                df = pd.read_csv(path, encoding='utf-8')

            # 映射列名
            rename_map = {'日期/时间': 'date', 'Date': 'date', '收盘价': 'close', 'Close': 'close',
                          '最高价': 'high', 'High': 'high', '最低价': 'low', 'Low': 'low', '开盘价': 'open',
                          'Open': 'open'}
            for col in df.columns:
                for k, v in rename_map.items():
                    if k in col and v not in df.columns: df.rename(columns={col: v}, inplace=True)

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close', 'high', 'low', 'open'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 计算 NATR
            prev_close = df['close'].shift(1)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()],
                           axis=1).max(axis=1)
            atr = tr.rolling(ATR_WINDOW).mean()
            natr = atr / df['close']

            name = file.split('.')[0]
            price_dict[name] = df['close']
            vol_dict[name] = natr
            low_dict[name] = df['low']
            open_dict[name] = df['open']
        except:
            continue

    return pd.DataFrame(price_dict).ffill(), pd.DataFrame(vol_dict).ffill(), \
        pd.DataFrame(low_dict).ffill(), pd.DataFrame(open_dict).ffill()


# ==========================================
# 4. 核心回测引擎 (每日逐行输出)
# ==========================================
def run_strategy(df_p, df_v, df_l, df_o, logger):
    logger.log("开始回测 (逐日模式)...", True)

    # 1. 因子计算
    mom_short = df_p.pct_change(LOOKBACK_SHORT)
    mom_long = df_p.pct_change(LOOKBACK_LONG)
    momentum_score = 0.4 * mom_short + 0.6 * mom_long
    ma_filter = df_p > df_p.rolling(FILTER_MA).mean()

    # 2. 变量初始化
    dates = df_p.index
    capital = 1.0
    nav_record = []
    asset_contribution = {}

    current_holdings = {}
    entry_prices = {}

    start_idx = 0
    min_calc_window = max(LOOKBACK_LONG, FILTER_MA, ATR_WINDOW)
    for i, d in enumerate(dates):
        if d >= pd.to_datetime(START_DATE) and i >= min_calc_window:
            start_idx = i
            break

    total_cost = 0.0

    # --- 逐日循环 ---
    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]
        if curr_date > pd.to_datetime(END_DATE): break

        # --- A. 选股与目标权重 ---
        target_holdings = {}
        try:
            scores = momentum_score.loc[prev_date].dropna()
            valid_pool = [a for a in scores.index if ma_filter.loc[prev_date, a]]
            ranked_pool = scores.loc[valid_pool].sort_values(ascending=False)

            # 排名缓冲逻辑
            keepers = []
            for asset in current_holdings.keys():
                if asset in ranked_pool.index:
                    rank = ranked_pool.index.get_loc(asset) + 1
                    if rank <= BUFFER_RANK: keepers.append(asset)

            slots_needed = HOLD_NUM - len(keepers)
            new_picks = []
            if slots_needed > 0:
                for asset in ranked_pool.index:
                    if asset not in keepers:
                        new_picks.append(asset)
                        if len(new_picks) == slots_needed: break

            final_assets = keepers + new_picks

            if final_assets:
                vols = df_v.loc[prev_date, final_assets]
                inv = 1.0 / (vols + 1e-6)
                target_holdings = (inv / inv.sum()).to_dict()
        except:
            target_holdings = {}

        # --- B. 生成调仓日志 (对比 Target vs Current) ---
        action_log = []
        all_assets = set(current_holdings.keys()) | set(target_holdings.keys())
        turnover = 0.0

        for asset in all_assets:
            w_old = current_holdings.get(asset, 0.0)
            w_new = target_holdings.get(asset, 0.0)
            turnover += abs(w_new - w_old)

            if w_old == 0 and w_new > 0:
                action_log.append(f"开仓 {asset}({w_new:.1%})")
                entry_prices[asset] = df_p.loc[prev_date, asset]  # 记录成本
            elif w_old > 0 and w_new == 0:
                action_log.append(f"清仓 {asset}")
            elif abs(w_new - w_old) > 0.01:  # 变动超过1%才显示，避免日志太乱
                action_log.append(f"调整 {asset}({w_old:.1%}->{w_new:.1%})")

        daily_cost = turnover * (COMMISSION + SLIPPAGE)
        total_cost += daily_cost
        current_holdings = target_holdings.copy()

        # --- C. 计算收益与风控 ---
        daily_gross_pnl = 0.0
        risk_msgs = []

        for asset, w in list(current_holdings.items()):
            if w == 0: continue

            ref_price = entry_prices.get(asset, df_p.loc[prev_date, asset])
            stop_price = ref_price * (1 - STOP_LOSS_PCT)

            today_open = df_o.loc[curr_date, asset]
            today_low = df_l.loc[curr_date, asset]
            today_close = df_p.loc[curr_date, asset]
            prev_close = df_p.loc[prev_date, asset]

            triggered = False
            actual_ret = 0.0

            if today_open < stop_price:  # 跳空
                actual_ret = (today_open - prev_close) / prev_close
                triggered = True
                risk_msgs.append(f"!!! {asset} 跳空止损 (开盘:{today_open})")
            elif today_low < stop_price:  # 盘中
                actual_ret = (stop_price - prev_close) / prev_close
                triggered = True
                risk_msgs.append(f"!!! {asset} 盘中止损 (低点:{today_low})")
            else:
                actual_ret = (today_close - prev_close) / prev_close

            daily_gross_pnl += w * actual_ret
            asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret

            if triggered:
                current_holdings[asset] = 0
                if asset in entry_prices: del entry_prices[asset]

        daily_net_ret = daily_gross_pnl - daily_cost
        capital *= (1 + daily_net_ret)
        nav_record.append({'date': curr_date, 'nav': capital})

        # --- D. 输出每日日志 ---
        date_str = curr_date.strftime("%Y-%m-%d")

        # 1. 标题行
        logger.log(f"[{date_str}] 净值: {capital:.4f} ({daily_net_ret * 100:+.2f}%) | 成本: {daily_cost * 10000:.1f}bp",
                   True)

        # 2. 持仓详情
        hold_str = ", ".join([f"{a}:{w * 100:.0f}%" for a, w in current_holdings.items() if w > 0])
        if not hold_str: hold_str = "空仓"
        logger.log(f"  持仓: {hold_str}", True)

        # 3. 调仓动作 (如果有)
        if action_log:
            logger.log(f"  调仓: {', '.join(action_log)}", True)

        # 4. 风控警报 (如果有)
        if risk_msgs:
            logger.log(f"  风控: {' | '.join(risk_msgs)}", True)

        # 分隔线
        if i % 5 == 0: logger.log("-" * 40, False)  # 文件里加分隔线，控制台不加

    return pd.DataFrame(nav_record).set_index('date'), pd.DataFrame(list(asset_contribution.items()),
                                                                    columns=['Asset', 'Contrib'])


# ==========================================
# 5. 报表生成 (不变)
# ==========================================
def create_report(res_df, contrib_df, logger):
    if res_df.empty: return
    # 绘图字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 1. 计算核心指标 ---
    # 总收益
    total_ret = res_df['nav'].iloc[-1] - 1
    # 交易天数转换年化
    days = (res_df.index[-1] - res_df.index[0]).days
    ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

    # 最大回撤
    dd_series = (res_df['nav'] - res_df['nav'].cummax()) / res_df['nav'].cummax()
    max_dd = dd_series.min()

    # 夏普率 (假设无风险利率为0)
    daily_rets = res_df['nav'].pct_change().dropna()
    volatility = daily_rets.std() * np.sqrt(252)  # 年化波动率
    sharpe = (ann_ret - RISK_FREE_RATE) / volatility if volatility != 0 else 0

    # 卡玛率
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # --- 2. 打印详细成绩单 ---
    logger.log("\n" + "=" * 20 + " 最终绩效评估 " + "=" * 20, True)
    logger.log(f"1. 收益表现:", True)
    logger.log(f"   - 总收益率: {total_ret * 100:+.2f}%", True)
    logger.log(f"   - 年化收益: {ann_ret * 100:+.2f}%", True)
    logger.log(f"\n2. 风险特征:", True)
    logger.log(f"   - 最大回撤: {max_dd * 100:+.2f}% (最惨时的跌幅)", True)
    logger.log(f"   - 年化波动: {volatility * 100:.2f}% (账户晃动的幅度)", True)
    logger.log(f"\n3. 性价比指标:", True)
    logger.log(f"   - 夏普率 (Sharpe): {sharpe:.2f} (每承担1单位波动赚取的收益，>1为佳)", True)
    logger.log(f"   - 卡玛率 (Calmar): {calmar:.2f} (年化收益/最大回撤，>1.5为优秀)", True)
    logger.log("=" * 54 + "\n", True)

    # --- 3. 绘图 (保持不变) ---
    x_data = res_df.index.to_numpy()
    y_nav = res_df['nav'].to_numpy()
    y_dd = dd_series.to_numpy()

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x_data, y_nav, color='#d62728', linewidth=2, label='Daily Strategy')
    ax1.fill_between(x_data, y_nav, 1, color='#d62728', alpha=0.1)
    ax1.set_title(f"策略净值 | Sharpe:{sharpe:.2f} | Calmar:{calmar:.2f} | Ret:{total_ret * 100:.1f}%")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.fill_between(x_data, y_dd, 0, color='gray', alpha=0.3)
    ax2.plot(x_data, y_dd, color='gray', linewidth=1)
    ax2.set_title(f"动态回撤 (最大回撤: {max_dd * 100:.2f}%)")
    ax2.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_FOLDER, "Daily_Report.jpg"))
    contrib_df.to_csv(os.path.join(OUTPUT_FOLDER, "Asset_Contrib.csv"), index=False, encoding='gbk')
    print(f"\n图表已保存: {os.path.join(OUTPUT_FOLDER, 'Daily_Report.jpg')}")


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    logger = Logger(os.path.join(OUTPUT_FOLDER, "Daily_Log.txt"))
    df_p, df_v, df_l, df_o = load_data(DATA_FOLDER)
    if df_p is not None:
        r_df, c_df = run_strategy(df_p, df_v, df_l, df_o, logger)
        create_report(r_df, c_df, logger)