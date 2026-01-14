import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import unicodedata
from datetime import datetime, date

# ================= 1. 系统配置 =================
st.set_page_config(page_title="Dual Momentum回测系统", layout="wide", page_icon="⚡")

# --- A. 字体与显示适配 (解决Linux/云端中文乱码) ---
# 优先尝试加载项目根目录下的 SimHei.ttf
FONT_FILE = "SimHei.ttf" 
if os.path.exists(FONT_FILE):
    my_font = fm.FontProperties(fname=FONT_FILE)
else:
    # 本地 Windows 兜底
    my_font = fm.FontProperties(family='SimHei')

# --- B. 路径自动适配 ---
# 优先本地路径，其次 relative path (data/)
local_absolute_path = r"D:\SAR日频\全部品种日线"
relative_path = "data"

if os.path.exists(local_absolute_path):
    DEFAULT_DATA_FOLDER = local_absolute_path
elif os.path.exists(relative_path):
    DEFAULT_DATA_FOLDER = relative_path
else:
    DEFAULT_DATA_FOLDER = "."

# ================= 2. 数据处理 (保持健壮性) =================

def read_robust_csv(f):
    for enc in ['gbk', 'utf-8', 'gb18030', 'cp936']:
        try:
            df = pd.read_csv(f, encoding=enc, engine='python')
            cols = [str(c).strip() for c in df.columns]
            rename_map = {}
            for c in df.columns:
                c_str = str(c).strip()
                if c_str in ['日期', '日期/时间', 'date', 'Date']: rename_map[c] = 'date'
                if c_str in ['收盘价', '收盘', 'close', 'price', 'Close']: rename_map[c] = 'close'
                if c_str in ['最高价', '最高', 'high', 'High']: rename_map[c] = 'high'
                if c_str in ['最低价', '最低', 'low', 'Low']: rename_map[c] = 'low'
                if c_str in ['开盘价', '开盘', 'open', 'Open']: rename_map[c] = 'open'

            df.rename(columns=rename_map, inplace=True)
            if 'date' in df.columns and 'close' in df.columns:
                return df
        except: continue
    return None

@st.cache_data(ttl=3600)
def load_data_and_calc_atr(folder, atr_window=20):
    if not os.path.exists(folder):
        return None, None, None, None, f"路径不存在: {folder}"

    try:
        files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    except:
        return None, None, None, None, "无法读取目录"

    if not files:
        return None, None, None, None, "无CSV文件"

    price_dict, vol_dict, low_dict, open_dict = {}, {}, {}, {}
    progress_bar = st.progress(0, text="正在加载数据...")

    for i, file in enumerate(files):
        file_norm = unicodedata.normalize('NFC', file)
        if "纤维板" in file_norm or "胶合板" in file_norm or "线材" in file_norm: continue

        name = file_norm.split('.')[0].replace("主连", "").replace("日线", "")
        path = os.path.join(folder, file)
        df = read_robust_csv(path)
        if df is None: continue

        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close', 'high', 'low', 'open'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.sort_values('date', inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            df.set_index('date', inplace=True)

            prev_close = df['close'].shift(1)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(atr_window).mean()
            natr = atr / df['close']

            price_dict[name] = df['close']
            vol_dict[name] = natr
            low_dict[name] = df['low']
            open_dict[name] = df['open']
        except: continue

        if i % 10 == 0: progress_bar.progress((i + 1) / len(files), text=f"加载: {name}")

    progress_bar.empty()
    
    if not price_dict: return None, None, None, None, "数据解析为空"

    return (pd.DataFrame(price_dict).ffill(), pd.DataFrame(vol_dict).ffill(), 
            pd.DataFrame(low_dict).ffill(), pd.DataFrame(open_dict).ffill(), None)


# ================= 3. 核心策略逻辑 (已同步 1.py 的所有高级逻辑) =================

def run_strategy_logic(df_p, df_v, df_l, df_o, params):
    # 解包参数
    lookback_short = params['short']
    lookback_long = params['long']
    hold_num = params['hold_num']
    buffer_rank = params['buffer_rank'] # 新增：排名缓冲
    filter_ma = params['ma']
    stop_loss_pct = params['stop_loss_pct']
    commission_rate = params.get('commission', 0.0)
    slippage_rate = params.get('slippage', 0.0)

    start_date = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])

    # 因子计算
    mom_short = df_p.pct_change(lookback_short)
    mom_long = df_p.pct_change(lookback_long)
    momentum_score = 0.4 * mom_short + 0.6 * mom_long
    ma_filter = df_p > df_p.rolling(filter_ma).mean()
    
    # 准备回测
    dates = df_p.index
    capital = 1.0
    nav_record = []
    asset_contribution = {}
    logs = []
    
    current_holdings = {}
    entry_prices = {}
    
    # 定位起点
    try: start_idx = dates.get_indexer([start_date], method='bfill')[0]
    except: start_idx = 0
    min_idx = max(lookback_long, filter_ma, 20)
    start_idx = max(start_idx, min_idx)
    
    if start_idx >= len(dates): return pd.DataFrame(), pd.DataFrame(), ["数据不足"]

    cycle_details = []
    cycle_count = 1

    # --- 逐日回测 ---
    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        if curr_date > end_date: break
        prev_date = dates[i - 1]
        
        target_holdings = {}
        daily_cost = 0.0

        # A. 选股 (包含 Buffer Logic)
        try:
            scores = momentum_score.loc[prev_date].dropna()
            valid_pool = [a for a in scores.index if ma_filter.loc[prev_date, a]]
            ranked_pool = scores.loc[valid_pool].sort_values(ascending=False)
            
            # --- 核心同步：排名缓冲逻辑 ---
            keepers = []
            for asset in current_holdings.keys():
                if asset in ranked_pool.index:
                    rank = ranked_pool.index.get_loc(asset) + 1
                    if rank <= buffer_rank: keepers.append(asset)
            
            slots_needed = hold_num - len(keepers)
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
                
            # 计算成本
            turnover = 0.0
            all_assets = set(current_holdings.keys()) | set(target_holdings.keys())
            for a in all_assets:
                w_old = current_holdings.get(a, 0.0)
                w_new = target_holdings.get(a, 0.0)
                turnover += abs(w_new - w_old)
                if w_new > 0 and w_old == 0: # 记录新开仓成本价
                    entry_prices[a] = df_p.loc[prev_date, a]
            
            daily_cost = turnover * (commission_rate + slippage_rate)
            current_holdings = target_holdings.copy()

        except:
            target_holdings = {}
            current_holdings = {}
        
        # B. 结算与风控 (包含 Gap Logic)
        daily_gross_pnl = 0.0
        stopped_assets = []
        
        for asset, w in list(current_holdings.items()):
            if w == 0: continue
            
            ref_price = entry_prices.get(asset, df_p.loc[prev_date, asset])
            stop_price = ref_price * (1 - stop_loss_pct)
            
            today_open = df_o.loc[curr_date, asset]
            today_low = df_l.loc[curr_date, asset]
            today_close = df_p.loc[curr_date, asset]
            prev_close = df_p.loc[prev_date, asset]
            
            triggered = False
            actual_ret = 0.0
            
            # --- 核心同步：真实跳空逻辑 ---
            if today_open < stop_price:
                actual_ret = (today_open - prev_close) / prev_close
                triggered = True
                stopped_assets.append(f"{asset}(跳空)")
            elif today_low < stop_price:
                actual_ret = (stop_price - prev_close) / prev_close
                triggered = True
                stopped_assets.append(f"{asset}(止损)")
            else:
                actual_ret = (today_close - prev_close) / prev_close
                
            daily_gross_pnl += w * actual_ret
            asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret
            
            if triggered:
                current_holdings[asset] = 0
                if asset in entry_prices: del entry_prices[asset]
        
        daily_net_pnl = daily_gross_pnl - daily_cost
        capital *= (1 + daily_net_pnl)
        nav_record.append({'date': curr_date, 'nav': capital})
        
        # 日志缓存
        cycle_details.append({
            'date': curr_date, 'ret': daily_net_pnl, 'cost': daily_cost,
            'nav': capital, 'hold': current_holdings.copy(), 'stop': stopped_assets[:]
        })
        
        if stopped_assets:
            logs.append(f"⚠️ [{curr_date.date()}] 风控: {', '.join(stopped_assets)}")

        # 周期输出
        if len(cycle_details) == 5 or i == len(dates) - 1 or curr_date == end_date:
            c_ret = (np.prod([1+d['ret'] for d in cycle_details]) - 1)
            c_cost = sum([d['cost'] for d in cycle_details])
            
            h_str = f"=== 周期 {cycle_count} ({cycle_details[0]['date'].date()} ~ {cycle_details[-1]['date'].date()}) "
            h_str += f"收益: {c_ret*100:+.2f}% | 成本: {c_cost*10000:.1f}bp | 净值: {capital:.4f} ==="
            logs.append(h_str)
            
            for d in cycle_details:
                h_txt = ",".join([f"{k}({v:.0%})" for k,v in d['hold'].items() if v>0]) or "空仓"
                s_txt = f" [止损:{','.join(d['stop'])}]" if d['stop'] else ""
                logs.append(f"  [{d['date'].date()}] {d['ret']*100:+.2f}% | 成本:{d['cost']*10000:.0f}bp | 持仓: {h_txt}{s_txt}")
            
            logs.append("-" * 40)
            cycle_details = []
            cycle_count += 1

    return pd.DataFrame(nav_record), pd.DataFrame(list(asset_contribution.items()), columns=['Asset', 'Contribution']), logs

# ================= 4. UI 界面 =================

with st.sidebar:
    st.header("Dual Momentum (Pro)")
    st.caption(f"源: `{DEFAULT_DATA_FOLDER}`")
    data_folder = st.text_input("数据路径", value=DEFAULT_DATA_FOLDER)
    
    st.divider()
    col1, col2 = st.columns(2)
    start_d = col1.date_input("开始", pd.to_datetime("2025-01-01"))
    end_d = col2.date_input("结束", pd.to_datetime("2025-12-31"))
    
    st.subheader("⚙️ 仓位风控")
    c1, c2 = st.columns(2)
    hold_num = c1.number_input("持仓数", 1, 20, 5)
    buffer_rank = c2.number_input("排名缓冲", 1, 20, 8, help="前X名不换股 (逻辑同步自1.py)")
    stop_loss = st.number_input("止损 (%)", 0.0, 20.0, 4.0, step=0.5) / 100.0
    
    st.subheader("💸 交易成本")
    cc1, cc2 = st.columns(2)
    comm_bp = cc1.number_input("手续费(bp)", 0.0, 50.0, 0.0)
    slip_bp = cc2.number_input("滑点(bp)", 0.0, 50.0, 0.0)

    with st.expander("🛠️ 因子参数"):
        s_win = st.number_input("短期窗口", value=5)
        l_win = st.number_input("长期窗口", value=20)
        ma_win = st.number_input("均线过滤", value=60)
        atr_win = st.number_input("ATR周期", value=20)
        
    run_btn = st.button("🚀 运行策略", type="primary", use_container_width=True)

# 主显示区
st.title("Dual Momentum 回测")

if run_btn:
    with st.spinner("加载数据..."):
        df_p, df_v, df_l, df_o, err = load_data_and_calc_atr(data_folder, atr_win)
    
    if err:
        st.error(err)
    else:
        params = {
            'short': s_win, 'long': l_win, 'ma': ma_win,
            'hold_num': hold_num, 'buffer_rank': buffer_rank, # 关键参数
            'stop_loss_pct': stop_loss,
            'start_date': start_d, 'end_date': end_d,
            'commission': comm_bp/10000, 'slippage': slip_bp/10000
        }
        
        with st.spinner("策略计算中..."):
            res_nav, res_contrib, res_logs = run_strategy_logic(df_p, df_v, df_l, df_o, params)
            
        if res_nav.empty:
            st.warning("无交易结果")
        else:
            res_nav.set_index('date', inplace=True)
            res_contrib.sort_values('Contribution', ascending=False, inplace=True)
            
            # 指标计算
            tot_ret = res_nav['nav'].iloc[-1] - 1
            days = (res_nav.index[-1] - res_nav.index[0]).days
            ann_ret = (1 + tot_ret) ** (365/days) - 1 if days > 0 else 0
            dd = (res_nav['nav'] - res_nav['nav'].cummax()) / res_nav['nav'].cummax()
            max_dd = dd.min()
            d_rets = res_nav['nav'].pct_change().dropna()
            sharpe = (d_rets.mean()*252) / (d_rets.std()*np.sqrt(252)) if d_rets.std()!=0 else 0
            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
            
            # 显示指标
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益", f"{tot_ret*100:.2f}%")
            k2.metric("年化", f"{ann_ret*100:.2f}%")
            k3.metric("最大回撤", f"{max_dd*100:.2f}%")
            k4.metric("夏普", f"{sharpe:.2f}")
            
            t1, t2, t3 = st.tabs(["📈 净值曲线", "📊 盈亏分布", "📝 交易日志"])
            
            # --- 核心修改：Matplotlib 绘图 (移除回撤子图) ---
            with t1:
                # 显式创建 Figure，避免 st.pyplot() 调用空白
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # 只画净值，不画回撤
                x = res_nav.index
                y = res_nav['nav']
                ax.plot(x, y, color='#d62728', lw=2, label='Strategy Nav')
                ax.fill_between(x, y, 1, color='#d62728', alpha=0.1)
                
                # 设置标题字体
                title_str = f"Net Value Curve | Ret:{tot_ret*100:.1f}% | MaxDD:{max_dd*100:.1f}%"
                ax.set_title(title_str, fontproperties=my_font, fontsize=12)
                
                ax.grid(True, alpha=0.3)
                ax.legend(prop=my_font)
                
                # 传递 fig 对象，确保不显示空白
                st.pyplot(fig)
                
            # --- 盈亏分布 ---
            with t2:
                if not res_contrib.empty:
                    st.dataframe(res_contrib.style.format({'Contribution': '{:.2%}'}).background_gradient(cmap='RdYlGn'), use_container_width=True)
            
            # --- 日志 ---
            with t3:
                st.text_area("详细日志", "\n".join(res_logs), height=500)
else:
    st.info("👈 请在左侧确认参数并点击【运行策略】")


