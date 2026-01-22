import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import unicodedata
from datetime import datetime, date

# ================= 1. 系统配置 =================
st.set_page_config(page_title="Dual Momentum", layout="wide", page_icon="🛡️")

# --- A. 字体与显示适配 ---
FONT_FILE = "SimHei.ttf"
if os.path.exists(FONT_FILE):
    my_font = fm.FontProperties(fname=FONT_FILE)
else:
    my_font = fm.FontProperties(family='SimHei')

# --- B. 路径自动适配 ---
local_absolute_path = r"D:\SAR日频\全部品种日线"
relative_path = "data"

if os.path.exists(local_absolute_path):
    DEFAULT_DATA_FOLDER = local_absolute_path
elif os.path.exists(relative_path):
    DEFAULT_DATA_FOLDER = relative_path
else:
    DEFAULT_DATA_FOLDER = "."


# ================= 2. 数据处理 =================

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
        except:
            continue
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

            # ATR计算（全量数据计算）
            prev_close = df['close'].shift(1)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()],
                           axis=1).max(axis=1)
            atr = tr.rolling(atr_window).mean()
            natr = atr / df['close']

            price_dict[name] = df['close']
            vol_dict[name] = natr
            low_dict[name] = df['low']
            open_dict[name] = df['open']
        except:
            continue

        if i % 10 == 0: progress_bar.progress((i + 1) / len(files), text=f"加载: {name}")

    progress_bar.empty()

    if not price_dict: return None, None, None, None, "数据解析为空"

    return (pd.DataFrame(price_dict).ffill(), pd.DataFrame(vol_dict).ffill(),
            pd.DataFrame(low_dict).ffill(), pd.DataFrame(open_dict).ffill(), None)


# ================= 3. 核心策略逻辑 =================

def run_strategy_logic(df_p, df_v, df_l, df_o, params):
    # 1. 参数解包
    lookback_short = params['short']
    lookback_long = params['long']
    hold_num = params['hold_num']
    buffer_rank = params['buffer_rank']
    filter_ma = params['ma']

    # 止损参数
    stop_loss_trail = params['stop_loss_trail']  # 移动止损
    stop_loss_hard = params['stop_loss_hard']  # 硬止损

    commission_rate = params.get('commission', 0.0)
    slippage_rate = params.get('slippage', 0.0)

    start_date = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])

    # 2. 全局因子计算
    mom_short = df_p.pct_change(lookback_short)
    mom_long = df_p.pct_change(lookback_long)
    momentum_score = 0.4 * mom_short + 0.6 * mom_long
    ma_filter = df_p > df_p.rolling(filter_ma).mean()

    # 3. 定位回测起点
    dates = df_p.index
    try:
        start_idx = dates.get_indexer([start_date], method='bfill')[0]
    except:
        start_idx = 0
    if start_idx < 1: start_idx = 1

    if start_idx >= len(dates):
        return pd.DataFrame(), pd.DataFrame(), ["选定日期后无数据"]

    # 4. 初始化
    capital = 1.0
    nav_record = []
    asset_contribution = {}
    logs = []

    current_holdings = {}
    entry_prices = {}  # 记录开仓成本价

    # 周期统计
    cycle_details = []
    last_iso_week = None
    cycle_count = 1

    # --- 辅助函数：生成周度日志块 ---
    def generate_weekly_log(details, count, current_nav):
        if not details: return []

        block_logs = []
        c_ret = (np.prod([1 + d['ret'] for d in details]) - 1)
        c_cost = sum([d['cost'] for d in details])
        start_d_str = details[0]['date'].date()
        end_d_str = details[-1]['date'].date()

        # 1. 周报头
        header = f"第{count}周：{start_d_str} ~ {end_d_str} 收益: {c_ret * 100:+.2f}% | 成本: {c_cost * 10000:.0f}bp | 净值: {current_nav:.4f}"
        block_logs.append(header)

        # 2. 止损警报 (计算总收益贡献)
        for d in details:
            if d['stops']:
                for s in d['stops']:
                    total_loss_contrib = s['weight'] * s['ret']
                    # 显示止损类型
                    warn_line = f"⚠️ [{d['date'].date()}] {s['asset']} {s['reason']} (仓位:{s['weight']:.0%}, 总收益:{total_loss_contrib:.2%})"
                    block_logs.append(warn_line)

        # 3. 每日明细 (强制对齐)
        block_logs.append("")
        for d in details:
            hold_list = []
            for k, v in d['hold'].items():
                if v > 0:
                    r = d['asset_rets'].get(k, 0.0)
                    hold_list.append(f"{k}({v:.0%}, {r:+.1%})")
            h_txt = ",".join(hold_list) or "空仓"

            stop_tail = ""
            if d['stops']:
                s_list = []
                for s in d['stops']:
                    contrib = s['weight'] * s['ret']
                    s_list.append(f"{s['asset']}({contrib:.2%})")
                stop_tail = f" [止损:{','.join(s_list)}]"

            date_str = f"[{d['date'].date()}]"
            ret_str = f"{d['ret'] * 100:+.2f}%"
            cost_str = f"成本:{d['cost'] * 10000:.0f}bp"

            day_line = f"  {date_str} {ret_str:>7} | {cost_str:>9} | 持仓: {h_txt}{stop_tail}"

            block_logs.append(day_line)

        block_logs.append("-" * 65)
        return block_logs

    # --- 5. 逐日回测 ---
    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        if curr_date > end_date: break

        prev_date = dates[i - 1]

        # --- A. 自然周切分 ---
        curr_iso = curr_date.isocalendar()[:2]
        if last_iso_week is not None and curr_iso != last_iso_week:
            week_logs = generate_weekly_log(cycle_details, cycle_count, cycle_details[-1]['nav'])
            logs.extend(week_logs)
            cycle_count += 1
            cycle_details = []

        last_iso_week = curr_iso

        # --- B. 选股逻辑 ---
        target_holdings = {}
        daily_cost = 0.0

        try:
            if ma_filter.loc[prev_date].any():
                scores = momentum_score.loc[prev_date].dropna()
                valid_pool = [a for a in scores.index if ma_filter.loc[prev_date, a]]
                ranked_pool = scores.loc[valid_pool].sort_values(ascending=False)

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

                turnover = 0.0
                all_assets = set(current_holdings.keys()) | set(target_holdings.keys())
                for a in all_assets:
                    w_old = current_holdings.get(a, 0.0)
                    w_new = target_holdings.get(a, 0.0)
                    turnover += abs(w_new - w_old)
                    # 记录新开仓的成本价
                    if w_new > 0 and w_old == 0:
                        entry_prices[a] = df_p.loc[prev_date, a]

                daily_cost = turnover * (commission_rate + slippage_rate)
                current_holdings = target_holdings.copy()
            else:
                current_holdings = {}
                daily_cost = 0.0

        except:
            target_holdings = {}
            current_holdings = {}

        # --- C. 结算与风控 (双重止损逻辑) ---
        daily_gross_pnl = 0.0
        stopped_assets_info = []
        daily_asset_rets = {}

        for asset, w in list(current_holdings.items()):
            if w == 0: continue

            # 1. 移动止损线 (Trailing Stop)
            ref_trail = df_p.loc[prev_date, asset]
            stop_price_trail = ref_trail * (1 - stop_loss_trail)

            # 2. 硬止损线 (Hard Stop based on Entry)
            ref_entry = entry_prices.get(asset, ref_trail)  # 兜底，防止数据丢失
            stop_price_hard = ref_entry * (1 - stop_loss_hard)

            # 3. 最终有效止损价 (取两者较高者，保护性更强)
            effective_stop_price = max(stop_price_trail, stop_price_hard)

            # 获取当日数据
            today_open = df_o.loc[curr_date, asset]
            today_low = df_l.loc[curr_date, asset]
            today_close = df_p.loc[curr_date, asset]
            prev_close = df_p.loc[prev_date, asset]

            triggered = False
            actual_ret = 0.0
            stop_reason = ""

            # --- 判定逻辑 ---
            if today_open < effective_stop_price:
                # 场景A: 开盘直接低开在止损线下方 -> 跳空止损
                actual_ret = (today_open - prev_close) / prev_close
                triggered = True

                # 区分是哪种止损导致的
                if stop_price_hard > stop_price_trail:
                    stop_reason = "硬止损(跳空)"
                else:
                    stop_reason = "移动止损(跳空)"

            elif today_low < effective_stop_price:
                # 场景B: 盘中击穿止损线 -> 盘中止损
                # 按止损价离场
                actual_ret = (effective_stop_price - prev_close) / prev_close
                triggered = True

                if stop_price_hard > stop_price_trail:
                    stop_reason = "硬止损(盘中)"
                else:
                    stop_reason = "移动止损(盘中)"
            else:
                # 场景C: 安全持有
                actual_ret = (today_close - prev_close) / prev_close

            daily_gross_pnl += w * actual_ret
            asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret

            daily_asset_rets[asset] = actual_ret

            if triggered:
                current_holdings[asset] = 0
                if asset in entry_prices: del entry_prices[asset]
                stopped_assets_info.append({
                    'asset': asset,
                    'ret': actual_ret,
                    'reason': stop_reason,
                    'weight': w
                })

        daily_net_pnl = daily_gross_pnl - daily_cost
        capital *= (1 + daily_net_pnl)

        nav_record.append({'date': curr_date, 'nav': capital})

        cycle_details.append({
            'date': curr_date, 'ret': daily_net_pnl, 'cost': daily_cost,
            'nav': capital, 'hold': current_holdings.copy(),
            'stops': stopped_assets_info[:],
            'asset_rets': daily_asset_rets.copy()
        })

    # --- 6. 尾部处理 ---
    if cycle_details:
        week_logs = generate_weekly_log(cycle_details, cycle_count, capital)
        logs.extend(week_logs)

    return pd.DataFrame(nav_record), pd.DataFrame(list(asset_contribution.items()),
                                                  columns=['Asset', 'Contribution']), logs


# ================= 4. UI 界面 =================

with st.sidebar:
    st.header("Dual Momentum")
    st.caption(f"源: `{DEFAULT_DATA_FOLDER}`")
    data_folder = st.text_input("数据路径", value=DEFAULT_DATA_FOLDER)

    st.divider()
    col1, col2 = st.columns(2)

    # 定义可选的极宽范围 (例如：2000年 到 2050年)
    min_date = datetime(2000, 1, 1)
    max_date = datetime(2050, 12, 31)

    # 默认显示的时间
    default_start = pd.to_datetime("2025-01-01")
    default_end = pd.to_datetime("2026-12-31")

    start_d = col1.date_input(
        "开始日期",
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )

    end_d = col2.date_input(
        "结束日期",
        value=default_end,
        min_value=min_date,
        max_value=max_date
    )

    st.subheader("⚙️ 仓位与风控")
    c1, c2 = st.columns(2)
    hold_num = c1.number_input("持仓数", 1, 20, 5)
    buffer_rank = c2.number_input("排名缓冲", 1, 20, 8)

    st.markdown("---")
    st.write("🛑 **双重止损设置**")
    s1, s2 = st.columns(2)
    # 移动止损：保护利润
    stop_trail = s1.number_input("移动止损(%)", 0.0, 20.0, 4.0, step=0.5,
                                 help="基于前一日收盘价。如果今天回撤超过此幅度，止损。")
    # 硬止损：保本
    stop_hard = s2.number_input("硬止损(%)", 0.0, 20.0, 4.0, step=0.5,
                                help="基于开仓成本价。如果总亏损超过此幅度，无条件止损。")

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
st.title("Dual Momentum回测")

if run_btn:
    with st.spinner("加载数据..."):
        df_p, df_v, df_l, df_o, err = load_data_and_calc_atr(data_folder, atr_win)

    if err:
        st.error(err)
    else:
        if start_d >= end_d:
            st.error("开始日期必须早于结束日期")
        else:
            params = {
                'short': s_win, 'long': l_win, 'ma': ma_win,
                'hold_num': hold_num, 'buffer_rank': buffer_rank,
                'stop_loss_trail': stop_trail / 100.0,  # 移动止损参数
                'stop_loss_hard': stop_hard / 100.0,  # 硬止损参数
                'start_date': start_d, 'end_date': end_d,
                'commission': comm_bp / 10000, 'slippage': slip_bp / 10000
            }

            with st.spinner("策略计算中..."):
                res_nav, res_contrib, res_logs = run_strategy_logic(df_p, df_v, df_l, df_o, params)

            if res_nav.empty:
                st.warning("在此时间段内无交易或数据不足")
            else:
                res_nav.set_index('date', inplace=True)
                res_contrib.sort_values('Contribution', ascending=False, inplace=True)

                tot_ret = res_nav['nav'].iloc[-1] - 1
                days = (res_nav.index[-1] - res_nav.index[0]).days
                ann_ret = (1 + tot_ret) ** (365 / days) - 1 if days > 0 else 0
                dd = (res_nav['nav'] - res_nav['nav'].cummax()) / res_nav['nav'].cummax()
                max_dd = dd.min()
                d_rets = res_nav['nav'].pct_change().dropna()
                sharpe = (d_rets.mean() * 252) / (d_rets.std() * np.sqrt(252)) if d_rets.std() != 0 else 0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("总收益", f"{tot_ret * 100:.2f}%")
                k2.metric("年化收益", f"{ann_ret * 100:.2f}%")
                k3.metric("最大回撤", f"{max_dd * 100:.2f}%")
                k4.metric("夏普比率", f"{sharpe:.2f}")

                t1, t2, t3 = st.tabs(["📈 净值曲线", "📊 盈亏分布", "📝 交易日志"])

                with t1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = res_nav.index
                    y = res_nav['nav']
                    ax.plot(x, y, color='#d62728', lw=2, label='Strategy Nav')
                    ax.fill_between(x, y, 1, color='#d62728', alpha=0.1)
                    title_str = f"Net Value Curve | Ret:{tot_ret * 100:.1f}% | MaxDD:{max_dd * 100:.1f}%"
                    ax.set_title(title_str, fontproperties=my_font, fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(prop=my_font)
                    st.pyplot(fig)

                with t2:
                    if not res_contrib.empty:
                        st.dataframe(
                            res_contrib.style.format({'Contribution': '{:.2%}'}).background_gradient(cmap='RdYlGn'),
                            use_container_width=True)

                with t3:
                    st.text_area("详细日志", "\n".join(res_logs), height=600)
else:
    st.info("👈 请在左侧确认参数并点击【运行策略】")
