import os, pandas as pd, ib_insync, datetime, time, screeninfo, cv2
from lightweight_charts import Chart
from utils import constants, helpers

# Doc lightweight-charts: https://lightweight-charts-python.readthedocs.io/en/stable/tutorials/getting_started.html


def separate_df(df, column):

    dfs = []
    list_temp = []
    index_new_df = 0
    for index, df_element in enumerate(df[column]):

        if not pd.isna(df_element):
            list_temp.append(df_element)
            if index == len(df[column])-1:
                dfs.append(pd.DataFrame({"date": df["date"][index_new_df:index], column+str(len(dfs)):list_temp}))
        else:
            if index > 0 and not pd.isna(df[column][index-1]):
                dfs.append(pd.DataFrame({"date": df["date"][index_new_df:index-1], column+str(len(dfs)):list_temp}))
                list_temp = []
            index_new_df = index
    return dfs


def create_chart(df, symbol, timeframe, start_time, end_time, marker_list=[], screenshot_path="", show_time="block"):

    # Remove TZ from date otherwise automatically set to GMT
    for i, d in enumerate(df['date']):
        df['date'].iloc[i] = helpers.date_to_EST_aware(d, reverse=True)
    start_time = helpers.date_to_EST_aware(start_time, reverse=True)
    end_time = helpers.date_to_EST_aware(end_time, reverse=True)

    screen_number = 1 if len(screeninfo.get_monitors()) > 1 else 0
    chart_width = 0.9 * screeninfo.get_monitors()[screen_number].width
    chart_height = 0.9 * screeninfo.get_monitors()[screen_number].height
    chart_inner_height = 1
    if "macd" in df.columns: chart_inner_height -= 0.2
    if "rsi" in df.columns: chart_inner_height -= 0.2
    chart = Chart(inner_width=1, inner_height=chart_inner_height, screen=screen_number, width=chart_width, height=chart_height, scale_candles_only=True)
    chart.time_scale(visible=False)

    # Plot
    # ----
    chart.set(df.applymap(lambda x: str(x)))
    chart.watermark(symbol + " - " + timeframe)
    chart.set_visible_range(start_time=max(start_time, df["date"][0]), end_time=min(end_time, df["date"][len(df["date"])-1]))
    # chart.fit()

    # Indicators
    if "ema9" in df.columns:
        line_ema9 = chart.create_line(name="ema9", color="blue", style="solid", width=2, price_line=False, price_label=True)
        line_ema9.set(df[["date", "ema9"]].copy())
    if "ema20" in df.columns:
        line_ema20 = chart.create_line(name="ema20", color="green", style="solid", width=2, price_line=False, price_label=True)
        line_ema20.set(df[["date", "ema20"]].copy())
    if "ema50" in df.columns:
        line_ema50 = chart.create_line(name="ema50", color="yellow", style="solid", width=2, price_line=False, price_label=True)
        line_ema50.set(df[["date", "ema50"]].copy())
    if "vwap" in df.columns:
        line_vwap = chart.create_line(name="vwap", color="red", style="solid", width=2.5, price_line=False, price_label=True)
        line_vwap.set(df[["date", "vwap"]].copy())
    if "bband_h" in df.columns:
        line_bband_h = chart.create_line(name="bband_h", color="red", style="solid", width=1, price_line=False, price_label=True)
        line_bband_h.set(df[["date", "bband_h"]].copy())
    if "bband_l" in df.columns:
        line_bband_l = chart.create_line(name="bband_l", color="green", style="solid", width=1, price_line=False, price_label=True)
        line_bband_l.set(df[["date", "bband_l"]].copy())
    if "bband_mavg" in df.columns:
        line_bband_mavg = chart.create_line(name="bband_mavg", color="blue", style="solid", width=1, price_line=False, price_label=True)
        line_bband_mavg.set(df[["date", "bband_mavg"]].copy())
    for level in [{"name": "pdc", "color":"#40E0D0", "style":"solid", "width":1},
                  {"name": "pdh", "color":"green", "style":"solid", "width":1},
                  {"name": "pdl", "color":"red", "style":"solid", "width":1},
                  {"name": "pmh", "color":"purple", "style":"solid", "width":1},
                  {"name": "pml", "color":"orange", "style":"solid", "width":1},
                  {"name": "s3", "color":"#db944d", "style":"sparse_dotted", "width":2},
                  {"name": "s4", "color":"#d46724", "style":"sparse_dotted", "width":2},
                  {"name": "s5", "color":"#d44a24", "style":"sparse_dotted", "width":2},
                  {"name": "r3", "color":"#db944d", "style":"sparse_dotted", "width":2},
                  {"name": "r4", "color":"#d46724", "style":"sparse_dotted", "width":2},
                  {"name": "r5", "color":"#d44a24", "style":"sparse_dotted", "width":2}]:
        if level["name"] in df.columns:
            dfs_level = separate_df(df=df, column=level["name"])
            for index, df_level in enumerate(dfs_level):
                line_level = chart.create_line(name=level["name"]+str(index), color=level["color"], style=level["style"], width=level["width"], price_line=False, price_label=True)
                line_level.set(df_level[["date", level["name"]+str(index)]].copy())

    if "macd" in df.columns:
        chart_macd = chart.create_subchart(width=1, height=0.2, sync=True)
        chart_macd.time_scale(visible=False)
        if "macd_diff" in df.columns:
            line_macd_diff = chart_macd.create_histogram(name="macd_diff", color="green", price_line=False, price_label=True)#  , scale_margin_top=0.5, scale_margin_bottom=0.5)
            line_macd_diff.set(df[["date", "macd_diff"]].copy())
        line_macd = chart_macd.create_line(name="macd", color="blue", style="solid", width=1, price_line=False, price_label=True)
        line_macd.set(df[["date", "macd"]].copy())
        if "macd_signal" in df.columns:
            line_macd_signal = chart_macd.create_line(name="macd_signal", color="orange", style="solid", width=1, price_line=False, price_label=True)
            line_macd_signal.set(df[["date", "macd_signal"]].copy())

    if "rsi" in df.columns:
        chart_rsi = chart.create_subchart(width=1, height=0.2, sync=True)
        line_rsi = chart_rsi.create_line(name="rsi", color="blue", style="solid", width=1, price_line=False, price_label=True)
        line_rsi.set(df[["date", "rsi"]].copy())#.dropna())

        line_rsi_upper = chart_rsi.create_line(name="rsi_upper", color="white", width=1, style="solid", price_line=False, price_label=True)
        line_rsi_lower = chart_rsi.create_line(name="rsi_lower", color="white", width=1, style="solid", price_line=False, price_label=True)

        df_rsi_limits = pd.DataFrame({"date": df["date"], "rsi_upper": [70] * len(df), "rsi_lower": [30] * len(df)})
        line_rsi_upper.set(df_rsi_limits[["date", "rsi_upper"]].copy())
        line_rsi_lower.set(df_rsi_limits[["date", "rsi_lower"]].copy())
        # line_rsi_upper = chart_rsi.horizontal_line(price=70.0, color="white", width=1, style="solid")
        # line_rsi_lower = chart_rsi.horizontal_line(price=30.0, color="white", width=1, style="solid")

    # Markers
    # Remove TZ from markers otherwise automatically set to GMT
    for marker in marker_list:
        marker["time"] = helpers.date_to_EST_aware(marker["time"], reverse=True)
    chart.marker_list(marker_list)

    if show_time == "block": chart.show(block=True) # If block is enabled, the method will block code execution until the window is closed.
    else: chart.show()


    # Screenshot
    # ----------
    if screenshot_path:

        img = chart.screenshot()
        with open("img.jpg", 'wb') as f:
            f.write(img)
        f.close()
        img_list = [cv2.imread("img.jpg")]
        os.remove("img.jpg")

        if "macd" in df.columns:
            img_macd = chart_macd.screenshot()
            with open("img_macd.jpg", 'wb') as f:
                f.write(img_macd)
            f.close()
            img_list.append(cv2.imread("img_macd.jpg"))
            os.remove("img_macd.jpg")

        if "rsi" in df.columns:
            img_rsi = chart_rsi.screenshot()
            with open("img_rsi.jpg", 'wb') as f:
                f.write(img_rsi)
            f.close()
            img_list.append(cv2.imread("img_rsi.jpg"))
            os.remove("img_rsi.jpg")

        cv2.imwrite(screenshot_path, cv2.vconcat(img_list))

    if show_time != "na": time.sleep(show_time)
    chart.exit()


if __name__ == "__main__":

    import helpers

    print(os.getcwd())
    ib, ibConnection = helpers.IBKRConnect()

    symbol = "TSLA"
    timeframe = "1 min"

    # time_now_EST = helpers.date_local_to_EST(datetime.datetime.now())
    queryTime = helpers.date_to_EST_aware(datetime.datetime(2024, 8, 13, 16, 0))
    df = helpers.get_symbol_hist_data(ib, symbol, timeframe=timeframe, duration="1 D", queryTime=queryTime)

    create_chart(df, symbol, timeframe, screenshot=True)


    print("\n\n")
    # input("\nEnter anything to exit")