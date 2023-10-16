#basic packages
import streamlit as st
import math
import pandas as pd
import numpy as np
from numpy.ma.core import log
from datetime import datetime, timedelta

#changepoint detection library
import ruptures as rpt

import chart_studio.plotly as py
from plotly import graph_objs as go

from google.oauth2 import service_account
from google.cloud import bigquery

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

#Page Configuration
st.set_page_config(
                    page_title='Snapchat Dynamic Scheduling',
                    page_icon = 'https://w7.pngwing.com/pngs/481/484/png-transparent-snapchat-logo-snap-inc-social-media-computer-icons-snapchat-text-logo-smiley.png',
                    layout='wide'
                  )
# header of the page
html_temp = """
            <div style ="background-color:#00008B; border: 8px darkblue; padding: 18px; text-align: right">
            <!<img src="https://www.rewindandcapture.com/wp-content/uploads/2014/04/snapchat-logo.png" width="100"/>>
            <h1 style ="color:lightgrey;text-align:center;">Snapchat Dynamic Scheduling</h1>
            </div>
            """
st.markdown(html_temp,unsafe_allow_html=True)

#Minor template configurations
css_background = """
                  <style>
                  h1   {color: darkblue;}
                  p    {color: darkred;}
                  </style>
                  """
st.markdown(css_background,unsafe_allow_html=True)

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

#Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)

#Functions


#Functions powering the app

#Round to multiple
def round_to_multiple(number, multiple):
  return multiple * math.ceil(number / multiple)

#Total view forecast
def forecast_totalview(choose_episode, choose_hours):
  this_episode_df = df[df['story_id'].isin([choose_episode])].drop_duplicates(subset='interval_time')
  this_episode_df['actual'] = this_episode_df['actual'].astype('float').fillna(np.nan)

  #New variable for df for further data manipulation/cleaning
  data = this_episode_df

  #Historical df
  historical = data[data['forecast_type'].isin(['history'])]

  #Df conditions, daily predictions & momentum
  #Episodes less than 10 rows of data
  if len(historical) < 10 or len(data.loc[data['forecast_type'] == 'future']) <=0:
    prediction = historical

    total_prediction =None
    daily_prediction =None
    momentum =None

  #Episodes fit for forecasting
  else:
    prediction = data.loc[data['true_hour'] <= choose_hours]

    #Total Forecast
    total_prediction = round(prediction.tail(1)['topsnap_views'].values[0])
    #Daily Fcst
    start_end = prediction.tail(25)
    start = start_end.head(1)['topsnap_views'].values[0]
    end = start_end.tail(1)['topsnap_views'].values[0]

    #Momentum
    momentum_df = prediction.tail(49)

    if choose_hours <= 24:
      daily_prediction = total_prediction
      momentum=None

    else:
      daily_prediction = round(end-start)

      if choose_hours == 48:
        previous_day = float(momentum_df.loc[momentum_df['true_hour'] == choose_hours-24, ['topsnap_views']].head(1).values[0])
      else:
        m_start = momentum_df.head(1)['topsnap_views'].values[0]
        m_end = float(momentum_df.loc[momentum_df['true_hour'] == choose_hours-24, ['topsnap_views']].head(1).values[0])
        previous_day = round(m_end-m_start)

      try:
          momentum = (daily_prediction-previous_day)/previous_day
      except ZeroDivisionError:
          momentum = 0

      if momentum > 0:
        momentum = f'+{round(momentum*100, 2)}%'
      else:
        momentum = f'{round(momentum*100, 2)}%'

  #Changepoint Detection
  if this_episode_df['true_hour'].values[-1] < 24:
    first_y = np.nan
    first_x = np.nan
    second_y =  np.nan
    second_x = np.nan

    first_dy = np.nan
    first_dx = np.nan
    second_dy =  np.nan
    second_dx = np.nan

  else:
    #Add & Merge changepoint analysis & metrics
    current = changepoint_df(choose_episode)
    merged = prediction.merge(current[['true_hour','direction']], on='true_hour', how='left')

    #Conditional logic for Change Point Detection (most recent 2 hot and cold change points)
    up = merged.loc[merged['direction'] == 'upward change', ['interval_time', 'topsnap_views']]
    down = merged.loc[merged['direction'] == 'downward change', ['interval_time', 'topsnap_views']]

    #Up-tick
    if len(up) == 0:
      first_y = np.nan
      first_x = np.nan
      second_y =  np.nan
      second_x = np.nan
    elif len(up) == 1:
      first_y= up['topsnap_views'].values[-1]
      first_x = pd.to_datetime(up['interval_time'].values[-1])
      second_y = np.nan
      second_x = np.nan
    elif len(up) >= 2:
      first_y= up['topsnap_views'].values[-1]
      first_x = pd.to_datetime(up['interval_time'].values[-1])
      second_y = up['topsnap_views'].values[-2]
      second_x = pd.to_datetime(up['interval_time'].values[-2])

    #Down-tick
    if len(down) == 0:
      first_dy = np.nan
      first_dx = np.nan
      second_dy =  np.nan
      second_dx = np.nan
    elif len(down) == 1:
      first_dy= down['topsnap_views'].values[-1]
      first_dx = pd.to_datetime(down['interval_time'].values[-1])
      second_dy = np.nan
      second_dx = np.nan
    elif len(down) >= 2:
      first_dy= down['topsnap_views'].values[-1]
      first_dx = pd.to_datetime(down['interval_time'].values[-1])
      second_dy = down['topsnap_views'].values[-2]
      second_dx = pd.to_datetime(down['interval_time'].values[-2])


  yhat = go.Scatter(x = prediction['interval_time'],
                  y = prediction['future_fcst'],
                    #y = prediction['yhat24'],
                    mode = 'lines',
                    marker = {'color': 'blue'},
                    line = {'width': 4},
                    name = 'Future Forecast',
                    )

  yhat_2 = go.Scatter(x = prediction['interval_time'],
                  y = prediction['historical_fcst'],
                    #y = prediction['yhat24'],
                    mode = 'lines',
                    marker = {'color': 'darkslateblue'},
                    line = {'width': 4},
                    name = 'Historical Forecast',
                    )

  yhat_lower = go.Scatter(x = prediction['interval_time'],
                        y = prediction['confidence_interval_lower_bound'],
                          marker = {'color': 'powderblue'},
                          showlegend = False,
                          #hoverinfo = 'none',
                          )

  yhat_upper = go.Scatter(x = prediction['interval_time'],
                        y = prediction['confidence_interval_upper_bound'],
                          fill='tonexty',
                          fillcolor = 'powderblue',
                          name = 'Confidence (80%)',
                          #hoverinfo = 'yhat_upper',
                          mode = 'none'
                          )

  actual = go.Scatter(x = prediction['interval_time'],
                    y = prediction['actual'],
                    mode = 'markers',
                    marker = {'color': '#fffaef','size': 10,'line': {'color': '#000000',
                                                                      'width': 0.8}},
                      name = 'Actual'
                      )

  layout = go.Layout(yaxis = {'title': 'Topsnap Views',},
                     hovermode = 'x',
                     xaxis = {'title': 'Hours/Days'},
                     margin = {'t': 20,'b': 50,'l': 60,'r': 10},
                     legend = {'bgcolor': 'rgba(0,0,0,0)'})

  layout_data = [yhat_lower, yhat_upper, yhat, yhat_2, actual]

  # Get Episode Name
  episode_name = this_episode_df.head(1)['title'].values[0]

  #Get Channel Name
  channel_df = benchmarks[benchmarks['name'].isin(this_episode_df.name)]
  channel_name = channel_df.head(1)['name'].values[0]

  #Banger Benchmark
  banger = channel_df.loc[channel_df['true_hour'] == 168, ['topsnap_views_total']]
  if len(banger) == 0:
    banger_bench = 0
  else:
    banger_bench = banger['topsnap_views_total'].mean()*2

  #Get current hour benchmark
  def get_benchmarks(choose):
    b_channel = channel_df.loc[channel_df['true_hour'] == choose, ['topsnap_views_total']]
    if len(b_channel)<= 0:
      channel_bench = 0
    else:
      channel_bench = b_channel['topsnap_views_total'].mean()
    return channel_bench

  if choose_hours <= 24:
    channel_bench = get_benchmarks(24)
    day = 'Day 1'

  elif ((choose_hours > 24) and (choose_hours <= 48)):
    channel_bench = get_benchmarks(48)
    day = 'Day 2'

  elif ((choose_hours > 48) and (choose_hours <= 72)):
    channel_bench = get_benchmarks(72)
    day = 'Day 3'

  elif ((choose_hours > 72) and (choose_hours <= 96)):
    channel_bench = get_benchmarks(96)
    day = 'Day 4'

  elif ((choose_hours > 96) and (choose_hours <= 120)):
    channel_bench = get_benchmarks(120)
    day = 'Day 5'

  elif ((choose_hours > 120) and (choose_hours <= 144)):
    channel_bench = get_benchmarks(144)
    day = 'Day 6'

  elif ((choose_hours > 144) and (choose_hours <= 168)):
    channel_bench = get_benchmarks(168)
    day = 'Day 7'

  elif ((choose_hours > 168) and (choose_hours <= 192)):
    channel_bench = get_benchmarks(192)
    day = 'Day 8'

  elif ((choose_hours > 192) and (choose_hours <= 216)):
    channel_bench = get_benchmarks(216)
    day = 'Day 9'

  elif ((choose_hours > 216) and (choose_hours <= 240)):
    channel_bench = get_benchmarks(240)
    day = 'Day 10'

  elif ((choose_hours > 240) and (choose_hours <= 264)):
    channel_bench = get_benchmarks(264)
    day = 'Day 11'

  elif ((choose_hours > 264) and (choose_hours <= 288)):
    channel_bench = get_benchmarks(288)
    day = 'Day 12'

  elif ((choose_hours > 288) and (choose_hours <= 312)):
    channel_bench = get_benchmarks(312)
    day = 'Day 13'

  elif ((choose_hours > 312) and (choose_hours <= 336)):
    channel_bench = get_benchmarks(336)
    day = 'Day 14'

  #Enough hours to forecast
  if choose_hours < this_episode_df.tail(1)['true_hour'].values[0] and len(historical) >= 10:
    if channel_bench == 0:
      trending = None
    elif channel_bench > 0:
      trending = round(((total_prediction-channel_bench)/channel_bench)*100)
      if trending > 0:
        trending = f'+{round(trending):,}% above'
      else:
        trending = f'{round(trending):,}% below'
    else:
      trending = None

    total_prediction = f'{round(total_prediction):,}'
    daily_prediction = f'{round(daily_prediction):,}'

  else:
    total_prediction = 'Not enough data to forecast OR prediction is past 72 hours'
    daily_prediction = None
    momentum = None
    trending = None
    day = ''

  #Store line graph layout
  fig = go.Figure(data= layout_data, layout=layout)

  #Update layout with metrics and benchmarks
  fig.update_layout(title={'text': (f'<b>{episode_name} - {channel_name}</b><br><br><sup>Total Topsnap Prediction = <b>{total_prediction}</b> ({trending} Avg)<br>{day} Topsnap Prediction = <b>{daily_prediction}</b><br>Daily Momentum % = <b>{momentum}</b></sup>'),
                           'y':0.91,
                           'x':0.075,
                           'font_size':22,
                           })

  fig.update_traces(hovertext=prediction.true_hour)

  fig.add_hline(y=channel_bench, line_dash="dot", line_color='purple',
                annotation_text=(f"Channel Avg at {choose_hours}hrs: <b>{round(channel_bench):,}</b>"),
              annotation_position="bottom right",
              annotation_font_size=14,
              annotation_font_color="purple"
             )

  fig.add_hline(y=banger_bench, line_dash="dot", line_color='gold',
                annotation_text="168hr Banger Benchmark",
              annotation_position="bottom right",
              annotation_font_size=14,
              annotation_font_color="black"
             )

  fig.add_trace(
    go.Scatter(
        mode='markers+text',
        x=[first_x, second_x],
        y=[first_y, second_y],
        name='improving trend',
        text='🔥',
        textposition='middle center',
        marker=dict(
            color='#FF4500',
            size=5)
    )
    )

  fig.add_trace(
    go.Scatter(
        mode='markers+text',
        x=[first_dx, second_dx],
        y=[first_dy, second_dy],
        name='slowing trend',
        text='🥶',
        textposition='middle center',
        marker=dict(
            color='#1E90FF',
            size=5)
    )
  )
  fig.update_traces(textfont_size=22)

  return fig

#Momentum chart function
def momentum_chart(choose_episode):
  this_episode_df = df[df['story_id'].isin([choose_episode])].drop_duplicates(subset='interval_time')
  this_episode_df['actual'] = this_episode_df['actual'].astype('float').fillna(np.nan)

  tt = pd.Timestamp.today().floor('H')
  np_today = tt.to_numpy()
  today = np_today - np.timedelta64(4, 'h')

  last_hour = int(this_episode_df.loc[this_episode_df['interval_time'] == today, ['true_hour']].values[0])
  fcst_hour = round_to_multiple(last_hour, 24)

  daily_df = this_episode_df.loc[this_episode_df['true_hour'].isin([24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336]), ['name', 'title', 'interval_time', 'true_hour', 'topsnap_views']].reset_index().drop(columns=['index'])
  daily_df = daily_df.loc[daily_df['true_hour'] <= fcst_hour]

  channel_alleps = benchmarks[benchmarks['name'].isin(this_episode_df.name)]

  channel_avg_list = []
  day_list = []

  for index, row in daily_df.iterrows():
    channel_avg = round(int(channel_alleps.loc[channel_alleps['true_hour']<= 168, ['topsnap_daily_diff']].mean()))

    if daily_df['true_hour'].values[index] == 24:
      day = 'Day 1'
    elif daily_df['true_hour'].values[index] == 48:
      day = 'Day 2'
    elif daily_df['true_hour'].values[index] == 72:
      day = 'Day 3'
    elif daily_df['true_hour'].values[index] == 96:
      day = 'Day 4'
    elif daily_df['true_hour'].values[index] == 120:
      day = 'Day 5'
    elif daily_df['true_hour'].values[index] == 144:
      day = 'Day 6'
    elif daily_df['true_hour'].values[index] == 168:
      day = 'Day 7'
    elif daily_df['true_hour'].values[index] == 192:
      day = 'Day 8'
    elif daily_df['true_hour'].values[index] == 216:
      day = 'Day 9'
    elif daily_df['true_hour'].values[index] == 240:
      day = 'Day 10'
    elif daily_df['true_hour'].values[index] == 264:
      day = 'Day 11'
    elif daily_df['true_hour'].values[index] == 288:
      day = 'Day 12'
    elif daily_df['true_hour'].values[index] == 312:
      day = 'Day 13'
    elif daily_df['true_hour'].values[index] == 336:
      day = 'Day 14'

    channel_avg_list.append(channel_avg)
    day_list.append(day)

  join_df = pd.DataFrame({'Day': day_list,
                        'Channel Daily Episode Avg': channel_avg_list
                         })

  rough_df = daily_df.join(join_df)
  rough_df['topsnap_lag'] = rough_df['topsnap_views'].shift(1).fillna(0)
  rough_df['daily_performance'] = rough_df['topsnap_views'] - rough_df['topsnap_lag']
  rough_df['prior_day'] = rough_df['daily_performance'].shift(1)
  rough_df['momentum'] = round((rough_df['daily_performance'] - rough_df['prior_day']) / (rough_df['prior_day']), 2)

  momentum_df = pd.DataFrame({'Channel': rough_df['name'],
                            'Episode': rough_df['title'],
                            'Interval Time': rough_df['interval_time'],
                            'Day': rough_df['Day'],
                            'Hour': rough_df['true_hour'],
                            'Daily Performance': rough_df['daily_performance'],
                            'Momentum %': rough_df['momentum'],
                            'Daily Channel Avg': rough_df['Channel Daily Episode Avg']
                         })

  #Formatting
  momentum_df['Momentum %'] = momentum_df['Momentum %'].map("{:,.2%}".format).replace('nan%', np.nan)
  momentum_df['Daily Performance'] = momentum_df['Daily Performance'].map("{:,.0f}".format)
  momentum_df['Daily Channel Avg'] = momentum_df['Daily Channel Avg'].map("{:,.0f}".format)

  return momentum_df


#Changepoint function
def changepoint_df(choose_episode):
  #Identify episode
  story_id = choose_episode

  channel_df = df[df.story_id.isin([story_id])]
  history = channel_df.loc[channel_df['forecast_type'] == 'history']

  #Create differences df
  differences = channel_df.loc[:, ['story_id', 'name','title', 'interval_time', 'topsnap_views', 'forecast_type', 'true_hour']]
  differences['lag'] = differences['topsnap_views'].shift(+1).fillna(0)
  differences['topsnap_diff'] = differences['topsnap_views'] - differences['lag']

  #Today
  tt = pd.Timestamp.today().floor('H')
  np_today = tt.to_numpy()
  today = np_today - np.timedelta64(4, 'h')

  last_hour = int(differences.loc[differences['interval_time'] == today, ['true_hour']].values[0])
  window = round_to_multiple(last_hour, 24)

  current = differences.loc[differences['true_hour'] <= window, ['interval_time', 'topsnap_views', 'topsnap_diff', 'true_hour']]

  #PELT Change point analysis
  ts  = np.array(current['topsnap_diff'])
  #Detect the change points
  algo = rpt.Pelt(model="rbf").fit(ts)
  change_location = algo.predict(pen=6)

  #Identify true hours for change detection
  true_hour_changes = []

  for change in change_location:
    c_true_hour = current.iloc[(change-1), 3]
    true_hour_changes.append(c_true_hour)

  #Conditional logic to create new field identifying change detection or not
  current['change_detection'] = np.select(
      [((current['true_hour'].isin(true_hour_changes[:-1])).astype('bool')),
      ((~current['true_hour'].isin(true_hour_changes[:-1])).astype('bool'))],
      ['change detected', np.nan],
      default=np.nan
  )

  #Rolling 12-window averages (preceding and following) to compare actual changes and determine direction of change
  window_size = 12
  current['rolling_10_preceding'] = current['topsnap_diff'].rolling(window_size, min_periods=1).mean()
  current['rolling_10_following'] = current['topsnap_diff'].rolling(window_size, min_periods=1).mean().shift(-window_size+1)[::-1]

  #Conditional logic to identify the direction of the change
  current['direction'] = np.select(
      [(current['change_detection'].isin(['change detected']))
      & ((current['rolling_10_following'])>=(current['rolling_10_preceding'])),

      (current['change_detection'].isin(['change detected']))
      & ((current['rolling_10_following'])<(current['rolling_10_preceding'])),

      (~current['change_detection'].isin(['change detected']))
      ],

      ['upward change',
        'downward change',
        np.nan],

        default=np.nan
  )

  change_df = current.loc[current['true_hour'] > 18]

  return change_df

#Summary Table
def summary_table():
  id_list = []
  episode_list = []
  channel_list = []

  last_reported_list = []
  hours_running = []
  actual_list = []
  actual_bench_list = []
  actual_trend_list = []

  fcst_hours_list = []
  fcst_views_list = []
  fcst_bench_list = []
  fcst_trend_list = []

  ctr_list = []

  trend_sentiment_list = []
  momentum_list = []

  daily_perf_list = []
  daily_avg_list = []

  for story in df.story_id.unique():
    channel_df = df[df.story_id.isin([story])]
    historical = channel_df[channel_df['forecast_type'].isin(['history'])]

    #story ID
    id = channel_df.story_id.values[0]
    id_list.append(id)
    #Episode
    episode = channel_df.title.values[0]
    episode_list.append(episode)
    #Channel
    channel = channel_df.name.values[0]
    channel_list.append(channel)

    #Get today's value, and the difference between the last reported hour and the current hour for conditional logic

    #Today
    tt = pd.Timestamp.today().floor('H')
    np_today = tt.to_numpy()
    today = np_today - np.timedelta64(4, 'h')

    #Published date
    published = channel_df['published_at'].head(1).values[0]

    #Last reported datetime
    lst_actual_dt = int(channel_df.loc[channel_df['forecast_type'] == 'history', ['interval_time']].tail(1).values[0])
    lst_actual_dt = pd.to_datetime(lst_actual_dt)

    #Difference between last reported time and current time
    difference = today - lst_actual_dt
    hours_diff = int(difference/ np.timedelta64(1, 'h'))

    #Hours running, last hour reported and performance

    #Last actual if there isn't enough data for forecasting OR there is long delays in data reporting
    if len(historical) < 10 or len(channel_df.loc[channel_df['forecast_type'] == 'future']) <=0:
      last_hour = int(channel_df.loc[channel_df['forecast_type'] == 'history', ['true_hour']].tail(1).values[0])
      last_reported = np.nan

      actual_views = int(channel_df.loc[channel_df['forecast_type'] == 'history', ['topsnap_views']].tail(1).values[0])

    #Metrics if there is long delays in data
    elif hours_diff > 72:
      actual_diff = today - published
      last_hour = int(actual_diff / np.timedelta64(1, 'h'))
      last_reported = int(channel_df.loc[channel_df['forecast_type'] == 'history', ['true_hour']].tail(1).values[0])

      actual_views = int(channel_df.loc[channel_df['forecast_type'] == 'history', ['topsnap_views']].tail(1).values[0])

    #ACTUAL current time and its corresponding hour
    else:
      tt = pd.Timestamp.today().floor('H')
      np_today = tt.to_numpy()
      today = np_today - np.timedelta64(4, 'h')

      last_hour = int(channel_df.loc[channel_df['interval_time'] == today, ['true_hour']].values[0])
      try:
        last_reported = round(channel_df.loc[:, ['actual', 'true_hour']].dropna().tail(1)['true_hour'].values[0])
      except IndexError:
        last_reported = np.nan
      actual_views = float(channel_df.loc[channel_df['true_hour'] == last_hour, ['topsnap_views']].values[0])

    #Append variables outside if/then logic
    last_reported_list.append(last_reported)
    hours_running.append(last_hour)
    actual_list.append(actual_views)

    #Actual benchmark
    channel_alleps = benchmarks[benchmarks['name'].isin(channel_df.name)]
    channel_hour = channel_alleps.loc[channel_alleps['true_hour'] == last_hour, ['topsnap_views_total']]
    actual_bench = channel_hour['topsnap_views_total'].mean()
    try:
      actual_bench = round(actual_bench)
    except ValueError:
      actual_bench = actual_bench
    actual_bench_list.append(actual_bench)
    #Actual Trend
    try:
      trending_actual = round(((actual_views - actual_bench) / actual_bench), 2)
    except ValueError:
      trending_actual = ((actual_views - actual_bench) / actual_bench)
    actual_trend_list.append(trending_actual)

    #Forecasted hours
    fcst_hours = round_to_multiple(last_hour, 24)
    fcst_hours_list.append(fcst_hours)

    #Forecasted benchmark
    channel_fcst_hour = channel_alleps.loc[channel_alleps['true_hour'] == fcst_hours, ['topsnap_views_total']]
    fcst_bench = channel_fcst_hour['topsnap_views_total'].mean()
    try:
      fcst_bench = round(fcst_bench)
    except ValueError:
      fcst_bench = fcst_bench
    fcst_bench_list.append(fcst_bench)

    if len(historical) < 10 or len(channel_df.loc[channel_df['forecast_type'] == 'future']) <=0 or hours_diff > 72:
      fcst_views = np.nan
      trending = np.nan
      momentum = np.nan
      daily_prediction = np.nan
      if len(channel_alleps['topsnap_daily_diff'].dropna()) == 0:
        daily_avg = 0
      else:
        daily_avg = round(int(channel_alleps.loc[channel_alleps['true_hour']<= 168, ['topsnap_daily_diff']].mean()))

    else:
      #Forecasted topsnaps
      try:
        fcst_views = int(channel_df.loc[channel_df['true_hour'] == fcst_hours, ['topsnap_views']].values[0])
      except IndexError:
        fcst_views = np.nan

      #Fcst Trend
      try:
        trending = round(((fcst_views - fcst_bench) / fcst_bench), 4)
      except ValueError:
        trending = (fcst_views - fcst_bench) / fcst_bench

      #Momentum
      current_df = channel_df.loc[channel_df.true_hour <= fcst_hours]
      momentum_df = current_df.loc[current_df['true_hour'] >= fcst_hours-48]
      #momentum_df =current_df.tail(49)

      if fcst_hours <= 24:
        momentum = np.nan
        daily_prediction = np.nan
        if len(channel_alleps['topsnap_daily_diff'].dropna()) == 0:
          daily_avg = 0
        else:
          daily_avg = round(int(channel_alleps.loc[channel_alleps['true_hour']<= 120, ['topsnap_daily_diff']].mean()))
      else:
        start_end = current_df.tail(25)
        start = start_end.head(1)['topsnap_views'].values[0]
        end = start_end.tail(1)['topsnap_views'].values[0]
        daily_prediction = round(end-start)
        if len(channel_alleps['topsnap_daily_diff'].dropna()) == 0:
          daily_avg = 1
        else:
          daily_avg = round(int(channel_alleps.loc[channel_alleps['true_hour']<= 120, ['topsnap_daily_diff']].mean()))

        if fcst_hours == 48:
          previous_day = float(momentum_df.loc[momentum_df['true_hour'] == fcst_hours-24, ['topsnap_views']].head(1).values[0])
        else:
          m_start = momentum_df.head(1)['topsnap_views'].values[0]
          m_end = float(momentum_df.loc[momentum_df['true_hour'] == fcst_hours-24, ['topsnap_views']].head(1).values[0])
          previous_day = round(m_end-m_start)

        try:
          momentum = round((daily_prediction-previous_day)/previous_day, 2)
        except ZeroDivisionError:
          momentum = 0

    #Trend Sentiment
    if channel_df['true_hour'].values[-1] < 24:
      sentiment = np.nan

    else:
      cpd = changepoint_df(story)
      sentiment = cpd.loc[cpd['interval_time']<= today]
      last_36 = sentiment[-48:]

      last_36['ranking'] = last_36.loc[last_36['direction'].isin(['upward change', 'downward change'])].groupby('direction')['interval_time'].rank(method='dense', ascending=False)

      try:
        most_recent = str(last_36.loc[last_36['ranking'] == 1, ['direction']].values[-1])
        if most_recent == "['upward change']":
          sentiment = '🔥'

        if most_recent == "['upward change']" and daily_prediction > (daily_avg*1.5):
          sentiment = '🔥🔥'

        if most_recent == "['upward change']" and daily_prediction > (daily_avg*2):
          sentiment = '🔥🔥🔥'

        if most_recent == "['downward change']":
          sentiment = '🥶'

        if most_recent == "['downward change']" and daily_prediction < (daily_avg*0.5):
          sentiment = '🥶🥶'

        if most_recent == "['downward change']" and daily_prediction < (daily_avg*0.25):
          sentiment = '🥶🥶🥶'

      except IndexError:
        sentiment = np.nan

    #Append remaining variables outside of if/else statement
    fcst_views_list.append(fcst_views)
    fcst_trend_list.append(trending)
    trend_sentiment_list.append(sentiment)
    momentum_list.append(momentum)
    daily_perf_list.append(daily_prediction)
    daily_avg_list.append(daily_avg)

  final_df = pd.DataFrame({'Story ID': id_list,
                         'Channel': channel_list,
                         'Episode': episode_list,
                         "Last Reported Hour": last_reported_list,
                         'Current Hour': hours_running,
                         'Current Performance': actual_list,
                         "Current Benchmark": actual_bench_list,
                         "% vs Bench": actual_trend_list,
                         'Fcst Period': fcst_hours_list,
                         'Forecast': fcst_views_list,
                         'Fcst Benchmark': fcst_bench_list,
                         'Fcst % vs Bench': fcst_trend_list,
                         'Trend Sentiment': trend_sentiment_list,
                         'Momentum %': momentum_list,
                         'Daily Performance': daily_perf_list,
                         'Daily Avg': daily_avg_list
                         })

  #Create Decision logic
  final_df['Consideration'] = np.select(
    [   #Let It Ride
        (~final_df['Channel'].isin(["Channels of Choice"]))
        &(final_df['Fcst Period']==48)
        &(final_df['Fcst % vs Bench']>=0.75)
        &((final_df['Daily Performance'])>(final_df['Daily Avg']*0.6))
        #48 hours
        |(~final_df['Channel'].isin(["Channels of Choice"]))
        &(final_df['Fcst Period']==48)
        &(final_df['Fcst % vs Bench']>0.5)
        &((final_df['Daily Performance'])>=(final_df['Daily Avg']*1.5))

        #72 hours
        |(final_df['Fcst Period']==72)
        &(final_df['Fcst % vs Bench']>=0.75)
        &((final_df['Daily Performance'])>=(final_df['Daily Avg']*0.8))
        #72 hours
        |(final_df['Fcst Period']==72)
        &(final_df['Fcst % vs Bench']>=0.5)
        &(final_df['Trend Sentiment']== '🔥🔥')
        |(final_df['Fcst Period']>=72) &(final_df['Fcst Period']<=96)
        &(final_df['Trend Sentiment']== '🔥🔥🔥')

        #96 hours
        |(final_df['Fcst Period']==96)
        &(final_df['Fcst % vs Bench']>=1.0)
        &((final_df['Daily Performance'])>=(final_df['Daily Avg']*0.9))
        #96 hours
        |(final_df['Fcst Period']==96)
        &(final_df['Fcst % vs Bench']>0.5)
        &(final_df['Trend Sentiment']== '🔥🔥🔥')

        #120 and 168
        |(final_df['Fcst Period']>=120) & (final_df['Fcst Period']<=168)
        &(final_df['Fcst % vs Bench']>=1.0)
        &((final_df['Daily Performance'])>=(final_df['Daily Avg']))
        &(final_df['Trend Sentiment']!= '🥶')
        #120 to 168 
        |(final_df['Fcst Period']>=120) & (final_df['Fcst Period']<=168)
        &((((final_df['Daily Performance']) - (final_df['Daily Avg'])) / (final_df['Daily Avg'])) >= 1.0)
        #120-168
        |(final_df['Fcst Period']>=120) & (final_df['Fcst Period']<=168)
        &(final_df['Trend Sentiment']== '🔥🔥🔥')

        #168 and 192 hours 
        |(final_df['Fcst Period']>168) & (final_df['Fcst Period']<=192)
        &(final_df['Fcst % vs Bench']>=1.5)
        &((final_df['Daily Performance'])>(final_df['Daily Avg']))

        #192
        |(final_df['Fcst Period']==192)
        &(final_df['Fcst % vs Bench']>=2.0)
        &((final_df['Daily Performance'])>(final_df['Daily Avg']))

        #216 and 240
        |(final_df['Fcst Period']>=216) & (final_df['Fcst Period']<=240)
        &(final_df['Fcst % vs Bench']>=3.0)
        &((final_df['Daily Performance'])>(final_df['Daily Avg']))
    
        #240
        |(final_df['Fcst Period']>240) & (final_df['Fcst % vs Bench']>=4.0)
        &((final_df['Daily Performance'])>(final_df['Daily Avg'])),




        #Replace It

        #48 hours
        (final_df['Fcst Period']== 48)&(final_df['Fcst % vs Bench']<= -0.5)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']))
        #48
        |(final_df['Fcst Period']== 48)&(final_df['Fcst % vs Bench']<= -0.75)
        #48
        |(final_df['Fcst Period']== 48)&(final_df['Fcst % vs Bench'] < 0)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']*0.8))

        #72hrs
        |(final_df['Fcst Period']== 72)&(final_df['Fcst % vs Bench']< 0)
        #72
        |(final_df['Fcst Period']== 72)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']*0.75))
        #72
        |(final_df['Fcst Period']== 72)&(final_df['Fcst % vs Bench']< 0.25)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']*0.9))

        #96hrs
        |(final_df['Fcst Period']== 96)&(final_df['Fcst % vs Bench']< 0.5)
        &((final_df['Daily Performance']) < (final_df['Daily Avg']*1.5))
        #96
        |(final_df['Fcst Period']== 96)
        &(final_df['Trend Sentiment']== '🥶🥶🥶')
        #96
        |(final_df['Fcst Period']== 96)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']*0.9))
        #96
        |(final_df['Fcst Period']== 96)&(final_df['Fcst % vs Bench']< 0.75)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']))

        #120-168hrs
        |(final_df['Fcst Period']>= 120)&(final_df['Fcst Period']<= 168)
        &(final_df['Fcst % vs Bench']<= 0.75)
        #120-168
        |(final_df['Fcst Period']>= 120)&(final_df['Fcst Period']<= 168)
        &(final_df['Trend Sentiment']== '🥶🥶')
        |(final_df['Fcst Period']>= 120)&(final_df['Fcst Period']<= 168)
        &(final_df['Trend Sentiment']== '🥶🥶🥶')
        #120-168
        |(final_df['Fcst Period']>= 120)&(final_df['Fcst Period']<= 192)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']))

        192hrs
        |(final_df['Fcst Period'] == 192)
        &(final_df['Fcst % vs Bench'] <2.0)
        #192 
        |(final_df['Fcst Period'] == 192)
        &((final_df['Daily Performance'])<(final_df['Daily Avg']))

        #216-240hrs
        |(final_df['Fcst Period'] >= 216) &(final_df['Fcst Period'] <= 240)
        &(final_df['Fcst % vs Bench'] < 3.0)
        #216hrs+
        |(final_df['Fcst Period'] >= 216)
        &((final_df['Daily Performance']) < (final_df['Daily Avg']))
        #264+
        |(final_df['Fcst Period'] >= 264)
        &(final_df['Fcst % vs Bench'] < 4.0)
        ,




      # Investigate - Bullish

      #48 hours 
      (~final_df['Channel'].isin(['Channels of Choice']))
      &(final_df['Fcst Period']==48)
      &(final_df['Fcst % vs Bench']>=0.5)
      &(final_df['Trend Sentiment']== '🔥')
      #48 
      |(~final_df['Channel'].isin(['Channels of Choice']))
      &(final_df['Fcst Period']==48)
      &(final_df['Fcst % vs Bench']> 0) &(final_df['Fcst % vs Bench']< 0.5)
      &((final_df['Daily Performance'])>=(final_df['Daily Avg']*1.5))
      #48 
      |(~final_df['Channel'].isin(['Channels of Choice']))
      &(final_df['Fcst Period']==48)
      &(final_df['Fcst % vs Bench']>=0.25) & (final_df['Fcst % vs Bench']<=0.5)
      &(final_df['Momentum %'] >= 0.5)
      #48
      |(~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
      &(final_df['Fcst Period']==48)
      &(final_df['Fcst % vs Bench']>0) & (final_df['Fcst % vs Bench']<=0.25)
      &(final_df['Momentum %'] >= 0.75)

      #72hrs
      |(final_df['Fcst Period']==72)
      &(final_df['Fcst % vs Bench']>=0.5) & (final_df['Fcst % vs Bench']<=0.75)
      &((final_df['Daily Performance'])>=(final_df['Daily Avg']*1.75))
      #72
      |(final_df['Fcst Period']==72)
      &(final_df['Fcst % vs Bench']>= 0.5) & (final_df['Fcst % vs Bench']<0.75)
      &(final_df['Trend Sentiment']== '🔥')
      #72
      |(final_df['Fcst Period']==72)
      &(final_df['Fcst % vs Bench']> 0) & (final_df['Fcst % vs Bench']<=0.5)
      &(final_df['Trend Sentiment']== '🔥🔥')

      #96hrs
      |(final_df['Fcst Period'] == 96)
      &(final_df['Fcst % vs Bench']>= 0.75) &(final_df['Fcst % vs Bench']<= 1.0)
      &(final_df['Trend Sentiment']== '🔥🔥')
      |(final_df['Fcst Period'] == 96)
      &(final_df['Fcst % vs Bench']>= 0.75) &(final_df['Fcst % vs Bench']<= 1.0)
      &(final_df['Trend Sentiment']== '🔥')
      #96
      |(final_df['Fcst Period']==96)
      &(final_df['Fcst % vs Bench']>= 0.75) & (final_df['Fcst % vs Bench']< 1.0)
      &(final_df['Trend Sentiment']== '🔥')
      |(final_df['Fcst Period']==96)
      &(final_df['Fcst % vs Bench']>= 0.75) & (final_df['Fcst % vs Bench']< 1.0)
      &(final_df['Momentum %'] > 0)
      #96
      |(final_df['Fcst Period']==96)
      &(final_df['Fcst % vs Bench']>0.25)
      &(final_df['Trend Sentiment']== '🔥🔥🔥')
      |(final_df['Fcst Period']==96)
      &(final_df['Fcst % vs Bench']>0.25)
      &(final_df['Trend Sentiment']== '🔥🔥')
      |(final_df['Fcst Period']==96)
      &(final_df['Fcst % vs Bench']>0.25)
      &((final_df['Daily Performance'])>=(final_df['Daily Avg']*1.5))
      #120-168
      |(final_df['Fcst Period']>=120) &(final_df['Fcst Period']<=168)
      &(final_df['Fcst % vs Bench']>=0.75) & (final_df['Fcst % vs Bench']<1.0)
      &((((final_df['Daily Performance']) - (final_df['Daily Avg'])) / (final_df['Daily Avg'])) >= 0.5)

      #120-168hrs
      |(final_df['Fcst Period']>=120) &(final_df['Fcst Period']<=168)
      &(final_df['Fcst % vs Bench']>=0.75) & (final_df['Fcst % vs Bench']<1.0)
      &(final_df['Trend Sentiment']== '🔥')
      |(final_df['Fcst Period']>=120) &(final_df['Fcst Period']<=168)
      &(final_df['Fcst % vs Bench']>=0.75) & (final_df['Fcst % vs Bench']<1.0)
      &(final_df['Trend Sentiment']== '🔥🔥')
      #192hrs+
      |(final_df['Fcst Period']==192)
      &(final_df['Fcst % vs Bench'] >= 1.5)&(final_df['Fcst % vs Bench'] < 2.0)
      &((final_df['Daily Performance'])>=(final_df['Daily Avg']*1.5)),





     #Investigate - Bearish
     #48 hrs
     (~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period']==48)
     &(final_df['Fcst % vs Bench']< 0) & (final_df['Fcst % vs Bench']> -0.75)
     &((final_df['Daily Performance']) < (final_df['Daily Avg']*1.25))
     #48
     |(~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period']==48)
     &(final_df['Fcst % vs Bench']< 0.75) & (final_df['Fcst % vs Bench']>= -0.25)
     &(final_df['Trend Sentiment']== '🥶')
     |(~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period']==48)
     &(final_df['Fcst % vs Bench']< 0.75) & (final_df['Fcst % vs Bench']>= -0.25)
     &((final_df['Daily Performance'])<= (final_df['Daily Avg']*0.8))
     #48
     |(~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period']==48)
     &(final_df['Fcst % vs Bench']>= 0.75)
     &((final_df['Daily Performance'])<= (final_df['Daily Avg']*0.6))
     #48
     |(~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period']==48)
     &(final_df['Trend Sentiment']== '🥶🥶')
     |(~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period']==48)
     &(final_df['Trend Sentiment']== '🥶🥶🥶')

     #72
     |(final_df['Fcst Period']==72)
     &(final_df['Fcst % vs Bench']< 0.25) & (final_df['Fcst % vs Bench']>= -0.25)
     #72
     |(final_df['Fcst Period']==72)
     &(final_df['Fcst % vs Bench']> 0) &(final_df['Fcst % vs Bench']< 0.5)
     &((final_df['Daily Performance'])<(final_df['Daily Avg']))
     #72
     |(final_df['Fcst Period']==72)
     &(final_df['Trend Sentiment']== '🥶🥶')
     #72hrs
     |(final_df['Fcst Period']==72)
     &((final_df['Daily Performance'])<(final_df['Daily Avg']*0.75))
     #72-96hrs
     |(final_df['Fcst Period']>=72) &(final_df['Fcst Period']<=96)
      &(final_df['Trend Sentiment']== '🥶')
      &((final_df['Daily Performance'])<(final_df['Daily Avg']))

      #96hrs
     |(final_df['Fcst Period']==96)
     &((final_df['Daily Performance'])<(final_df['Daily Avg']))
     #96hrs
     |(final_df['Fcst Period']==96)
     &(final_df['Fcst % vs Bench']>= 0.25) &(final_df['Fcst % vs Bench']<= 0.75)
     &(final_df['Trend Sentiment']== '🥶')
     |(final_df['Fcst Period']==96)
     &(final_df['Fcst % vs Bench']>= 0.25) &(final_df['Fcst % vs Bench']<= 0.75)
     &((final_df['Daily Performance'])<(final_df['Daily Avg']))

     #120-168hrs
     |(final_df['Fcst Period']>=120) & (final_df['Fcst Period']<=168)
     &(final_df['Trend Sentiment']== '🥶')
     #120-168hrs
     |(final_df['Fcst Period']>=120) & (final_df['Fcst Period']<=168)
     &(final_df['Fcst % vs Bench']>= 1.0)
     &((final_df['Daily Performance'])<(final_df['Daily Avg']))
     #120-168hrs
     |(final_df['Fcst Period']>=120) & (final_df['Fcst Period']<=168)
     &(final_df['Fcst % vs Bench']>= 0.75) &(final_df['Fcst % vs Bench']<= 1.0)
     &(final_df['Momentum %'] <= 0)

      #168-192hrs
      |(final_df['Fcst Period']>168) & (final_df['Fcst Period']<=192)
      &(final_df['Fcst % vs Bench'] >= 1.5)
      &(final_df['Momentum %'] <= -0.25),





     #Investigate - Average
     #48 hrs
     (~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period'] == 48)
     &(final_df['Fcst % vs Bench'] >= 0) &(final_df['Fcst % vs Bench'] <= 0.75)
     #48 hrs
     |(~final_df['Channel'].isin(['Channels of Choice']))
     &(final_df['Fcst Period'] == 48)
     &(final_df['Fcst % vs Bench'] >= -0.25) &(final_df['Fcst % vs Bench'] < 0)
     &((final_df['Daily Performance'])>=(final_df['Daily Avg']*1.25))

     #72hrs
     |(final_df['Fcst Period']==72)
     &(final_df['Fcst % vs Bench'] >= 0.25)&(final_df['Fcst % vs Bench'] < 0.75)

     #96hrs
     |(final_df['Fcst Period']==96)
     &(final_df['Fcst % vs Bench'] >= 0.5)&(final_df['Fcst % vs Bench'] < 1.0)

     #120-168hrs
     |(final_df['Fcst Period']>=120)&(final_df['Fcst Period']<=168)
     &(final_df['Fcst % vs Bench']>= 0.75) & (final_df['Fcst % vs Bench']< 1.0)

    ],

    ['Let It Ride',
      'Replace It',
     'Investigate - Bullish',
     'Investigate - Bearish',
     'Investigate - Average'
    ],
    default='No Decision'
  )

  df_order = ['Story ID',
            'Channel',
            'Episode',
            'Consideration',
            'Last Reported Hour',
            'Current Hour',
            'Current Performance',
            'Current Benchmark',
             '% vs Bench',
             'Fcst Period',
             'Forecast',
             'Fcst Benchmark',
            'Fcst % vs Bench',
            'Trend Sentiment',
            'Momentum %',
            'Daily Performance',
            'Daily Avg']
            #'Test CTR(%)']

  #Create summary df, sort and reset index
  summary_df = final_df[df_order].sort_values(['Fcst % vs Bench'], ascending=False)
  summary_df = summary_df.reset_index().drop(columns=['index'])

  return summary_df



#Data Functions

#Evaluate Model
@st.cache_data(ttl=3600)
def evaluate_model():
  eval = ('''SELECT story_id_fixed as story_id,
                    pub.published_at,
                    pub.account_name AS name,
                    pub.title,
                    --arima.*
                    arima.has_drift,
                    arima.AIC,
                    arima.non_seasonal_p as p,
                    arima.non_seasonal_d as d,
                    arima.non_seasonal_q AS q,
                    arima.variance,
                    arima.seasonal_periods
              FROM ML.EVALUATE(MODEL`sample_database.dynamic_scheduling.arima_dynamic_scheduling`) arima
              LEFT JOIN (SELECT account_id,
                                post_id,
                                account_name,
                                title,
                                datetime(published_at, "America/Toronto") AS published_at
                          FROM `sample_database.Views.snap_publisher_profile`
                          ) AS pub
              ON arima.story_id_fixed = pub.post_id
              WHERE pub.account_name NOT IN ('Excluded Channels')
              ORDER BY arima.AIC ASC;''')

  eval_model = pd.read_gbq(eval, credentials = credentials)
  return eval_model

#ARIMA df
@st.cache_data(ttl=900)
def get_data():
  arima_query_test = ('''
    WITH cte AS
      (SELECT story_id_fixed story_id,
          datetime(forecast_timestamp) interval_time,
          forecast_value topsnap_views,
          confidence_interval_lower_bound,
          confidence_interval_upper_bound,
          CASE WHEN story_id_fixed IS NOT NULL THEN 'future'
            END AS forecast_type
      FROM ML.FORECAST(MODEL `sample_database.dynamic_scheduling.arima_dynamic_scheduling`,
                                    STRUCT(72 AS horizon, 0.80 AS confidence_level))
      UNION ALL
      SELECT story_id_fixed story_id,
            datetime(time_series_timestamp) interval_time,
            time_series_data topsnap_views,
            prediction_interval_lower_bound confidence_interval_lower_bound,
            prediction_interval_upper_bound confidence_interval_upper_bound,
            time_series_type forecast_type
        FROM (SELECT *
              FROM ML.EXPLAIN_FORECAST(MODEL `sample_database.dynamic_scheduling.arima_historical`,
                                        STRUCT(24 AS horizon, 0.8 AS confidence_level)) AS hist
              WHERE time_series_type in ('history')) hist
        ORDER BY story_id, interval_time ASC
        ),
    cte_2 AS
    (
    SELECT   pub.account_name AS name,
            pub.title,
            datetime(pub.published_at, "America/Toronto") published_at,
            cte.story_id,
            cte.interval_time,
            cte.topsnap_views,
            cte.confidence_interval_lower_bound,
            cte.confidence_interval_upper_bound,
            cte.forecast_type,
            actuals.views AS actual,
            CASE WHEN forecast_type in ('history') THEN cte.topsnap_views
                END AS historical_fcst,
            CASE WHEN forecast_type in ('future') THEN cte.topsnap_views
                END AS future_fcst,
            DATETIME_DIFF(cte.interval_time, datetime(pub.published_at, "America/Toronto"), HOUR) AS true_hour
    FROM cte
    LEFT JOIN (SELECT post_id,
                      account_name,
                      title,
                      published_at
                FROM `sample_database.ingest.post_details`) AS pub
    ON (cte.story_id = pub.post_id)
    LEFT JOIN (SELECT datetime(DATE_TRUNC(interval_time, HOUR)) AS interval_rounded,
                        interval_time,
                        post_id,
                        views,
                        datetime(published_at) published_at
                  FROM `sample_database.Views.snap_posts_rt`
                  WHERE TRUE
                  AND EXTRACT(MINUTE FROM interval_time) = 0
                ) AS actuals
    ON (cte.story_id = actuals.post_id) AND (cte.interval_time = datetime(actuals.interval_rounded))
    WHERE pub.account_name NOT IN ('Excluded Channels')
    ORDER BY story_id, interval_time ASC
    )
    SELECT *
    FROM cte_2
    UNION ALL
    SELECT realtime.*
    FROM
          (SELECT v.publisher_name AS name,
                v.title,
                v.published_at,
                v.post_id AS story_id,
                DATE_TRUNC(v.interval_time, HOUR) interval_time,
                v.views AS topsnap_views,
                CASE WHEN post_id IS NOT NULL THEN NULL
                      END AS confidence_interval_lower_bound,
                CASE WHEN post_id IS NOT NULL THEN NULL
                      END AS confidence_interval_upper_bound,
                CASE WHEN post_id IS NOT NULL THEN 'history'
                      END AS forecast_type,
                v.views AS actual,
                CASE WHEN post_id IS NOT NULL THEN NULL
                      END AS historical_fcst,
                CASE WHEN post_id IS NOT NULL THEN NULL
                      END AS future_fcst,
                DATETIME_DIFF(v.interval_time, v.published_at, HOUR) AS true_hour
          FROM `sample_database.Views.snap_posts_rt` AS v
          WHERE TRUE
          AND EXTRACT(MINUTE FROM interval_time) = 0
          AND v.post_id in (SELECT story_id_fixed
                            FROM `sample_database.dynamic_scheduling.dynamic_scheduling_episodes`
                            WHERE story_id_fixed IS NOT NULL
                            GROUP BY story_id_fixed)
          AND v.post_id NOT IN (SELECT story_id
                                  FROM cte_2
                                  GROUP BY story_id)
          AND v.publisher_name NOT IN ('Excluded Channels')
          )realtime
    ORDER BY story_id, interval_time ASC;
  ''')

  df = pd.read_gbq(arima_query_test, credentials = credentials)
  return df

#Channel data (benchmarks)
@st.cache_data(ttl=43200)
def benchmark_data():
  sql_query2 = ('''WITH cte AS
                    (
                      SELECT *,
                            DATETIME_DIFF(interval_time, published_at, HOUR) AS true_hour
                      FROM `sample_database.Views.snap_posts_rt`
                      WHERE published_at >= current_date - 90
                      AND EXTRACT(MINUTE FROM interval_time) = 0
                    )
                    SELECT cte.publisher_name AS name,
                          cte.title,
                          cte.published_at,
                          cte.interval_time,
                          cte.post_id AS story_id,
                          cte.true_hour,
                          cte.views_diff AS topsnap_views_diff,
                          cte.views AS topsnap_views_total,
                          daily.topsnap_views_total - COALESCE(LAG(daily.topsnap_views_total) OVER (PARTITION BY daily.name, daily.story_id ORDER BY daily.true_hour), 0) topsnap_daily_diff
                    FROM cte
                    LEFT JOIN (SELECT publisher_name AS name,
                                      title,
                                      interval_time,
                                      post_id AS story_id,
                                      true_hour,
                                      views_diff AS topsnap_views_diff,
                                      views AS topsnap_views_total
                              FROM cte
                              WHERE true_hour in (1, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336)
                              ) daily
                    ON (cte.post_id = daily.story_id) AND (cte.true_hour = daily.true_hour)
                    ORDER BY story_id, interval_time ASC;''')

  benchmarks = pd.read_gbq(sql_query2, credentials = credentials)
  #benchmarks['best_test_ctr'] = benchmarks['best_test_ctr'].astype('float')
  benchmarks['topsnap_views_total'] = benchmarks['topsnap_views_total'].astype('float')

  return benchmarks


#Build Streamlit App

#Create sidebar
menu = ["Episode Summary", "Forecasting + Momentum", "Model Performance"]
choice = st.sidebar.selectbox("Menu", menu)

st.write("*Real-time data derived from a Google BigQuery database (hourly - cached every 30 minutes), and forecasting is powered by BigQuery's AutoARIMA model*")

if choice == 'Episode Summary':
    # Create dropdown-menu / interactive forecast graph
    st.write("# Episode Summary - Recent Episodes + Performance")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        * Click the 'View Summary Table' below to see metrics and forecast for the most recent episodes on all Snapchat channels
                        * "Considerations" are provided based on current performance, forecasts, timeliness and momentum. It is strongly recommended that you use this table in conjunction with the Topsnap Forecasting tab to make scheduling decisions.
                        * If forecast values and/or %'s render as NULL (nan), then there simply isn't enough data provided for that episode yet to make an accurate forecast - wait a couple of hours in this case to see forecast results!

                        **NOTE: Snap delays often lead to not having data on an episode until its 6th-8th hour on platform. If an episode that has been released is not appearing in the table, this is likely the reason. For episodes that do appear in the table, you can view reporting discrepancies by comparing the "Last Reported Hour" to the "Current Hour".**
                       """)

    df = get_data()
    benchmarks = benchmark_data()

    chart_button = st.button("View Summary Table")
    if chart_button:
      ag_df = summary_table()

      ag_df['% v Bench'] = ag_df['% vs Bench']
      ag_df['Fcst % v Bench'] = ag_df['Fcst % vs Bench']
      ag_df['Momentum%'] = ag_df['Momentum %']

      percentages = ['% v Bench', 'Fcst % v Bench', 'Momentum%']
      values = ['Current Performance', 'Current Benchmark', 'Forecast', 'Fcst Benchmark', 'Daily Performance', 'Daily Avg']

      for column in percentages:
        ag_df[column] = ag_df[column].map("{:,.2%}".format)
        ag_df[column] = ag_df[column].replace('nan%', np.nan)

      for column in values:
        ag_df[column] = ag_df[column].map("{:,.0f}".format)

      new_order = ['Story ID',
                  'Channel',
                  'Episode',
                  'Consideration',
                  'Last Reported Hour',
                  'Current Hour',
                  'Current Performance',
                  'Current Benchmark',
                  '% v Bench',
                  'Fcst Period',
                  'Forecast',
                  'Fcst Benchmark',
                  'Fcst % v Bench',
                  'Trend Sentiment',
                  'Momentum%',
                  'Daily Performance',
                  'Daily Avg',
                  #'Test CTR(%)',
                  '% vs Bench',
                  'Fcst % vs Bench',
                  'Momentum %'
                  ]
      ag_df = ag_df[new_order]

      #Formatting
      jscode = JsCode("""
            function(params) {
                if (params.data.Consideration === 'Let It Ride') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#BAFFC9'
                    }
                }
                if (params.data.Consideration === 'Investigate - Bullish') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#BAE1FF'
                    }
                }
                if (params.data.Consideration === 'Investigate - Bearish') {
                    return {
                        'color': 'white',
                        'backgroundColor': '#F4A460'
                    }
                }
                if (params.data.Consideration === 'Investigate - Average') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#FFFACD'
                    }
                }
                if (params.data.Consideration === 'Replace It') {
                    return {
                        'color': 'white',
                        'backgroundColor': '#FF6347'
                    }
                }
                if (params.data.Consideration === 'No Decision') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#F5F5F5'
                    }
                }
            };
            """)

      jsforecast = JsCode("""
            function (params) {

            if (params.data['Fcst % vs Bench'] >=1.0) {
                return {
                        'color': 'white',
                        'backgroundColor': '#00e673'
                    }
            }
            if (params.data['Fcst % vs Bench'] >=0.5) {
                return {
                        'color': 'black',
                        'backgroundColor': '#66ffb3'
                    }
            }
            if (params.data['Fcst % vs Bench'] > 0) {
                return {
                        'color': 'black',
                        'backgroundColor': '#BAFFC9'
                    }
            }
            if (params.data['Fcst % vs Bench'] >=-0.25) {
                return {
                        'color': 'black',
                        'backgroundColor': '#ffc2b3'
                    }
            }
            if (params.data['Fcst % vs Bench'] >=-0.8) {
                return {
                        'color': 'black',
                        'backgroundColor': '#ff8566'
                    }
            }
            if (params.data['Fcst % vs Bench'] < -0.8) {
                return {
                        'color': 'white',
                        'backgroundColor': '#ff471a'
                    }
            }
            };
                      """)

      gb = GridOptionsBuilder.from_dataframe(ag_df)
      gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
      gb.configure_side_bar() #Add a sidebar

      gb.configure_column('% vs Bench', hide=True)
      gb.configure_column('Fcst % vs Bench', hide=True)
      gb.configure_column('Momentum %', hide=True)
      #gb.configure_column('Test CTR(%)', hide=True)

      gb.configure_column('Fcst % v Bench', cellStyle=jsforecast)

      gridOptions = gb.build()
      gridOptions['getRowStyle'] = jscode

      grid_response = AgGrid(ag_df,
                            gridOptions=gridOptions,
                            allow_unsafe_jscode=True,
                            fit_columns_on_grid_load=True,
                            update_mode='NO_UPDATE',
                            width='100%')



if choice == 'Forecasting + Momentum':

    # Create dropdown-menu / interactive forecast graph
    st.write("# Forecasting Topsnaps")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        * Enter an episode's Story ID and select 'Momentum Table' to display an overview of the episode's performance and daily trends.
                        * In addition, choose an hourly window from the drop-down menu below and select 'Forecast Performance' to view an interactive chart displaying both the historic and predicted performance of the current episode.

                        **NOTE: Forecast windows are set so the maximum forecast range is 72 hours into the future from the current date - for the purpose of this tool, we generally only predict performance to the next 24-hour future window**
                       """)

    df = get_data()
    benchmarks = benchmark_data()

    #Choose an episode
    episode = st.text_input("Enter the Story ID here:", "")

    momentum_table = st.button('Momentum Table')
    if momentum_table:
      st.dataframe(momentum_chart(episode))

    hour_choices = {24: '24', 48: '48', 72: '72', 96: '96', 120:'120', 144:'144', 168:'168', 192:'192', 216:'216', 240:'240', 264:'264', 288:'288', 312:'312', 336:'336'}
    hours = st.selectbox("Select the hourly window you would like to forecast to", options=list(hour_choices.keys()), format_func = lambda x:hour_choices[x])

    forecast_total = st.button("Forecast Performance")
    if forecast_total:
      st.plotly_chart(forecast_totalview(episode, hours), use_container_width=True, theme=None)

if choice == "Model Performance":
    # Create dropdown-menu / interactive forecast graph
    st.write("# Validating Model Performance")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        **Current tab is for the use of the Data Dept only**

                        * Select 'Evaluate Model' button to display performance metrics for all ARIMA models
                        * AIC (Akaike Information Criteria) is the main metric offered to determine model selection (the lower the better)
                        * Models are also represented by p (AR term), d (differencing term), and q (MA term) for greater interpretation
                       """)
    evaluate = st.button("Evaluate Model")
    if evaluate:
      st.dataframe(evaluate_model())
