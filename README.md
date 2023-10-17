# Snapchat Dynamic Scheduling 
## Overview 
*This is a replication of a deprecated real-world project that has been edited for the suitability of a github repo*

Using Auto-ARIMA timeseries modelling to forecast social media video viewership performance. PELT change point detection (ruptures library) is layered on the forecast to identify major changes in trends in reference to actual performance + the model's prediction (offline change detection updated as the timeseries model updates). Models are deployed via Streamlit Web-app, providing real-time analytics and scheduling recommendations that update hourly.



## Purpose 
*This is a replication of a deprecated real-world project that has been edited for the suitability of a github repo*

Through previous analysis, it has been discovered that episodes on a social media platform (such as Snapchat) have their performance thwarted with the following episode's release on channel - with this, scheduling content to reflect performance rather than a weekly set schedule becomes an area of interest, i.e dynamically scheduling episodes to give longer running time to high performers, and cutting off time on platform for low performers.  

Predicting the future performance of an episode at incremental periods is valuable in that it can inform these dynamic scheduling decisions, especially when compared against benchmarks on channel. Furthermore, identifying major changes in trend is useful when looking to make real-time, performance-based scheduling decisions. 

The purpose of this web-app (as an analytics tool) is to help augment and expedite this data-informed process that may otherwise require rigorous analysis daily.



## ARIMA
Auto Regressive Integrated Moving Average using auto regression (p - predicting current value based on previous time periods' values), differencing (d), and error (q) to make future predictions is a widley used statistical technique in timeseries forecasting. The final version of the dynamic scheduling tool leverages BigQuery ML's Auto-ARIMA functionality to make non-seasonal predictions of video viewership performance in hourly intervals. 

Typically ARIMA models are quite reliable and more effective in making short term forecasting predictions vs other popular techniques such as Exponential Smoothing. Deep Learning options, of course, also exist (first iterations of this model utilizing FB Neural Prophet's AR-Net) but are often over-complicated and perform worse than their statistical counterparts. 

See [ds_app_2.py](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/blob/main/ds_app_2.py) or the Model Performance section for model details.



## Streamlit Web-App
The following web-app utilizes streamlit-cloud to deploy several ML models (Auto-ARIMA; PELT cpd) created from different sources (BQML; Python ruptures) to provide functional, advanced analytics in the form of an internal business tool. 

### Data 
Data is queried from a larger table in a BQ database (API ingestion every half hour) to isolate for the most recent episode for every channel through a for loop using SQL. Data transformation is applied to ensure the data is prepared to be loaded into a BQ auto ARIMA model. Alterations to the query are also made (through ranking) to ensure that the table is robust to changes such as when single or multiple episodes are deleted from the social media channel itself, altering the most up-to-date episode. See below:
```
CREATE OR REPLACE TABLE `insert_table_here` AS
SELECT CAST(NULL AS TIMESTAMP) filled_time,
     CAST(NULL AS INT64) topsnap_views,
     CAST(NULL AS STRING) story_id_fixed;

FOR variable IN
(SELECT COUNT(DISTINCT post_id) story_counts,
      post_id,
      title,
      publisher_name
FROM (WITH cte AS(-- Account for "deleting" strategy to identify the currently running episode (could be 2nd not 1st episode)
                  -- Get Max interval time for the 2 most recent episodes per channel
                  SELECT MAX(interval_time) interval_time, 
                        published_at,
                        title, 
                        post_id,
                        publisher_name,
                        rolling_recency_ranking
                  FROM (--Rank all of the most recent episodes per channel from 1 and on
                        SELECT rt.account_id,
                                dt.account_name AS publisher_name, 
                                rt.platform, 
                                rt.post_id, 
                                datetime(dt.published_at, "America/Toronto") published_at,
                                datetime(rt.interval_time, "America/Toronto") interval_time, 
                                dt.title, 
                                dt.description, 
                                dt.tile_url, 
                                dt.logo_url,
                                rt.views, 
                                --LAG(rt.views) OVER(PARTITION BY rt.post_id ORDER BY rt.interval_time ASC) lag_views, 
                                rt.views - COALESCE((LAG(rt.views) OVER(PARTITION BY rt.post_id ORDER BY rt.interval_time ASC)), 0) views_diff, 
                                DENSE_RANK() OVER (PARTITION BY rt.account_id ORDER BY dt.published_at DESC) rolling_recency_ranking
                          FROM realtime_table rt
                          LEFT JOIN table_details dt
                          ON (rt.account_id = dt.account_id) AND (rt.post_id = dt.post_id)
                          WHERE TRUE 
                          AND rt.platform IN ('Snapchat')
                          ORDER BY post_id NULLS LAST, interval_time ASC
                          )
                  WHERE rolling_recency_ranking <= 20
                  GROUP BY published_at, title, post_id, rolling_recency_ranking, publisher_name
                  ORDER BY publisher_name, interval_time DESC, rolling_recency_ranking ASC
                  )
      -- Apply the final ranking to decide between the top 2 most recent episodes in a channel
      SELECT *, 
            DENSE_RANK() OVER(PARTITION BY publisher_name ORDER BY interval_time DESC, published_at DESC, rolling_recency_ranking ASC) final_rank
      FROM cte
      )
WHERE final_rank  = 1
AND publisher_name IS NOT NULL
GROUP BY post_id, title, publisher_name)

DO
INSERT INTO `insert_table_here`
SELECT timestamp(DATE_TRUNC(interval_time, HOUR)) AS filled_time,
       views topsnap_views, 
       post_id AS story_id_fixed
FROM `Views.snap_posts_rt`
WHERE TRUE 
AND post_id = variable.post_id
AND EXTRACT(MINUTE FROM interval_time) = 0
ORDER BY interval_time ASC;

END FOR;
```

### Summary Table
- Summary table compiles information regarding real-time video performance, timeseries forecasting data, changepoint detection data, daily changes in momentum (24 hour deltas), daily channel performance averages (90-day rolling), and hourly benchmarks to provide a high(er) level view on which episodes to keep running vs which to replace.
- Decisions are generated via conditional logic, informed by a combination of model outputs, benchmarks, and other metrics of interest.
- agGrid compatibility provides the ability to filter and select/unselect columns for scalability (episode names and ID's are discluded in the instance below)
- Data is cached periodically to save on computing power, and updated as data in the GCP database is updated.

![image](https://github.com/a-memme/snap_dynamic_scheduling/assets/79600550/92c2712b-5beb-4063-86ca-f5d39d88fc3f)

##### Current Section 
*i.e Current Hour, Current Perforance, Current Benchmark and % v Bench*
- represents how many hours the episode has been running for, its current performance (at that hour) and channel benchmark at that hour

##### Forecast Section 
*i.e Fcst Period, Forecast, Fcst Bench, and Fcst % against bench*
- represents the cumulative predicted performance of the episode at the forecasted hour (nearest 24-hour window), and how that relates to the channel benchmark at the respective forecasted hour.
- ARIMA forecast model is compiled in BigQuery ML and run on an hourly schedule. See the following code below:
```
CREATE OR REPLACE MODEL `insert_model_name_here`
OPTIONS(MODEL_TYPE='ARIMA_PLUS',
       time_series_timestamp_col='filled_time',
       time_series_data_col='topsnap_views',
       TIME_SERIES_ID_COL = 'story_id_fixed',
       AUTO_ARIMA = TRUE,
       DATA_FREQUENCY = 'HOURLY',
       CLEAN_SPIKES_AND_DIPS = TRUE,
       ADJUST_STEP_CHANGES = TRUE,
       TREND_SMOOTHING_WINDOW_SIZE = 6,
       MAX_TIME_SERIES_LENGTH = 18,
       SEASONALITIES = ['NO_SEASONALITY']) 
       AS
SELECT * FROM `insert_table_here`
WHERE filled_time IS NOT NULL
ORDER BY story_id_fixed, filled_time ASC;
```

##### Trend Sentiment 
- Results of the changepoint detection model.
- 🔥 represents an increase in trend (in a recent time-frame - say past 48hrs for example) while a 🥶 represents a decrease in trend (in a recent timeframe).
- The number of emojis depicts the intensity of said trend. See "Forecasting + Momentum" section below for more details.


- The model is compiled in the function below, utilizing the PELT (Pruned Extract Linear Time) algorithm which identifies change points through minimizing a penalized sum of costs. Here, a penalty of 6 is used as we want to balance meaningful intepretation with model sensitivity.
     - The first 18 hours are also cut off from interpretation as this is a period to which we always expect to see negative change so this identification wouldn't be very meaningful. See the code below or [ds_app_2.py](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/blob/main/ds_app_2.py) for more details.

```
def changepoint_df(choose_episode):
  #identify episode
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
  # Detect the change points
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
```


### Forecasting + Momentum 
- Cumulative performance of an episode can be plotted using the respective story ID and 24 hour window in which we wish to forecast to (from the drop-down selection).
- The cumulative line graph shows the relevant benchmarks as well as areas in which positive or negative change has been detected depicted by 🔥 or 🥶 respectively (offline detection of the nearest 24 hour prediction).
- Historical performance is represented by the dark purple line while forecasted performance is represented by royal blue (See Legend). 

![image](https://github.com/a-memme/snap_dynamic_scheduling/assets/79600550/10329ab1-8f8d-48aa-b42b-9752a6b7d97c)





### Model Performance
- Evaluation of the model can be easily visualized in the webapp via the "Evaluate Model" button in the Model Performance Section.
- p, d, and q values generated by the Auto-ARIMA model as well as AIC results are visualized in the table. 
- Model testing is done internally and not shown below.

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/9493e53b-15ab-478d-b4fa-f194f0101d45)
