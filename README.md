# LTBoost
Boosted Hybrids of ensemble gradient algorithm for the long-term time series forecasting (LTSF)

## Datasets

The LTBoost framework is evaluated using nine well-established open-source long-term time series forecasting benchmarks. These datasets are tailored to specific use cases, including traffic, weather, disease spread, and industrial problems, characterized by their multivariate nature. This setup is emblematic of complex forecasting challenges commonly encountered in practice, making these datasets exceptionally well-suited for evaluating models that are lightweight, efficient, and quick in forecasting multivariate long-term time series.

### Dataset Descriptions

Below is a summary of the datasets used:

- **Electricity** [🔗](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) [^1]: Hourly electricity consumption data of 321 clients from 2012 to 2014 with a total of 26,304 time steps. The last client is labeled as the target value "OT".
- **Exchange Rate** [^2]: Daily exchange rates of eight countries' currencies against the US dollar from 1990 to 2010, totaling 7,588 timesteps.
- **Traffic** [🔗](https://pems.dot.ca.gov/): Hourly road occupancy rates of 862 sensors on San Francisco highways during 2015-2016, comprising 175,544 timesteps. The last sensor is labeled as the target "OT".
- **Weather** [🔗](https://www.bgc-jena.mpg.de/wetter/): Data recorded every 10 minutes in 2020, includes 52,696 time steps with 21 weather indicators. The target value is "$CO_2$", labeled as "OT".
- **ILI** [🔗](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html): Weekly ratios of patients with influenza-like illness from seven Centers for Disease Control and Prevention in the US, spanning from 2002 to 2021 with 966 time steps.
- **ETT** [🔗](https://github.com/zhouhaoyi/ETDataset) [^1]: Datasets ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-minute-level) detailing oil and load features of electricity transformers from July 2016 to July 2018.

[^1]: [Zhou et al., 2021 Informer](https://arxiv.org/abs/2012.07436)
[^2]: [Lai et al., 2018 Modeling](https://arxiv.org/abs/1703.07015)

### Dataset Characteristics Table

Refer to the following table for a summary of the datasets' characteristics, including their variates, timesteps, granularity, and stationarity tests. More detailed descriptions and additional tables are available in the Appendix.

| Dataset       | #Variates | #Timesteps | Granularity | p-value |
|---------------|-----------|------------|-------------|---------|
| Electricity   | 321       | 26,304     | hourly      | 0.0000  |
| Exchange Rate | 8         | 7,588      | daily       | 0.4166  |
| Traffic       | 862       | 175,544    | hourly      | 0.0000  |
| Weather       | 21        | 52,696     | 10 min      | 0.0000  |
| ILI           | 7         | 966        | weekly      | 0.7598  |
| ETTh1         | 7         | 17,420     | hourly      | 0.0246  |
| ETTh2         | 7         | 17,420     | hourly      | 0.0401  |
| ETTm1         | 7         | 69,680     | 15 min      | 0.0028  |
| ETTm2         | 7         | 69,680     | 15 min      | 0.0014  |

The comprehensive augmented Dickey-Fuller unit root tests conducted for each dataset help assess their stationarity, crucial for modeling long-term forecasts accurately.

# Detailed Dataset Descriptions

This appendix provides a more detailed description of the nine real-life datasets used in the evaluation of the Ltboost framework. These datasets, selected for their complexity and real-world applicability, span a wide range of domains including electricity consumption, currency exchange rates, traffic occupancy rates, weather conditions, influenza-like illness occurrences, and electricity transformer temperatures.

## Electricity
The [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/electricityLoadDiagrams20112014), available from the well established UC Irvine Machine Learning Repository, encompasses hourly electricity consumption data (in kWh) for 321 clients from 2012 to 2014. A [cleaned version](https://github.com/laiguokun/multivariate-time-series-data/tree/master/electricity) of this dataset, includes a comprehensive record across 26,304 timesteps. The dataset identifies its final client as target value *OT*, providing a detailed overview of consumption patterns over the specified period.

## Exchange Rate
The [Exchange Rate dataset](https://arxiv.org/abs/1703.07015) captures the daily exchange rates of eight countries' currencies against the U.S. dollar, encompassing Australia, Great Britain, Canada, Switzerland, China, Japan, New Zealand, and Singapore. Spanning from January 1, 1990, to October 10, 2010, this dataset offers a detailed view through 7,588 timesteps. For a succinct summary of the currencies involved, refer to the table below.

| Variate | Country       | Currency                |
|---------|---------------|-------------------------|
| 0       | Australia     | Australian dollar (AUD) |
| 1       | Great Britain | Sterling (GBP)          |
| 2       | Canada        | Canadian dollar (CAD)   |
| 3       | Switzerland   | Swiss franc (CHF)       |
| 4       | China         | Renminbi (CNY)          |
| 5       | Japan         | Japanese yen (JPY)      |
| 6       | New Zealand   | New Zealand dollar (NZD)|
| OT      | Singapore     | Singapore dollar (SGD)  |

## Traffic

The [Traffic dataset](https://pems.dot.ca.gov/), derived from the California Department of Transportation, features road occupancy rates measured on a scale from 0 to 1 across 17,544 hourly timesteps, spanning the years 2015 to 2016. Data were collected using 862 sensors deployed along the freeways of San Francisco, with the final sensor designated as target value *OT*.

## Weather

The [Weather dataset](https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf), recorded in 2020, captures data with a 10-minute granularity, incorporating 21 weather indicators across 52,696 timesteps. Within this dataset, the final indicator, specifically the *$CO_2$* concentration, is denoted as target value *OT*. For an introductory summary of these indicators, see the table below, and for further details.


| Symbol | Unit          | Variable                              |
|--------|---------------|---------------------------------------|
| p      | mbar          | air pressure                          |
| T      | °C            | air temperature                       |
| T_pot  | K             | potential temperature                 |
| T_dew  | °C            | dew point temperature                 |
| rh     | %             | relative humidity                     |
| VP_max | mbar          | saturation water vapor pressure       |
| VP_act | mbar          | actual water vapor pressure           |
| VP_def | mbar          | water vapor pressure deficit          |
| sh     | g kg^-1       | specific humidity                     |
| H2OC   | mmol mol^-1   | water vapor concentration             |
| rho    | g m^-3        | air density                           |
| wv     | m s^-1        | wind velocity                         |
| max. wv| m s^-1        | maximum wind velocity                 |
| wd     | degrees       | wind direction                        |
| rain   | mm            | precipitation                         |
| raining| s             | duration of precipitation             |
| SWDR   | W m^-2        | short wave downward radiation         |
| PAR    | μmol m^-2 s^-1| photosynthetically active radiation   |
| max. PAR| μmol m^-2 s^-1| maximum photosynthetically active radiation |
| Tlog   | °C            | internal logger temperature           |
| CO2    | ppm           | CO$_2$-concentration of ambient air   |

## Influenza-like Illness (ILI)

The [Influenza-like Illness (ILI)](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.htm) dataset encapsulates the weekly incidence rates of influenza-like illness reported by seven Centers for Disease Control and Prevention (CDC) across the United States. Spanning from 2002 to 2021, this dataset aggregates a total of 966 weekly observations, offering a comprehensive view of ILI trends over nearly two decades.

The dataset is particularly notable for its detailed breakdown of ILI cases, categorizing data according to patient age groups and the reporting healthcare providers, among other variates. Such granularity enables nuanced analyses of ILI spread patterns, potentially aiding in the development of targeted public health responses.

| Variable            | Description                                               |
|---------------------|-----------------------------------------------------------|
| % WEIGHTED ILI      | Percentage of ILI patients weighted by population size    |
| % UNWEIGHTED ILI    | Unweighted percentage of ILI patients                     |
| AGE 0-4             | Number of 0-4 year old ILI patients                       |
| AGE 5-24            | Number of 5-24 year old ILI patients                      |
| ILITOTAL            | Total number of ILI patients                              |
| NUM. OF PROVIDERS   | Number of healthcare providers                            |
| OT                  | Total number of patients                                  |

## Electricity Transformer Temperature (ETT)
The [ETT datasets](https://github.com/zhouhaoyi/ETDataset) include ETTh1, ETTh2 (hourly granularity), and ETTm1, ETTm2 (15-minute granularity). These datasets encompass detailed recordings of seven distinct oil and load features for two electricity transformers, offering an in-depth look at their operational dynamics. Spanning from July 2016 to July 2018, the datasets collectively provide respectively 17,420 and 69,680 timesteps of data, showcasing a comprehensive temporal coverage.

| Variable | Description        |
|----------|--------------------|
| HUFL     | High Useful Load   |
| HULL     | High Useless Load  |
| MUFL     | Middle Useful Load |
| MULL     | Middle Useless Load|
| LUFL     | Low Useful Load    |
| LULL     | Low Useless Load   |
| OT       | Oil Temperature    |
