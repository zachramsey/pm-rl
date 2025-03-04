### Overlap Studies
Abbreviation | Parameters                                  | Description
------------ | ------------------------------------------- | ------------
BBANDS       | period(5), nbdevup(2.0), nbdevdn(2.0)       | Bollinger Bands
DEMA         | period(30)                                  | Double Exponential Moving Average
EMA          | period(30)                                  | Exponential Moving Average
HT_TRENDLINE |                                             | Hilbert Transform - Instantaneous Trendline
KAMA         | period(30)                                  | Kaufman Adaptive Moving Average
MA           | period(30)                                  | Moving average
MAMA         | fastlim(0.0), slowlim(0.0)                  | MESA Adaptive Moving Average
MAVP         | periods(flt), minperiod(2), maxperiod(30) | Moving average with variable period
MIDPOINT     | period(14)                                  | MidPoint over period
MIDPRICE     | period(14)                                  | Midpoint Price over period
SAR          | acc(0.0), maximum(0.0)                      | Parabolic SAR
SAREXT       | startvalue(0.0), offsetonreverse(0.0), <br>accinitlong(0.0), acclong(0.0), <br>accmaxlong(0.0), accinitshort(0.0), <br>accshort(0.0), accmaxshort(0.0) | Parabolic SAR - Extended
SMA          | period(30)                                  | Simple Moving Average
T3           | period(5), vfactor(0.0)                     | Triple Exponential Moving Average (T3)
TEMA         | period(30)                                  | Triple Exponential Moving Average
TRIMA        | period(30)                                  | Triangular Moving Average
WMA          | period(30)                                  | Weighted Moving Average

### Momentum Indicators
Abbreviation | Parameters                                        | Description
------------ | ------------------------------------------------- | ------------
ADX          | period(14)                                        | Average Directional Movement Index
ADXR         | period(14)                                        | Average Directional Movement Index Rating
APO          | fastperiod(12), slowperiod(26)                    | Absolute Price Oscillator
AROON        | period(14)                                        | Aroon
AROONOSC     | period(14)                                        | Aroon Oscillator
BOP          |                                                   | Balance Of Power
CCI          | period(14)                                        | Commodity Channel Index
CMO          | period(14)                                        | Chande Momentum Oscillator
DX           | period(14)                                        | Directional Movement Index
MACD         | fastperiod(12), slowperiod(26), signalperiod(9)   | Moving Average Convergence/Divergence
MACDEXT      | fastperiod(12), slowperiod(26), signalperiod(9)   | MACD with controllable MA type
MACDFIX      | signalperiod(9)                                   | Moving Average Convergence/Divergence Fix 12/26
MFI          | volume, period(14)                                | Money Flow Index
MINUS_DI     | period(14)                                        | Minus Directional Indicator
MINUS_DM     | period(14)                                        | Minus Directional Movement
MOM          | period(10)                                        | Momentum
PLUS_DI      | period(14)                                        | Plus Directional Indicator
PLUS_DM      | period(14)                                        | Plus Directional Movement
PPO          | fastperiod(12), slowperiod(26)                    | Percentage Price Oscillator
ROC          | period(10)                                        | Rate of change : ((price/prevPrice)-1)*100
ROCP         | period(10)                                        | Rate of change Percentage: (price-prevPrice)/prevPrice
ROCR         | period(10)                                        | Rate of change ratio: (price/prevPrice)
ROCR100      | period(10)                                        | Rate of change ratio 100 scale: (price/prevPrice)*100
RSI          | period(14)                                        | Relative Strength Index
STOCH        | fastk_period(5), slowk_period(3), slowd_period(3) | Stochastic
STOCHF       | fastk_period(5), fastd_period(3)                  | Stochastic Fast
STOCHRSI     | period(14), fastk_period(5), fastd_period(3)      | Stochastic Relative Strength Index
TRIX         | period(30)                                        | 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
ULTOSC       | period1(7), period2(14), period3(28)              | Ultimate Oscillator
WILLR        | period(14)                                        | Williams' %R

### Volume Indicators
Abbreviation | Parameters                              | Description
------------ | --------------------------------------- | ------------
AD           |                                         | Chaikin A/D Line
ADOSC        | fastperiod(3), slowperiod(10)           | Chaikin A/D Oscillator
OBV          |                                         | On Balance Volume

### Cycle Indicators
Abbreviation | Parameters | Description
------------ | ---------- | ------------
HT_DCPERIOD  |            | Hilbert Transform - Dominant Cycle Period
HT_DCPHASE   |            | Hilbert Transform - Dominant Cycle Phase
HT_PHASOR    |            | Hilbert Transform - Phasor Components
HT_SINE      |            | Hilbert Transform - SineWave
HT_TRENDMODE |            | Hilbert Transform - Trend vs Cycle Mode

### Price Transform
Abbreviation | Parameters | Description
------------ | ---------- | ------------
AVGPRICE     |            | Average Price
MEDPRICE     |            | Median Price
TYPPRICE     |            | Typical Price
WCLPRICE     |            | Weighted Close Price

### Volatility Indicators
Abbreviation | Parameters          | Description
------------ | ------------------- | ------------
ATR          | period(14)          | Average True Range
NATR         | period(14)          | Normalized Average True Range
TRANGE       |                     | True Range

BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, SAREXT, SMA, T3, TEMA, TRIMA, WMA, ADX, ADXR, APO, AROON, AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR, AD, ADOSC, OBV, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, ATR, NATR, TRANGE