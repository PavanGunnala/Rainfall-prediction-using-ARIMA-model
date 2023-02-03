# This data aims at examining the rainfall pattern and fit a suitable model for rainfall 
# prediction in Kerala (KL) subdivision of India.
# Data from 1901 to 2017 is downloaded from data.gov.in. 
# Our aim is to find  out the rainfall pattern  in Kerala over last 117 years 
# and predict the rainfall in upcoming Months/Quarters/Annual.

#
# Shapelets classification is done prior to this so that we have applied the Model in this code
#The training model is aware of uncertanities
# Language used R Programming
# Method : Time Series Analysis
# Modelused Auto.ARIMA

# Load the necessary package
# install.packages("reshape","forecast","tseries","rmarkdown","knitr","ggplot2") # if required
library(reshape)
library(forecast)
library(tseries)
library(rmarkdown)
library(knitr)
library(ggplot2)

# Load the data
setwd('C:\Users\91897\Desktop\Mini\MINI Project\Project')
getwd()
rain_india = read.csv("Sub_Division_IMD_2017.csv")

# View structure of data
str(rain_india)
# Data has 4188 obs of 19 variables

# Kerala rain data from Jan 1901 to Dec 2017
KL_months = rain_india[rain_india$SUBDIVISION == "Kerala",c("YEAR","JAN","FEB","MAR","APR",
                                                            "MAY","JUN","JUL","AUG","SEP" ,
                                                            "OCT","NOV","DEC")]
# Renaming row names
rownames(KL_months) = KL_months$YEAR
head(KL_months,n=3)

# View structure of data
str(KL_months)
# Data has 117 obs of 12 variables

# Are there any unusual or No data values?
summary(KL_months)

# Converting Multivariate to Univariate data
KL_months_univ = melt(t(KL_months[,c(2:13)]),
                      varnames = c("MONTH","YEAR"))

head(KL_months_univ)
class(KL_months_univ)
# Type of the class is data.frame

# Converting data to Time Series object
KL_months_ts = ts(KL_months_univ$value,start= min(KL_months_univ$YEAR),frequency=12)
# As monthly forecast the freq is set to 12
class(KL_months_ts)
# Type of the class should be time series
head(KL_months_ts, n = 36)
tail(KL_months_ts, n = 36)

# Visualization
# Plot the Time Series
plot(KL_months_ts,
     main = "KL monthly Rain from 1901 to 2017",
     ylab = "Rainfall in mm")
abline(reg = lm(KL_months_ts ~ time(KL_months_ts)))
# The data looks stationary by observation
# Also the mean and variance is constant

boxplot(KL_months_ts ~ cycle(KL_months_ts),col = "yellow",names = month.abb,
        xlab ="Month",ylab ="Rainfall in mm", main = "KL monthly Rain summary")
# The data considered on monthly basis has some outliers
# The month of July recorded the highest amount of rainfall, followed closely by June
# The months of January,February and March recorded the least amount of rainfall.
# KL_months_ts = tsclean(KL_months_ts)


# Decomposition
decomp_KL_months_ts = decompose(KL_months_ts)
plot(decomp_KL_months_ts)
# The data has seasonal effect, with a usual rise and fall pattern being experienced 
# yearly over the period. This implies that regular monthly rainfall figures recorded 
# each year was influenced by the rise and fall pattern of the seasonality component. 
# However, the trend seems to be very constant over time, although there are 
# a few ups and downs over some periods. 
# The random effect is very stable over the time period.


# Is Stationary?
# Augmented Dickey Fuller Test
tseries::adf.test(KL_months_ts)
# Dickey-Fuller = -8.2612, p-value = 0.01
# p-value < 0.05 , so our data is stationary

# Autocorrelation Function
ggAcf(KL_months_ts) #q value
ggPacf(KL_months_ts) # p value

# Build model on Training time duration

# Training Set - Set trainging data from 1901 to 2015
KL_months_tr = window(KL_months_ts,start = c(1901,1),end = c(2015,12))

# Test Set - Set trainging data from 2016 to 2017
KL_months_te = window(KL_months_ts,start = c(2016,1),end =c(2017,12))

# # Plot some forecasts using 4 methods
plot(KL_months_tr,main ="Forecasting KL Rainfall using 4 methods",ylab ="Rainfall in mm",
     xlim = c(2013,2018),ylim=c(0,1500))

#Mean Method
lines(meanf(KL_months_tr,h=24)$mean,col=5)

# Naive Method
lines(rwf(KL_months_tr,h=24)$mean,col=2)

# Drift Method
lines(rwf(KL_months_tr,drift = TRUE,h=24)$mean,col=3)

# Seasonal Naive Method
lines(snaive(KL_months_tr,h=24)$mean,col=6)

# test set
lines(KL_months_te,col="darkblue")

#Legend
legend("topright",lty=1,col = c(5,2,3,6,"darkblue"),bty='n', cex = 0.8,
       legend = c("Mean Method","Naive Method","Drift Method","Seasonal Naive Method",
                  "Actual Test"))

# Accuracy
accuracy(meanf(KL_months_tr,h=24),KL_months_te)
accuracy(rwf(KL_months_tr,h=24),KL_months_te)
accuracy(rwf(KL_months_tr,drift = TRUE,h=24),KL_months_te)
accuracy(snaive(KL_months_tr,h=24),KL_months_te)
# Mean,Naive, Drift Method do not detect nor the trend neither the seasonality
# Seasonal Naive method detects the seasonality and prediction is from recent observation


# caution takes a while to compute
KL_months_tr_arima = auto.arima(KL_months_tr)
summary(KL_months_tr_arima)
# This will return best ARIMA model

# Forecast for Test time duration
KL_months_te_forecast_arima = forecast(KL_months_tr_arima, h = 24)
print(KL_months_te_forecast_arima)

# Visualization
plot(KL_months_te_forecast_arima,
     xlim = c(2013,2018),
     xlab = "Time",
     ylab = "rainfall(mm)",
     main = "Forecast for Test Time duration")

lines(KL_months_te,col= 6,lty=2)

# Legends
legend("topright",lty = 1,bty = "n",col = c(6,"mediumblue","lightsteelblue","gray88"),
       c("Testdata","ArimaPred","80% Confidence Interval","95% Confidence Interval")
       ,cex = 0.8)

#PointForecast
KL_months_te_pointforecast_arima = KL_months_te_forecast_arima$mean
KL_months_te_pointforecast_arima = round(KL_months_te_pointforecast_arima,digits = 1)
print(KL_months_te_pointforecast_arima)

# Actual Test data
print(KL_months_te)

# Test data vs ARIMA Point Forecast
plot( KL_months_te,
      ylim = c(-500, 1500),
      xlim = c(2016,2018),
      main = "Monthly : Testdata vs ARIMA PF",
      ylab = 'Test data/ARIMA PF',col= 6,lty=2)

lines(KL_months_te_pointforecast_arima,col="4")

#Legends
legend("topleft",legend = c("Test data","ARIMA PF"),
       col = c(6,4),lty = c(2,1),cex = 0.8)


# Evaluate the forecast/prediction accuracy (MAPE)
MAPE = function(actual, predicted) {
  abs_percent_diff = abs((actual - predicted) / actual)*100
  # percentage difference could be infinite if actual has value 0
  abs_percent_diff = abs_percent_diff[is.finite(abs_percent_diff)]
  mape = c(mean(abs_percent_diff), median(abs_percent_diff))
  names(mape) = c("Mean_APE", "Median_APE")
  return (mape)
}

MAPE(KL_months_te, KL_months_te_pointforecast_arima)
# 228.89 Mean_APE
# 45.79 Median_APE

# Forecast for future times
KL_months_arima = auto.arima(KL_months_ts)
summary(KL_months_arima)

KL_months_forecast_arima = forecast(KL_months_arima, h= 24)
print(KL_months_forecast_arima)

KL_months_pointforecast_arima = round(KL_months_forecast_arima$mean,digits = 1)
print(KL_months_pointforecast_arima)

# Diagnostic Test
# Ljung Test
Box.test(KL_months_forecast_arima$residuals, type = "Ljung-Box")
# p-value = 0.7606

checkresiduals(KL_months_forecast_arima$residuals)
# we can conclude that there is very little evidence for non-zero 
# autocorrelations in the forecast errors at lags 1-24

# Plotting Forecast

plot(KL_months_forecast_arima,xlab = "Time",ylab = "rainfall(mm)", 
     main = "Forecast from Arima for 2018 and 2019")

plot(KL_months_forecast_arima, xlim = c(2014,2020),ylim = c(-500,2000),
     ylab = "rainfall(mm)", main = "Forecast from Arima for 2018 and 2019")

lines(KL_months_te_pointforecast_arima,col="red")

legend("topright",legend = c("Actual","Test Time Forecast","Future Forecast"
                             ,"80% Confidence Interval","95% Confidence Interval"),
       col = c("black","red","mediumblue","lightsteelblue","gray88"),lty = 1,cex = 0.8)

boxplot(KL_months_pointforecast_arima ~ cycle(KL_months_pointforecast_arima),
        col = "yellow",names = month.abb,xlab ="Month",
        ylab ="Rainfall in mm", main = "KL monthly Rain summary")

# Conclucion
# The results revealed that the region experience much rainfall in the months of 
# June, July and least amount of rainfall in the months of January,February and March . 
# Auto.ARIMA has been identified as an appropriate model for predicting monthly average 
# rainfall figures for the Kerala Subdivision of India.

