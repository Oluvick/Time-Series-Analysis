

# data loading
data <- read.csv("/Users/victoriaa/Desktop/STAT5825/DATA/ASPUS.csv")
aspus <- ts(data$ASPUS, start = c(1963, 1), frequency = 4)
attributes(aspus)
length(aspus)

# plot time series of data
plot(aspus, ylab="Quarterly Sales Prices(Dollars)", main="Average Sales Prices of Houses Sold 1963 - 2023")

# plot acf
acf(aspus, lag.max = 48, main="Sample ACF")

# split data
train=aspus[1:196]
n_train=length(train) 
# Holdout Portion: aspus from 2012:2023
test=aspus[197:244]
n_test=length(test) 

n_train
n_test

# fit linear trend model and plot residuals
tfit=time(train)
mlr.lin = lm(train~tfit)
summary(mlr.lin)
par(mfrow=c(2,2)) 
plot(mlr.lin, main="",which = 1:4)

# fit quadratic model and plot residuals
tsqfit=tfit^2/factorial(2)
mlr.quad=lm(train~tfit+tsqfit)
summary(mlr.quad)
par(mfrow=c(2,2)) # plot 4 figures, 2 in each of 2 rows
plot(mlr.quad, main="",which = 1:4)

# fit cubic model and plot residuals
tcubfit=tfit^3/factorial(3)
mlr.cub=lm(train~tfit+tsqfit+tcubfit)
summary(mlr.cub)
par(mfrow=c(2,2)) # plot 4 figures, 2 in each of 2 rows
plot(mlr.cub, main="",which = 1:4)


# Plot training data and its linear, quadratic, and cubic fits in a 2x2 layout.
par(mfrow=c(2,2))
ts.plot(train) # Time Series Plot
# Plot of xfit vs mlr.lin$fitted
plin=cbind(train,mlr.lin$fitted)
ts.plot(plin,main="train and linear fit")
pquad=cbind(train,mlr.quad$fitted)
ts.plot(pquad,main="train and quadratic fit")
pcub=cbind(train,mlr.cub$fitted)
ts.plot(pcub,main="train and cubic fit")


# Anova test
anova(mlr.lin)
anova(mlr.cub)
# Perform ANOVA on linear and cubic models, extracting residuals' sum of squares, mean square, and degrees of freedom.
anova_lin <- anova(mlr.lin)
anova_cub <- anova(mlr.cub)
red_ss <- anova_lin["Residuals", "Sum Sq"]
red_ms <- anova_lin["Residuals", "Mean Sq"]
red_df <- anova_lin["Residuals", "Df"]
full_ss <- anova_cub["Residuals", "Sum Sq"]
full_ms <- anova_cub["Residuals", "Mean Sq"]
full_df <- anova_cub["Residuals", "Df"]
red_ss
full_ss
full_ms
full_df
# computing the Extra SS F-stat
extra_ss <- red_ss - full_ss
extra_df <- red_df - full_df
extra_ss_fstat <- (extra_ss/extra_df)/full_ms
extra_ss
extra_df
extra_ss_fstat
# F-critical value
qf(0.95,2,full_df)

# Calculate and display normalized AIC values for linear, quadratic, and cubic models.
AIC.lin = AIC(mlr.lin)/n_train
AIC.quad = AIC(mlr.quad)/n_train
AIC.cub = AIC(mlr.cub)/n_train

cat("The AIC for linear fit is:", AIC.lin, "\n")
cat("The AIC for quadratic fit is:", AIC.quad, "\n")
cat("The AIC for cubic fit is:", AIC.cub, "\n")


# Seasonality Fitting Using Regression with Seasonal Indicators
# linear
tfit=time(train)

per = 4
sets = n_train/per
quarter = factor(rep(1:per, sets))
ind.model = lm(train~tfit+quarter -1)
summary(ind.model)

# quadratic
tfit=time(train)
tsqfit=tfit^2/factorial(2)
per = 4
sets = n_train/per
quarter = factor(rep(1:per, sets))
ind.model1 = lm(train~tfit+tsqfit+quarter -1)
summary(ind.model1)

# cubic
tfit=time(train)
tsqfit=tfit^2/factorial(2)
tcubfit=tfit^3/factorial(3)
per = 4
sets = n_train/per
quarter = factor(rep(1:per, sets))
ind.model2 = lm(train~tfit+tsqfit+tcubfit+quarter -1)
summary(ind.model2)


# in sample prediction error evaluation
AIC.ind1 <- AIC(ind.model1)/n_train
AIC.ind2 = AIC(ind.model2)/n_train

AIC.ind1
AIC.ind2

# out-of-sample prediction errors MSE & MAPE
test_tfit=seq(n_train + 1, length.out = n_test, by = 1)
test_tsqfit=test_tfit^2/factorial(2)

per = 4
test_sets = n_test/per
test_quarter = factor(rep(1:per, test_sets))
newind1 <- data.frame(tfit=test_tfit, tsqfit=test_tsqfit, quarter=test_quarter)
pfore.ind1 <- predict(ind.model1, newind1, se.fit = TRUE)
efore.ind1=test-pfore.ind1$fit # Forecast errors
mse.ind1=sum(efore.ind1**2)/n_test
mae.ind1=mean(abs(efore.ind1))
mape.ind1=100*(mean(abs((efore.ind1)/test)))

mse.ind1
mae.ind1
mape.ind1


# out-of-sample prediction errors MSE & MAPE
test_tfit=seq(n_train + 1, length.out = n_test, by = 1)
test_tsqfit=test_tfit^2/factorial(2)
test_tcubfit=test_tfit^3/factorial(3)
per = 4
test_sets = n_test/per
test_quarter = factor(rep(1:per, test_sets))
newind2 <- data.frame(tfit=test_tfit, tsqfit=test_tsqfit, tcubfit=test_tcubfit, quarter=test_quarter)
pfore.ind2 <- predict(ind.model2, newind2, se.fit = TRUE)
efore.ind2=test-pfore.ind2$fit # Forecast errors
mse.ind2=sum(efore.ind2**2)/n_test
mae.ind2=mean(abs(efore.ind2))
mape.ind2=100*(mean(abs((efore.ind2)/test)))

mse.ind2
mae.ind2
mape.ind2

# Create a tibble summarizing AIC, MSE, MAE, and MAPE for quadratic and cubic models with seasonal indicators.
library(tibble)
models <- c("Quadratic Model with Seasonal Indicators", "Cubic Model with Seasonal Indicators")
AIC <- c(AIC.ind1, AIC.ind2)
MSE <- c(mse.ind1, mse.ind2)
MAE <- c(mae.ind1, mae.ind2)
MAPE <- c(mape.ind1, mape.ind2)
result_df <- tibble(Model = models, AIC = AIC, MSE = MSE, MAE = MAE, MAPE = MAPE )
result_df


# Generate detrended time series for linear, quadratic, and cubic models starting from 1963 with quarterly data.
detrended_lin_ts <- ts(ind.model$residuals, start = c(1963, 1), frequency = 4) # detrended time series based on model 1
detrended_quad_ts <- ts(ind.model1$residuals, start = c(1963, 1), frequency = 4) # detrended time series based on model 2
detrended_cub_ts <- ts(ind.model2$residuals, start = c(1963, 1), frequency = 4) # detrended time series based on model 3


# Plot ACFs for observed and detrended time series from linear, quadratic, and cubic models in a 2x2 layout.
par(mfrow=c(2,2))
acf(train, lag.max=48, main="Observed")
acf(detrended_lin_ts, lag.max=48, main="Linear detrended")
acf(detrended_quad_ts, lag.max=48, main="Quadratic detrended")
acf(detrended_cub_ts, lag.max=48, main="Cubic detrended")


# Transition to ARIMA Modeling

## Plot the original quarterly house sales price series, its ACF, and the first difference of the series.
ts.plot(aspus, main="Quarterly House Sales Price")
acf(aspus)
ts.plot(diff(aspus))

# Display the growth rate of quarterly US house sales prices and its ACF, calculated as the first difference of the logarithm.
library(TSA)
par(mfrow=c(1,2))
ts.plot(diff(log(aspus)),main="Growth Rate of Quarterly US ASP")
acf(as.vector(diff(log(aspus))), main="", lag.max = 60)


# Fit an autoregressive model to the differenced log of ASPUS and conduct an Augmented Dickey-Fuller test to check for stationarity.
ar( diff(log(aspus)))
library(fUnitRoots)
adfTest( diff(log(aspus)), lags=7, type = "c")


### Split the time series into calibration portion and holdout portion
aspusgr <- log(aspus)
aspusgr.all=aspusgr
aspusgr.calib=aspusgr.all[1:231]
aspusgr.hold=aspusgr.all[232:243]


par(mfrow=c(1,1), mar=c(4,4,3,3))
ts.plot(diff(aspusgr.calib),main="Growth Rate Quarterly US ASP: Calibration data")


# Identification of MA model via acf
library(TSA)
par(mfrow=c(1, 2))
acf(diff(aspusgr.calib), main="Sample ACF of Growth Rate") 
pacf(diff(aspusgr.calib), main="Sample PACF")

# Fit an MA(4) model to the calibrated growth rate data of ASPUS and display the model summary.
aspusgr.ma4.mle <- TSA::arima(aspusgr.calib, order=c(0,1,4), include.mean=TRUE)
aspusgr.ma4.mle

# extract residuals
ma4.resid=na.omit(aspusgr.ma4.mle$residuals)
length(ma4.resid)

# residual diagnostics
par(mfrow=c(2,2))
ts.plot(ma4.resid,main="Residuals from MA(4) fit" )
qqnorm(ma4.resid,main="Normal Q-Q plot",xlab="resid"); qqline(ma4.resid)
acf(as.numeric(ma4.resid), lag.max=40, main="")
pacf(as.numeric(ma4.resid),lag.max=40, main="")


# Shapiro-Wilk test for normality
shapiro.test(ma4.resid)

# box test
Box.test(ma4.resid, lag=4, type="Ljung", fitdf=1)
Box.test(ma4.resid, lag=8, type="Ljung", fitdf=1)
Box.test(ma4.resid, lag=12, type="Ljung", fitdf=1)

# forecast on holdout 12 quarters ahead
aspusgr.ma4.fore <- predict(aspusgr.ma4.mle, n.ahead=12)
aspusgr.ma4.fore


# 95% z-intervals
U.ma4.95 <- aspusgr.ma4.fore$pred + 1.96*aspusgr.ma4.fore$se
L.ma4.95 <- aspusgr.ma4.fore$pred - 1.96*aspusgr.ma4.fore$se
quarter = 190:243
plot(quarter,aspusgr[quarter],type="o",xlim=c(190,250),
     ylab="Price",ylim=c(min(L.ma4.95), max(U.ma4.95))) # data
lines(aspusgr.ma4.fore$pred,col="red",type="o") # point forecasts
lines(U.ma4.95,col="blue",lty="dashed") # upper limit
lines(L.ma4.95,col="blue",lty="dashed") # lower limit



### Forecast Evaluation Criteria based on Holdout Prediction
err.ma4=aspusgr.hold - aspusgr.ma4.fore$pred
me.ma4=mean(err.ma4)
mpe.ma4=100*(mean(err.ma4/aspusgr.hold))
mse.ma4=sum(err.ma4**2)/length(err.ma4)
mae.ma4=mean(abs(err.ma4))
mape.ma4=100*(mean(abs((err.ma4)/aspusgr.hold)))
mae.ma4
mse.ma4
mape.ma4



### AR(4)
aspusgr.ar4.mle = TSA::arima(aspusgr.calib, order=c(4,1,0),include.mean=TRUE)
aspusgr.ar4.mle

# extract residuals
ar4.resid=na.omit(aspusgr.ar4.mle$residuals)


# residual plots for ar(4)
par(mfrow=c(2,2))
ts.plot(ar4.resid,main="Residuals from AR(4) fit" )
qqnorm(ar4.resid,main="Normal Q-Q plot",xlab="resid"); qqline(ar4.resid)
acf(as.numeric(ar4.resid), lag.max=40, main="")
pacf(as.numeric(ar4.resid),lag.max=40, main="")


# Shapiro-Wilk test for normality
shapiro.test(ar4.resid)

# Box test
Box.test(ar4.resid, lag=4, type="Ljung", fitdf=1)
Box.test(ar4.resid, lag=8, type="Ljung", fitdf=1)
Box.test(ar4.resid, lag=12, type="Ljung", fitdf=1)


# Automated Model Selection with auto.arima
library(forecast)
arima.fit <- auto.arima(aspusgr.calib, ic="aic")
arima.fit
arima.fit.resid=na.omit(arima.fit$residuals)

# Residual Diagnostic Plots of auto-arima model(2,2,1
par(mfrow=c(2,2))
ts.plot(arima.fit.resid,main="Residuals from auto arima fit" )
qqnorm(arima.fit.resid,main="Normal Q-Q plot",xlab="resid"); qqline(arima.fit.resid)
acf(as.numeric(arima.fit.resid), lag.max=40, main="")
pacf(as.numeric(arima.fit.resid),lag.max=40, main="")

# Shapiro-Wilk test for normality
shapiro.test(arima.fit.resid)

# Box test
Box.test(arima.fit.resid, lag=4, type="Ljung", fitdf=1)
Box.test(arima.fit.resid, lag=8, type="Ljung", fitdf=1)
Box.test(arima.fit.resid, lag=12, type="Ljung", fitdf=1)

# forecast on holdout 12 quarters ahead
aspusgr.arima.fore <- predict(arima.fit, n.ahead=12)
aspusgr.arima.fore

# Forecast Evaluation Criteria based on Holdout Prediction
err.arima=aspusgr.hold - aspusgr.arima.fore$pred
me.arima=mean(err.arima)
mpe.arima=100*(mean(err.arima/aspusgr.hold))
mse.arima=sum(err.arima**2)/length(err.arima)
mae.arima=mean(abs(err.arima))
mape.arima=100*(mean(abs((err.arima)/aspusgr.hold)))
mae.arima
mse.arima
mape.arima


# 95% z-intervals
U.arima.95 <- aspusgr.arima.fore$pred + 1.96*aspusgr.arima.fore$se
L.arima.95 <- aspusgr.arima.fore$pred - 1.96*aspusgr.arima.fore$se
quarter = 190:243
plot(quarter,aspusgr[quarter],type="o",xlim=c(190,250),
     ylab="Price",ylim=c(min(L.arima.95), max(U.arima.95))) # data
lines(aspusgr.arima.fore$pred,col="red",type="o") # point forecasts
lines(U.arima.95,col="blue",lty="dashed") # upper limit
lines(L.arima.95,col="blue",lty="dashed") # lower limit


# Table of Results for the ARIMA Models
library(tibble)
results_table <- tibble(
  Model = c("MA(4)", "Auto Arima(2,2,1)"),
  MAE = c(mae.ma4, mae.arima),
  MSE = c(mse.ma4, mse.arima),
  MAPE = c(mape.ma4, mape.arima)
)
print(results_table)


# Including Exogenous Predictor

data2 <- read.csv("/Users/victoriaa/Desktop/STAT5825/DATA/GDP.csv")
gdp <- ts(data2$GDP, start = c(1963, 1), frequency = 4)
attributes(gdp)
length(gdp)

# time series plot of gdp
ts.plot(gdp, main="GDP")

# Plot the first difference of GDP and its autocorrelation function up to 60 lags.
ts.plot(diff(gdp))
acf(as.vector(diff(gdp)), main="", lag.max = 60)

#differenced ASPUS data and perform an Augmented Dickey-Fuller test on the differenced GDP with 21 lags.
ar( diff(aspus))
adfTest( diff(gdp), lags=21, type = "c")

# # Display side-by-side autocorrelation plots for the second differences of GDP and ASPUS, each up to 60 lags.
par(mfrow=c(1,2))
acf(as.vector(diff(diff(gdp))), 60, main="Sample ACF of doubly diff GDP", ylab="GDP")
acf(as.vector(diff(diff(aspus))),60,main="Sample ACF of doubly diff ASPUS", ylab="ASPUS")


# Calculate second differences for GDP and ASPUS to achieve stationarity and plot their cross-correlation function up to 24 lags.
stable.gdp <- diff(diff(gdp))
stable.aspus <- diff(diff(aspus))

ccf(stable.gdp, stable.aspus,24,main="Sample CCF of GDP and ASPUS")

# Intersect the time series of stable ASPUS and GDP into a data frame and display attributes of each series.
sales.price <- ts.intersect(stable.aspus, stable.gdp, dframe=TRUE)
attributes(sales.price$stable.aspus)
attributes(sales.price$stable.gdp)


# Fit a linear regression model of stable ASPUS on stable GDP and display the model summary.
lmfit <- lm(stable.aspus ~ stable.gdp,
            data=sales.price)
summary(lmfit)

# Extract residuals from the linear model and plot them to assess model fit.
wt <- residuals(lmfit)
ts.plot(wt, main="Residual Plot")

# Plot the ACF and PACF of regression residuals 
par(mfrow=c(1,2))
acf(wt, lag.max=72, main="ACF of Regression Residuals")
pacf(wt, lag.max=72, main="PACF of Regression Residuals")


# Fit an autoregressive model to the residuals and perform an Augmented Dickey-Fuller test to check for stationarity.

library(fUnitRoots)
ar(wt)
adfTest(wt, lags=20, type='c')


# ARMAX fit 
library(astsa)
armaxfit <- with(sales.price,sarima(stable.aspus, p=3, d=0, q=1, xreg=cbind(stable.gdp),
                                    no.constant = FALSE, details = TRUE))

armaxfit
