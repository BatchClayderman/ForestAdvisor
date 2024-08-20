# ForestAdvisor

ForestAdvisor is a system for collecting, selecting, and optimizing forest management plans. The system contains the natural language processing part and the carbon prediction part. 

## Natural Language Processing Part

This part contains parts of codes from the UWC and the Bert. 

### UWC

Some codes from the UWC are contained. Please visit [https://github.com/BatchClayderman/UWC](https://github.com/BatchClayderman/UWC) for details. 

### Bert

A trained Bert model is contained. 

## Carbon Prediction Part

This part contains software named ``CarbonPredictor`` for predicting carbon values. 

### ARIMA

This is a baseline model suggested for the research in the carbon prediction part. 

### CarbonPredictor

CarbonPredictor is a Python script with a friendly GUI for training and testing models including ARIMA, GRU, GWO-LSTM, LSTM, and SAES. 

Please modify the GWO implementation and the related calls to it when the CarbonPredictor is applied to different datasets. 

The original GWO with initial values specified and related calls in training and testing is just an example of use. 
