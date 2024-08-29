# ForestAdvisor

ForestAdvisor is a system for collecting, selecting, and optimizing forest management plans. The system contains the natural language processing part and the carbon prediction part. 

## The Natural Language Processing Part

This part contains parts of codes from the UWC and the Bert. 

### UWC

Some codes from the UWC are contained. Please visit [https://github.com/BatchClayderman/UWC](https://github.com/BatchClayderman/UWC) for details. 

### Bert

A trained Bert model is contained. 

## The Carbon Prediction Part

This part contains software named ``CarbonPredictor`` for predicting carbon values. 

### ARIMA

This is a baseline model suggested for research on carbon prediction. 

### CarbonPredictor

CarbonPredictor is a Python script with a friendly GUI for training and testing models, including ARIMA, GRU, GWO-LSTM, LSTM, and SAES. 

Currently, no regularization or other pre-processing procedures are implemented. 

Therefore, users may need to scale the pure values of their data to a suitable value range like $[1, 10000]$ via scientific ways to avoid underfitting. 

Please modify the GWO implementation and its related calls when the CarbonPredictor is applied to different datasets. 

The original GWO here with initial values specified and related calls in training and testing is just an example of use. 

#### v0.5

The previous version of CarbonPredictor without a GUI, aimed to predict short-term traffic flows at first. 

As this version has not been under maintenance for a long time, it may contain potential errors and exceptions. 

Please visit [https://github.com/BatchClayderman/CEEMDAN-SE-GWO-LSTM](https://github.com/BatchClayderman/CEEMDAN-SE-GWO-LSTM) to view this version. 

#### v1.0

This is the first version of CarbonPredictor with a GUI, aimed to predict carbon emissions. 

#### v2.0

ARIMA is combined. 

The codes are further optimized and decoupled with some logic adjusted. 

#### v2.1

In this version, the PDF file type is selected as the default file type for image storage. 

To avoid the image content going beyond the borders or having too much blank space around it, the option ``bbox_inches = "tight"`` is used. 

#### v2.2

The computing formula for the evaluation metric MAPE is revised. 

If values are different from those in the publication, please refer to the values here. 

The folder for the output results of ``runPlot.py`` is changed. 
