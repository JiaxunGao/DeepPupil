# DeepPupil Net: Deep Residual Network for Precise Pupil Center Localization

--------------------------------------------

<img src = "https://github.com/npoul/npoul.github.io/blob/master/images/Precise%20Localizations.png">


## Introduction

Precise eye center localization constitutes a very promising but challenging task in many human interaction applications due to many limitations related with the presence of photometric distortions and occlusions as well as pose and shape variations. DeepPupil Net is a Fully Convolutional Network (FCN) trained to localize precisely the eye centers by performing image-to-heatmap regression between the eye regions and the corresponding heatmaps. 

## Network Architecture

<img src = "https://github.com/npoul/npoul.github.io/blob/master/images/DeepEye.png">

## Usage

This sample code tests the DeepPupilNet eye center localizer, which was trained on MUCT, BioID and Gi4E face databases. This script requires the Deep Learning Toolbox. Tested on Matlab 2020b.

Please cite the following paper if you are using this code:

N. Poulopoulos and E. Z. Psarakis, "DeepPupil Net: Deep Residual Network for Precise Pupil Center Localization", VISAPP, vol. 5, pp. 297-304, 2022


            
