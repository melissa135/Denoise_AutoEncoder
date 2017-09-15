# Denoise_AutoEncoder

The implement of layer-wise training denoise autoencoder in pytorch. 

Compress the 5-minute K line in a day (48 dimensions) of stock in A-share market into a vector of 5 dimensions, through 3 pairs of encoder-decoder layer with denoising.
     

## Requirements
* Matplotlib  
* Pandas  
* Pytorch  
* Numpy
       

## Network
```
AutoEncoder_1 (
  (encoder1): Linear (48 -> 20)
  (decoder1): Linear (20 -> 48)
)  
```
```
AutoEncoder_2 (
  (encoder1): Linear (48 -> 20)
  (encoder2): Linear (20 -> 10)
  (decoder2): Linear (10 -> 20)
  (decoder1): Linear (20 -> 48)
)  
```
```
AutoEncoder_3 (
  (encoder1): Linear (48 -> 20)
  (encoder2): Linear (20 -> 10)
  (encoder3): Linear (10 -> 5)
  (decoder3): Linear (5 -> 10)
  (decoder2): Linear (10 -> 20)
  (decoder1): Linear (20 -> 48)
)  
```
Each layers of above network is simple full connection.
      

## Usage
1. Run `train_net_1.py` to train the `AutoEncoder_1`. 
2. Run `train_net_2.py`, which copies the parameters of encoder1/decoder1 from `AutoEncoder_1` as the fixed parameters for `AutoEncoder2` and only train the encoder2/decoder2.
3. Run `train_net_3.py`, which copies the parameters of encoder1/decoder1 and encoder2/decoder2 from `AutoEncoder_2` as the fixed parameters for `AutoEncoder3` and only train the encoder3/decoder3.
4. Run `make_encoder.py` , truncate the `AutoEncoder_3` and retain the front part of it as an compression encoder which converts a 48-dimensional vector into a 5-dimension vector.
5. Run `encode_vision.py` to show the compressed trend chart of stocks.
      

## Result
![](https://github.com/melissa135/Denoise_AutoEncoder/blob/master/Figure_1.png) 
Loss sequence on train and test set of each training stage.
 
![](https://github.com/melissa135/Denoise_AutoEncoder/blob/master/vision_10.png) 
The original 5-minute K line sequnce and the recovered sequence from compressed vector.
