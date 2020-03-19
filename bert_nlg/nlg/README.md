# Bert to NLG

## Requirements  
- python 3  
- tensorflow == 1.1x 

## Run example  
>>Please confit the config.py  
>>1.make data  
```
specify you data path in the config.py, then execute python data_utils.py
```    
>>1.train  
```shell
python train.py train  
```  
>>2.package pb model  
```shell
python train.py package  
```  
>>3.Prediction
```shell
python predict.py  
```  

## Reference  
- TensorFlow code and pre-trained models for BERT https://arxiv.org/abs/1810.04805 
