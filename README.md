# Moving Target Indication techniques for Ti FMCW radar

We implement moving target indication (MTI) filter to detect moving target including human and small ball. The mti techniques that we used were referred by Matthew Ash et. al,in the paper named "On Application of Digital Moving Target Indication Techniques to Short-Range FMCW Radar Data".

## Type of MTI

>there are 4 type of MTI in this paper

- Background subtraction with updating background every 0.1 second
- Stove technique (First order FIR filter)
- High pass FIR filter
- High pass IIR filter

The experiment consists of 3 tasks as shown in gif below. First is a human slow walking, Second is a human fast walking, Third is a small ball moving. <br />


 <img src="pic/moving_slow.gif" width="40%" height="40%">  <img src="pic/moving_fast.gif" width="40%" height="40%"> 


**Note** - **radar-config** : 300MHz slope, 200 adc samples (left) 66.626MHz slope, 1000 adc samples (right) 


## 1. Background subtraction  

>Human walking (slow) 

- range - time <br />
  
    <img src="pic/bg_sub_slow_300.png" width="40%" height="40%"> <img src="pic/bg_sub_slow_66.png" width="40%" height="40%">

>Human walking (fast)

- range - time <br />

    <img src="pic/bg_sub_fast_300.png" width="40%" height="40%"> <img src="pic/bg_sub_fast_66.png" width="40%" height="40%">


## 2. Background subtraction (with updating background)

>Human walking (slow)

- range - time <br />
    
    <img src="pic/bg_update_slow_300.png" width="40%" height="40%"><img src="pic/bg_update_slow_66.png" width="40%" height="40%">

>Human walking (fast)

- range - time <br />

    <img src="pic/bg_update_fast_300.png" width="40%" height="40%"> <img src="pic/bg_update_fast_66.png" width="40%" height="40%">


## 3. Stove technique (First order FIR filter)

>Human walking (slow)

- range - time <br />
    
    <img src="pic/stove_slow_300.png" width="40%" height="40%"><img src="pic/stove_slow_66.png" width="40%" height="40%">

>Human walking (fast)

- range - time <br />

    <img src="pic/stove_fast_300.png" width="40%" height="40%"> <img src="pic/stove_fast_66.png" width="40%" height="40%">

## 4. High pass FIR filter

>Human walking (slow)

- range - time <br />
    
    <img src="pic/FIR_slow_300.png" width="40%" height="40%"><img src="pic/FIR_slow_66.png" width="40%" height="40%">

>Human walking (fast)

- range - time <br />

    <img src="pic/FIR_fast_300.png" width="40%" height="40%"> <img src="pic/FIR_fast_66.png" width="40%" height="40%">


## 5. High pass IIR filter

>Human walking (slow)

- range - time <br />
    
    <img src="pic/IIR_slow_300.png" width="40%" height="40%"><img src="pic/IIR_slow_66.png" width="40%" height="40%">

>Human walking (fast)

- range - time <br />

    <img src="pic/IIR_fast_300.png" width="40%" height="40%"> <img src="pic/IIR_fast_66.png" width="40%" height="40%">

   

# Doppler - range image of each experiment