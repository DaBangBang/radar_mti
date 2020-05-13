# Moving Target Indication techniques for Ti FMCW radar

We implement moving target indication (MTI) filter to detect moving target including human and small ball. The mti techniques that we used were referred by Matthew Ash et. al,in the paper named "On Application of Digital Moving Target Indication Techniques to Short-Range FMCW Radar Data.

## Type of MTI

>there are 4 type of MTI in this paper

- Background subtraction with updating background every 0.1 second
- Stove technique (First order FIR filter)
- High pass FIR filter
- High pass IIR filter

## 1. Background subtraction  

>Human walking (slow)
- config 300MHz slope, 200 adc samples <br />

  <img src="pic/bg_sub_slow_300.png" width="50%" height="50%">

## 2. Background subtraction (with updating background)

>Human walking (slow)

- config 300MHz slope, 200 adc samples <br />
    
    <img src="pic/bg_update_slow_300.png" width="50%" height="50%">

## 3. Stove technique (First order FIR filter)

>Human walking (slow)

- config 300MHz slope, 200 adc samples <br />
    
    <img src="pic/stove_slow_300.png" width="50%" height="50%">

## 3. High pass FIR filter

>Human walking (slow)

- config 300MHz slope, 200 adc samples <br />
    
    <img src="pic/FIR_slow_300.png" width="50%" height="50%">

## 3. High pass IIR filter

>Human walking (slow)

- config 300MHz slope, 200 adc samples <br />
    
    <img src="pic/IIR_slow_300.png" width="50%" height="50%">