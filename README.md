# pyfiresight

- plotter.py - script for using the FPD as a simple plotter (when a pen is held in the end-effector)
 - set maximum velocity (mv) and time to max. velocity (tv):

    python plotter.py --mv 9000 --tv 0.16

 - draw a bitmap:
    
    python plotter.py bitmap --image img/2019_067.png --dpmm 2 --center --dots --hflip

 - plotting speed (`img/test_img_1.png`)
       mv       tv      time    speedup
        9000    0.12    153s     0%
        9000    0.05    148s     3%
       13000    0.12    150s     2% 
     z-up = 0
        9000    0.12    132s    14%
        9000    0.05    cca 120s, but missing steps
       13000    0.12    130s    15% 

 - recommended settings:
    --tv 0.12 --mv 9000 --z-up=0 --z-down=-10

