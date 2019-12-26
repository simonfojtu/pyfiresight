# pyfiresight

- plotter.py - script for using the FPD as a simple plotter (when a pen is held in the end-effector)
 - set maximum velocity (mv) and time to max. velocity (tv):

    python plotter.py --mv 9000 --tv 0.16

 - draw a bitmap:
    
    python plotter.py bitmap --image img/2019_067.png --dpmm 2 --center --dots --hflip
