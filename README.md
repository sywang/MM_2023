# clinical prediction for CAR-T study 2024

instruction:
1. clone the repo
2. create python env 
   a. use req.yaml to create conda enviment:
   conda install
   b. use requirments file with pipy
   pip install -r requirements
3. run the CAR_T_with_ltr.ipynb
   1. this notebook train models with different configuration and produces .pkl with results file
   2. run this 6 times with different configurations, the configurations are just after the imports
   3. results files (.pkl) will be saved to data dir, don't delete them
   4. edit general configuration
   5. experiment configurations needed for the visualizations:
      1. A=True, B= ....
      2. ...
      3. ...
      4. ...
      5. ...
      6. ...
4. run the visualization notebook