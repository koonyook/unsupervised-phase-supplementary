# Requirements
* Python 3
* TensorFlow 1.7
* Numpy
* Matplotlib
* plotly 4.1.1

# Step-by-step guide
1. Generate synthetic data for all 3 shapes.
```
python w1_genData.py
```
2. Calculate mean and standard deviation for each training data.
```
python w2_genMeanSd.py
```
3. Train a network. You must choose a shape ('heart', 'bow', or 'acrobat') and give a name to the network. The training may takes a while (1-10 minutes for 5000 iterations) depending on your processor. The progress is produced as figures in the folder with the network name (inside img folder).
```
python w3_network.py learn acrobat acroNet01
```
4. Infer phase from test data and visualize them. You must choose a shape ('heart', 'bow', or 'acrobat') and the trained network name. A result file (phaseAgainstTime_test.html) will be generated in the folder with the network name. It can be opened on any internet browser. The plot is interactive. You can drag on the plots to zoom in or out.
```
python w3_network.py infer acrobat acroNet01
```