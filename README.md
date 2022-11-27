# Background Removal application API using Deep Neural Networks.

This script removes the background from an input image.
### Setup
The script setup.sh downloads the trained model and sets it up so that the seg.py script can understand. 
>	./setup.sh

### Running the script
Go ahead and use the script as specified below:
>	python3 seg.py sample.jpg sample.png 1

For starting flask api server run:
>	python3 seg_flask.py 

### Dependencies
>	tensorflow, PIL
