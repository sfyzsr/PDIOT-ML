1.py can be used to train only one student respeck data and the model in 1.py for one student is valid
Using the gatherToFeather.ipynb to gather all students data into one feather store file.
Run ML for the feather in ml.ipynb
It is showing that the same model doesn t converge for the whole data.
The reason can be deduced as :
1. When creating the dataset in the timestamp window (length = 50), one window store one action for one student ideally. However, since the data is not uniform, the dataset creating process may produce the window store two actions data or for two students. This siuation may affect the tranning.
2. One solution is to adjust the network. Another solution is that we just use one file to trainning or maybe training one by one (haven t try this method)
# Need to resolve
- [ ] find the issue for the machine learning part or maybe the data gather/creation part
- [ ] provide an efficient solution for the ML part