# Character level text generation

Using Keras to generate text character by character.

## Dataset
The dataset used is Oliver Twist by Charles Dickens, stored in the `corpus.txt` file and available at the following link:
https://www.gutenberg.org/files/730/730-0.txt

## Dependencies
The only dependencies are Tensorflow, Keras and h5py. Run the following:
```bash
pip install tensorflow keras 'h5py<3.0.0'
```
## Training
* When training the model, the word-to-integer mapping, tokenizer, and weights files will be stored in the `data` directory. 
* **WARNING**: This process will overwrite existing pre-trained weights.
* Run the following command to train the model:
```bash
python rnn_train.py
```

## Testing
To test the model on the testing data provided by the IMDb dataset, run the following, specifying:
* The location of the weights file you want to use relative to the working directory:
(`data/weights-50.hdf5` is the recommended value for the `path_to_weights_file` parameter)
```bash
python manual_test.py 'path_to_weights_file'
```

## Examples
```python
>>>python manual_test.py data/weights-50.hdf5
Generated text:
one penny a square! wine-stains, fruit-solter of breaker (his attention stopped to the thing bucked of his before he had been a few doors of his before, and said her head he would say would have so wore of the board and streets, and streets, and the door was streets, and streets, and streets, and th
```
