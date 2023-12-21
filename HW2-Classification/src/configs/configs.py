# data parameters
# TODO: change the value of "concat_nframes" for medium baseline
concat_nframes = 3   # the number of frames to concat with, n must be odd (total 2k + 1 = n frames)
valid_ratio = 0.2   # the ratio of data used for validation, the rest will be used for training

# training parameters
seed = 1221          # random seed
batch_size = 512       # batch size
num_epochs = 5000         # the number of training epoch
learning_rate = 1e-4      # learning rate
model_path = './models/model.ckpt'  # the path where the checkpoint will be saved
data_path = "./data/libriphone"  # the path where the data set saved
preds_path = "./data/preds.csv"  # the path where the prediction saved
