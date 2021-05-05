import numpy as np
import os
import json
import tensorflow as tf
import random
#from vae.vae import CVAE
from vaegan.vaegan import VAEGAN
from utils import PARSER

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

args = PARSER.parse_args()
DATA_DIR = "results/{}/{}/record".format(args.exp_name, args.env_name)
SERIES_DIR = "results/{}/{}/series".format(args.exp_name, args.env_name)
model_path_name = "results/{}/{}/tf_vae".format(args.exp_name, args.env_name)

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def ds_gen():
    filenames = os.listdir(DATA_DIR)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(DATA_DIR, fname)

        data = np.load(file_path)
        img = data['obs']
        action = np.reshape(data['action'], newshape=[-1, args.a_width])
        reward = data['reward']
        done = data['done']
        N = data['N']
        
        n_pad = args.max_frames - img.shape[0] # pad so they are all a thousand step long episodes
        img = tf.pad(img, [[0, n_pad], [0, 0], [0, 0], [0, 0]])
        action = tf.pad(action, [[0, n_pad], [0, 0]])
        reward = tf.pad(reward, [[0, n_pad]])
        done = tf.pad(done, [[0, n_pad]], constant_values=done[-1])
        N = tf.pad(N, [[0, n_pad]], constant_values=N[-1])
        yield img, action, reward, done, N

def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32, tf.float32, tf.bool, tf.uint16), output_shapes=((args.max_frames, 64, 64, 3), (args.max_frames, args.a_width), (args.max_frames,), (args.max_frames,), (args.max_frames,)))
    return dataset

@tf.function
def encode_batch(batch_img):
  simple_obs = batch_img/255.0
  mu, logvar = vaegan.encode(simple_obs)
  return mu, logvar

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vaegan.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 3)
  return batch_img

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = create_tf_dataset()
dataset = dataset.batch(1, drop_remainder=True)

#vae = CVAE(args=args)
N_Z = args.z_size
encoder = [
    tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=N_Z*2),
    ]
decoder = [
    tf.keras.layers.Dense(units=1 * 1 * 4*256, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(1 * 1 * 4*256)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]
def vaegan_discrim():
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    conv1 = tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            )(inputs)
    conv2 = tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation="relu"
            )(conv1)
    flatten = tf.keras.layers.Flatten()(conv2)
    lastlayer = tf.keras.layers.Dense(units=512, activation="relu")(flatten)
    outputs = tf.keras.layers.Dense(units=1, activation = None)(lastlayer)
    return inputs, lastlayer, outputs
vaegan = VAEGAN(
    enc = encoder,
    dec = decoder,
    vae_disc_function = vaegan_discrim,
    lr_base_gen = args.vae_learning_rate, 
    lr_base_disc = args.vae_learning_rate,
    latent_loss_div=1, # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
    sig_mult = 10, # how binary the discriminator's learning rate is shifted (we squash it with a sigmoid)
    recon_loss_div = .001, # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
)

vaegan.set_weights(tf.keras.models.load_model(model_path_name, compile=False).get_weights())
mu_dataset = []
logvar_dataset = []
action_dataset = []
r_dataset = []
d_dataset = []
N_dataset = []

i=0
for batch in dataset:
  i += 1
  obs_batch, action_batch, r, d, N = batch
  obs_batch = tf.squeeze(obs_batch, axis=0)
  action_batch = tf.squeeze(action_batch, axis=0)
  r = tf.reshape(r, [-1, 1])
  d = tf.reshape(d, [-1, 1])
  N = tf.reshape(N, [-1, 1])

  mu, logvar = encode_batch(obs_batch)

  mu_dataset.append(mu.numpy().astype(np.float16))
  logvar_dataset.append(logvar.numpy().astype(np.float16))
  action_dataset.append(action_batch.numpy())
  r_dataset.append(r.numpy().astype(np.float16))
  d_dataset.append(d.numpy().astype(np.bool))
  N_dataset.append(N.numpy().astype(np.uint16))

  if ((i+1) % 100 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)
r_dataset = np.array(r_dataset)
d_dataset = np.array(d_dataset)
N_dataset = np.array(N_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset, reward=r_dataset, done=d_dataset, N=N_dataset)
