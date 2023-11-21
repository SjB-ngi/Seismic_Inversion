import segyio
import tensorflow as tf
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import keras
from keras.layers import Concatenate

dir_VEL = Path(r"P:\2023\02\20230203\Background-Others\BP_EnBW_AzureStorageExplorer\UHR_Velocity_Model\20230224_GBR_tariaa-1")
dir_DPT = Path(r"P:\2023\02\20230203\Background-Others\BP_EnBW_AzureStorageExplorer\Morven_Depth_UHR")
dir_TWT = Path(r"P:\2023\02\20230203\Background-Others\BP_EnBW_AzureStorageExplorer\Morven_SEGY_UHR_twt")

class SeismicDataset:

    def __init__(self, TWT_dir, VEL_dir, DPT_dir):
        self.TWT_dir = TWT_dir
        self.VEL_dir = VEL_dir
        self.DPT_dir = DPT_dir

    def read_file_id(self, file: Path):
        return file.stem.split("_")[0]
    
    def create_match_dict(self):
        match_dict = {}
        for file in self.VEL_dir.glob("*.sgy"):
            file_id = self.read_file_id(file)
            match_dict[file_id] = {}
            match_dict[file_id]["VEL"] = file
        for file in self.DPT_dir.glob("*.sgy"):
            file_id = self.read_file_id(file)
            match_dict[file_id]["DPT"] = file
        for file in self.TWT_dir.glob("*.sgy"):
            file_id = self.read_file_id(file)
            match_dict[file_id]["TWT"] = file

        with open("Morven_seismics/match_dict.pickle", "wb") as f:
            pickle.dump(match_dict, f)

    def load_match_dict(self):
        with open("Morven_seismics/match_dict.pickle", "rb") as f:
            match_dict = pickle.load(f)
        return match_dict
    
    def load_segy(self, file):
        with segyio.open(file, mode='r', ignore_geometry=True) as segy:
            return segy

    def load_segy_tensor(self, file):
        with segyio.open(file, "r") as segy:
            return tf.convert_to_tensor(segy.trace.raw[:], dtype=tf.float16)

class Unet(keras.Model):
    """ A convolutional mode following U-Net architecture"""
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        self.conv2 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        self.pool1 = keras.layers.MaxPooling2D()
        self.conv3 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.conv4 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.pool2 = keras.layers.MaxPooling2D()
        self.conv5 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")
        self.conv6 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")
        self.pool3 = keras.layers.MaxPooling2D()
        self.conv7 = keras.layers.Conv2D(512, 3, activation="relu", padding="same")
        self.conv8 = keras.layers.Conv2D(512, 3, activation="relu", padding="same")
        self.pool4 = keras.layers.MaxPooling2D()
        self.conv9 = keras.layers.Conv2D(1024, 3, activation="relu", padding="same")
        self.conv10 = keras.layers.Conv2D(1024, 3, activation="relu", padding="same")
        self.up1 = keras.layers.UpSampling2D()
        self.conv11 = keras.layers.Conv2D(512, 3, activation="relu", padding="same")
        self.conv12 = keras.layers.Conv2D(512, 3, activation="relu", padding="same")
        self.up2 = keras.layers.UpSampling2D()
        self.conv13 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")
        self.conv14 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")
        self.up3 = keras.layers.UpSampling2D()
        self.conv15 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.conv16 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.up4 = keras.layers.UpSampling2D()
        self.conv17 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        self.conv18 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        self.conv19 = keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same")

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.pool1(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.pool2(x5)
        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.pool3(x8)
        x10 = self.conv7(x9)
        x11 = self.conv8(x10)
        x12 = self.pool4(x11)
        x13 = self.conv9(x12)
        x14 = self.conv10(x13)
        x15 = self.up1(x14)
        x15 = keras.layers.Concatenate()([x15, x11])
        x16 = self.conv11(x15)
        x17 = self.conv12(x16)
        x18 = self.up2(x17)
        x18 = keras.layers.Concatenate()([x18, x8])
        x19 = self.conv13(x18)
        x20 = self.conv14(x19)
        x21 = self.up3(x20)
        x21 = keras.layers.Concatenate()([x21, x5])
        x22 = self.conv15(x21)
        x23 = self.conv16(x22)
        x24 = self.up4(x23)
        x24 = keras.layers.Concatenate()([x24, x2])
        x25 = self.conv17(x24)
        x26 = self.conv18(x25)
        x27 = self.conv19(x26)
        return x27
 
    def freeze_model(self, trainable = False,  which="all"):
        """Freeze or thaw the layers of the model"""
        if which == "all":
            index_slice = slice(None)
        elif which == "encoder":
            index_slice = slice(18)
        elif which == "decoder":
            index_slice = slice(18, None)

        for layer in self.layers[index_slice]:
            layer.trainable = trainable

    def get_representation_model(self, inputs):
        """Get the model with output being the representations from the decoder upsampled and concatenated"""
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.pool1(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.pool2(x5)
        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.pool3(x8)
        x10 = self.conv7(x9)
        x11 = self.conv8(x10)
        x12 = self.pool4(x11)
        x13 = self.conv9(x12)
        x14 = self.conv10(x13)
        x15 = self.up1(x14)
        x15 = keras.layers.concatenate([x11, x15])
        x16 = self.conv11(x15)
        x17 = self.conv12(x16)
        x18 = self.up2(x17)
        x18 = keras.layers.concatenate([x8, x18])
        x19 = self.conv13(x18)
        x20 = self.conv14(x19)
        x21 = self.up3(x20)
        x21 = keras.layers.concatenate([x5, x21])
        x22 = self.conv15(x21)
        x23 = self.conv16(x22)
        x24 = self.up4(x23)
        x24 = keras.layers.concatenate([x2, x24])
        x25 = self.conv17(x24)
        x26 = self.conv18(x25)
        x27 = self.conv19(x26)

        o1 = keras.layers.UpSampling2D()(x15)
        o2 = keras.layers.UpSampling2D()(x18)
        o3 = keras.layers.UpSampling2D()(x21)
        o4 = keras.layers.UpSampling2D()(x24)
        o5 = keras.layers.UpSampling2D()(x27)

        output = keras.layers.concatenate([o1, o2, o3, o4, o5])

        return keras.Model(inputs=inputs, outputs=output)

class CPT_regression(keras.Model):

    def __init__(self, encoder):
        super(CPT_regression, self).__init__()
        self.encoder = encoder
        self.conv1 = keras.layers.Conv2D(64, 3, activation="relu", padding="valid")
        self.conv2 = keras.layers.Conv2D(64, 3, activation="relu", padding="valid")
        self.pool1 = keras.layers.MaxPooling2D()
        self.conv3 = keras.layers.Conv2D(128, 3, activation="relu", padding="valid")
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512, activation="relu")
        self.dense2 = keras.layers.Dense(512, activation="relu")
        self.dense3 = keras.layers.Dense(1, activation="linear")
    
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


if __name__ == "__main__":
    
    seis_dataset = SeismicDataset(dir_TWT, dir_VEL, dir_DPT)
    match_dict = seis_dataset.load_match_dict()

    # for file_id, files in match_dict.items():
    #     X = seis_dataset.load_segy_tensor(files["TWT"])
    #     # y = seis_dataset.load_segy_tensor(files["VEL"])
    #     break
    
    # plt.imshow(X[0:1000,:].numpy().T, cmap="gray")
    # plt.show()

    # X = tf.expand_dims(X, axis=-1)
    # X = tf.expand_dims(X, axis=0)
    # # Create mosaic of 64x64 patches of X
    # X = tf.image.extract_patches(X, sizes=[1, 64, 64, 1],
    #                              strides=[1, 64, 64, 1],
    #                              rates=[1, 1, 1, 1], padding="VALID")

    # X = tf.reshape(X, [-1, 64, 64, 1])

    # # Save X
    # with open("Morven_seismics/X.pickle", "wb") as f:
    #     pickle.dump(X, f)
    

    # Load X
    with open("Morven_seismics/X.pickle", "rb") as f:
        X = pickle.load(f)
    

    print('Defining model...')
    unet = Unet()

    print('Compiling model...')
    unet.compile(optimizer="adam", loss="mae")
    print('Building model...')
    unet.build(input_shape=(None, 64, 64, 1))
    unet.summary()

    print('Training model...')
    unet.fit(X, X, epochs=10, batch_size=1)

    unet.save("Morven_seismics/unet")

    unet = keras.models.load_model("Morven_seismics/unet")
