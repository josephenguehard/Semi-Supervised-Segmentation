import nibabel as nib
import numpy as np
import random
from clustering_layer import ClusteringLayer
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Cropping3D
from keras.layers import Dropout, BatchNormalization
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.regularizers import l1_l2
from keras.models import Model
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
import itertools
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from IPython.display import display_html

def config():
    num_classes = 3

    patience = 5
    model_filename = 'models/iSeg2017/outrun_step_{}.h5'
    csv_filename = 'log/iSeg2017/outrun_step_{}.cvs'

    nb_epoch = 20
    validation_split = 0.2

    class_mapper = {0 : 0, 10 : 0, 150 : 1, 250 : 2}
    class_mapper_inv = {0 : 0, 1 : 10, 2 : 150, 3 : 250}

    PATCH_SHAPE = 27
    EXTRACTION_SHAPE = 9

    n_known = 6

    return num_classes, patience, model_filename, csv_filename, nb_epoch, validation_split, class_mapper, class_mapper_inv, PATCH_SHAPE, EXTRACTION_SHAPE, n_known

def generate_model(num_classes) :
    init_input = Input((2, 27, 27, 27))

    x = Conv3D(25, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(init_input)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(x)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(y)
    z = BatchNormalization()(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(z)
    z = BatchNormalization()(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(z)
    z = BatchNormalization()(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (3, 3)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(concat)
    fc = BatchNormalization()(fc)
    fc = PReLU()(fc)
    fc = Dropout(0.25)(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(fc)
    fc = BatchNormalization()(fc)
    fc = PReLU()(fc)
    fc = Dropout(0.25)(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(fc)
    fc = BatchNormalization()(fc)
    fc = PReLU()(fc)
    fc = Dropout(0.25)(fc)

    pred = Conv3D(num_classes, kernel_size=(1, 1, 1), kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=1e-6, l2=1e-4))(fc)
    fc = BatchNormalization()(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, 9 * 9 * 9))(pred)
    pred = Permute((2, 1))(pred)
    pred = ClusteringLayer(num_classes)(pred)

    model = Model(inputs=init_input, outputs=pred)
    model.compile(
        loss='kld',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model

# General utils for reading and saving data
def get_filename(set_name, case_idx, input_name, loc='datasets') :
    if(loc=='datasets'):
        pattern = '/data/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr'
    else:
        pattern = '{0}/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr'
    return pattern.format(loc, set_name, case_idx, input_name)

def get_set_name(case_idx) :
    return 'Training' if case_idx < 11 else 'Testing'

def read_data(case_idx, input_name, loc='datasets') :
    set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)

    return nib.load(image_path)

def read_vol(case_idx, input_name, loc='datasets') :
    image_data = read_data(case_idx, input_name, loc)

    return image_data.get_data()[:, :, :, 0]

def save_vol(segmentation, case_idx, loc='results') :
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, 'T1')
    
    #file_shape = input_image_data.shape[:3] + (3,)
    file_shape = input_image_data.shape

    segmentation_vol = np.empty(file_shape)
    #segmentation_vol[:144, :192, :256, :] = segmentation
    segmentation_vol[:144, :192, :256, 0] = segmentation
    
    filename = get_filename(set_name, case_idx, 'label', loc)
    nib.save(nib.analyze.AnalyzeImage(
        segmentation_vol.astype('uint8'), input_image_data.affine), filename)


def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

def build_set(T1_vols, T2_vols, label_vols, extraction_step=(9, 9, 9), num_classes=3) :
    patch_shape = (27, 27, 27)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 2, 27, 27, 27))
    y = np.zeros((0, 9 * 9 * 9, num_classes))
    for idx in range(len(T1_vols)) :
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)
        
        # Filtering extracted patches

        random.seed(1)
        #valid_idxs = random.sample(list(valid_idxs[0]), int(len(valid_idxs[0])/5.))
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 2, 27, 27, 27))))
        y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes))))

        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = np_utils.to_categorical(label_patches[i].flatten(), num_classes)

        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
        del T1_train

        # Sampling strategy: reject samples which labels are only zeros
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step)
        x[y_length:, 1, :, :, :] = T2_train[valid_idxs]
        del T2_train
        
        print("Finished segmentation of case # {}".format(idx))
    return x, y

def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]

    idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]

    return itertools.product(*idxs)

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape[:4]

    assert len(patch_shape) - 1 == len(expected_shape)

    #reconstructed_img = np.zeros(expected_shape + (3,))
    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

# computing an auxiliary target distribution
def target_distribution(pred):
    weight = pred ** 2 / pred.sum(0).sum(0)
    return (weight.T / weight.sum(2).T).T

def DSC(im1, im2, indice):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = (im1==indice)
    im2 = (im2==indice)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def save_vol_modif(segmentation, case_idx, loc='results', num_classes=3) :
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, 'T1')
    
    file_shape = input_image_data.shape[:3] + (num_classes,)

    segmentation_vol = np.empty(file_shape)
    segmentation_vol[:144, :192, :256, :] = segmentation
    
    filename = get_filename(set_name, case_idx, 'label', loc)
    nib.save(nib.analyze.AnalyzeImage(
        segmentation_vol, input_image_data.affine), filename)

def generate_indexes_modif(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]

    idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]
    idxs[3] = [3]

    return itertools.product(*idxs)

def reconstruct_volume_modif(patches, expected_shape) :
    patch_shape = patches.shape[:5]

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes_modif(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord)-1)]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

def read_vol_modif(case_idx, input_name, loc='datasets') :
    image_data = read_data(case_idx, input_name, loc)

    return image_data.get_data()[:, :, :, :]

def extract_patches_modif(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

def build_set_modif(T1_vols, T2_vols, label_vols, n_known, is_pseudo=False, extraction_step=(9, 9, 9), num_classes=3) :
    patch_shape = (27, 27, 27)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 2, 27, 27, 27), dtype='float32')
    y = np.zeros((0, 9 * 9 * 9, num_classes))
    for idx in range(len(T1_vols)) :
        y_length = len(y)

        label_patches_1 = extract_patches_modif(label_vols[idx,:,:,:,0], patch_shape, extraction_step)
        label_patches_2 = extract_patches_modif(label_vols[idx,:,:,:,1], patch_shape, extraction_step)
        label_patches_3 = extract_patches_modif(label_vols[idx,:,:,:,2], patch_shape, extraction_step)
        label_patches = np.stack((label_patches_1, label_patches_2, label_patches_3), axis=4)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        if not is_pseudo:
            random.seed(1)
            valid_idxs = np.where(np.sum(label_patches[:,:,:,:,1:], axis=(1, 2, 3, 4)) != 0)
            valid_idxs = random.sample(list(valid_idxs[0]), int(len(valid_idxs[0])/5.))
        if is_pseudo:
            valid_idxs = np.unique(np.where(label_patches[:,:,:,:,1:]>0.5)[0])

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        print(label_patches.shape[0])

        x = np.vstack((x, np.zeros((len(label_patches), 2, 27, 27, 27), dtype='float32')))
        y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes))))

        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = label_patches[i].reshape((9*9*9,3))

        del label_patches, label_patches_1, label_patches_2, label_patches_3

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = T1_train[valid_idxs].astype('float32')
        del T1_train

        # Sampling strategy: reject samples which labels are only zeros
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step)
        x[y_length:, 1, :, :, :] = T2_train[valid_idxs].astype('float32')
        del T2_train
        
        print("Finished segmentation of case # {}".format(idx))
    return x, y

def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
