# Flower Image Classification

This project aims to classify images of 5 different varieties of flowers using a convolutional neural network (CNN) built with TensorFlow and Keras. The dataset used is the "Cats vs. Dogs" dataset, which is available through TensorFlow's dataset repository.

## Dataset



The dataset we have downloaded has the following directory structure:

```
flower_photos
|__ daisy
|__ dandelion
|__ roses
|__ sunflowers
|__ tulips
```

As you can see, there are no folders containing training and validation data. Therefore, we will have to create our own training and validation sets. Let's write some code that will do this.

The code below creates a `train` and a `val` folder, each containing 5 folders (one for each type of flower). It then moves the images from the original folders to these new folders such that 80% of the images go to the training set and 20% of the images go into the validation set. In the end, our directory will have the following structure:

```
flower_photos
|__ daisy
|__ dandelion
|__ roses
|__ sunflowers
|__ tulips
|__ train
    |______ daisy: [1.jpg, 2.jpg, 3.jpg ....]
    |______ dandelion: [1.jpg, 2.jpg, 3.jpg ....]
    |______ roses: [1.jpg, 2.jpg, 3.jpg ....]
    |______ sunflowers: [1.jpg, 2.jpg, 3.jpg ....]
    |______ tulips: [1.jpg, 2.jpg, 3.jpg ....]
 |__ val
    |______ daisy: [507.jpg, 508.jpg, 509.jpg ....]
    |______ dandelion: [719.jpg, 720.jpg, 721.jpg ....]
    |______ roses: [514.jpg, 515.jpg, 516.jpg ....]
    |______ sunflowers: [560.jpg, 561.jpg, 562.jpg .....]
    |______ tulips: [640.jpg, 641.jpg, 642.jpg ....]
```

Since we don't delete the original folders, they will still be in our `flower_photos` directory, but they will be empty. The code below also prints the total number of flower images we have for each type of flower.

The dataset we downloaded contains images of 5 types of flowers:

1. Rose
2. Daisy
3. Dandelion
4. Sunflowers
5. Tulips



## Data Augumentation

Overfitting often occurs when we have a small number of training examples. One way to fix this problem is to augment our dataset so that it has sufficient number and variety of training examples.
Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples through random transformations that yield believable-looking images. 

The goal is that at training time, the model will never see the exact same picture twice. This exposes the model to more aspects of the data, allowing it to generalize better.

![image](https://github.com/user-attachments/assets/9fb6703a-3999-4b8f-bc88-4355687c75e1)


### Flipping the image horizontally

![image](https://github.com/user-attachments/assets/3e1722be-2198-43ae-9c1b-4c4e2d590ef4)


### Rotating the image

![image](https://github.com/user-attachments/assets/32ccb1aa-fb7c-4bd2-822d-3f5e46ae94ca)


### Applying Zoom

![image](https://github.com/user-attachments/assets/2d5bb6d3-6097-44da-ad7a-b34489243c1c)



## Installation

To get started with this project, you'll need to have TensorFlow and other dependencies installed. You can install the necessary packages using pip:

```bash
pip install tensorflow matplotlib numpy
```

## Project Structure

- **Data Preparation**: Downloads and extracts the dataset, and sets up data generators for training and validation.
- **Model Creation**: Defines and compiles a CNN model for image classification.
- **Training**: Trains the model using the training data and evaluates it on the validation data.
- **Evaluation**: Generates plots for training and validation accuracy and loss, and visualizes predictions on validation images.


## Model Summary


```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 150, 150, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2  (None, 75, 75, 16)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 75, 75, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 37, 37, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 18, 18, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 9, 9, 128)         0         
 g2D)                                                            
                                                                 
 conv2d_4 (Conv2D)           (None, 9, 9, 256)         295168    
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 4, 4, 256)         0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 4096)              0         
                                                                 
 dropout (Dropout)           (None, 4096)              0         
                                                                 
 dense (Dense)               (None, 512)               2097664   
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 5)                 2565      
                                                                 
=================================================================
Total params: 2492837 (9.51 MB)
Trainable params: 2492837 (9.51 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

## Results

The trained model achieved an accuracy of **81.90%** on the validation dataset.


##Visualizing results of the training

![image](https://github.com/user-attachments/assets/26a0ce05-8238-4766-bbaa-b84b6bc3bc63)



### Example Predictions

Here are 100 images from the validation dataset with their predictions:

![image](https://github.com/user-attachments/assets/d10aec98-e0ae-4609-9f06-d9fd951e35ac)




## Usage

1. **Prepare the Dataset**: Ensure the dataset is downloaded and extracted properly.
2. **Run the Code**: Execute the script to train the model and evaluate its performance.
3. **View Results**: Check the generated plots and predictions to analyze the model's performance.

## Saving and Downloading the Model

The trained model is saved as `Flower_Type.keras` and can be downloaded from:

[Download Flower_Type.keras](https://drive.google.com/drive/folders/1l1cikFWNjV_LdurGFtYmf199IiW6JkBE?usp=sharing)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

