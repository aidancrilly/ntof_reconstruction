import numpy as np
import tensorflow as tf
from keras.layers import Input, Activation, BatchNormalization, Conv2D, Layer, Reshape, Flatten, Lambda
from keras.models import Model

# Custom keras layer to form normal equation
class FormNormalEquations(Layer):
    def __init__(self, **kwargs):
        super(FormNormalEquations, self).__init__(**kwargs)
        self.lamb = self.add_weight(name='lamb', shape=(1,), initializer='uniform', trainable=True)

    def call(self, inputs):
        ATb, Z = inputs
        return tf.add(ATb,tf.multiply(self.lamb,Z))

    def getM(self, ATA):
        return tf.add(ATA,tf.multiply(self.lamb,tf.eye(ATA.shape[-1])))

# Custom keras layer to solve normal equation using Conjugate Gradients method
class Conjugate_Gradient(Layer):
    def __init__(self, niter, **kwargs):
        super(Conjugate_Gradient, self).__init__(**kwargs)
        self.niter = niter

    def call(self, X, M, RHS):
        r = tf.subtract(RHS,tf.linalg.matvec(M,X))
        p = tf.identity(r)
        n = 0
        while(n < self.niter and tf.reduce_sum(r*r) > 1e-10):
            Ap    = tf.linalg.matvec(M,p)
            rTr   = tf.reduce_sum(r*r)
            alpha = rTr/tf.reduce_sum(p*Ap)
            X     = X+alpha*p
            r     = r-alpha*Ap
            beta  = tf.reduce_sum(r*r)/rTr
            p     = r+beta*p
            n    += 1
        return X

# Construct CNN for image regularisation
def cnn(n_layers,training):
    cnn_layers = {}
    # CONV -> BN -> RELU Block
    for n in range(n_layers):
        if(n < n_layers-1):
            cnn_layers['conv'+str(n)] = Conv2D(32, (5, 5), strides = (1, 1), padding = 'same', name = 'conv'+str(n))
        else:
            cnn_layers['conv'+str(n)] = Conv2D(1, (5, 5), strides = (1, 1), padding = 'same', name = 'conv'+str(n))
        # Apply batchnorm
        cnn_layers['bn'+str(n)] = BatchNormalization(trainable = training, axis = -1, name = 'bn'+str(n))
        # Final layer is given no activation
        if(n < n_layers-1):
            cnn_layers['a'+str(n)] = Activation('relu')
    return cnn_layers

# Construct full deep learning model
def nn_model(input_shape,A_shape,n_layers,n_iter,n_CGiter,training):
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)
    A       = Input(A_shape)

    nrows = input_shape[0]
    ncols = input_shape[1]

    flat_layer = Flatten(name='Flatten')
    resh_layer = Reshape((nrows,ncols,1),name='Reshape')
    cnn_layers = cnn(n_layers,training)
    ATb        = flat_layer(X_input)
    ATA        = tf.matmul(A,A,transpose_a=True)
    normeq     = FormNormalEquations(name='NormEq')
    CG_init    = Conjugate_Gradient(5000,name='CG_init')
    CG_layer   = Conjugate_Gradient(n_CGiter,name='CG')

    X = X_input

    # Ravel
    X = flat_layer(X)
    # Create normal equations
    RHS = ATb
    M   = ATA
    # Call conjudate gradient to solve normal equations
    X = CG_init(X,M,RHS)

    for k in range(n_iter):
        # Form image
        Z = resh_layer(X)
        # Calculate prior first
        for n in range(n_layers):
            Z = cnn_layers['conv'+str(n)](Z)
            Z = cnn_layers['bn'+str(n)](Z)
            if(n < n_layers-1):
                Z = cnn_layers['a'+str(n)](Z)
        # Ravel
        Z = flat_layer(Z)
        # Create normal equations
        RHS = normeq((ATb,Z))
        M   = normeq.getM(ATA)
        # Call conjudate gradient to solve normal equations
        X = CG_layer(X,M,RHS)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = [X_input,A] , outputs = X, name='MoDL')

    return model
