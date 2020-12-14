import numpy as np
import tensorflow as tf
from keras.layers import Input, Activation, BatchNormalization, Conv2D, Layer, Reshape, Flatten, Lambda
from keras.models import Model

# Custom keras layer to form normal equation
class FormNormalEquations(Layer):
    def __init__(self, **kwargs):
        super(FormNormalEquations, self).__init__(**kwargs)
        initializer = tf.keras.initializers.Constant(1e-2)
        self.lamb = self.add_weight(name='lamb', shape=(1,), initializer=initializer, trainable=True)

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

    def call(self, inputs):
        X, M, RHS = inputs
        X = tf.zeros_like(X)

        def body(rTr,X,r,p):
            Ap    = tf.linalg.matvec(M,p)
            rTr   = tf.reduce_sum(r*r)
            alpha = rTr/tf.reduce_sum(p*Ap)
            X     = X+alpha*p
            r     = r-alpha*Ap
            beta  = tf.reduce_sum(r*r)/rTr
            p     = r+beta*p
            return tf.cast(tf.reduce_sum(r*r),dtype='float32'),X,r,p

        cond = lambda rTr,*_: rTr > 1e-10
        r = tf.subtract(RHS,tf.linalg.matvec(M,X))
        p = tf.identity(r)
        rTr = tf.cast(tf.reduce_sum(r*r),dtype='float32')
        loopinputs = rTr,X,r,p
        out  = tf.while_loop(cond,body,loopinputs,name='CGwhile',parallel_iterations=1,maximum_iterations=self.niter)[1]
        return out

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
            cnn_layers['a'+str(n)] = Activation('relu',name='a'+str(n))
    return cnn_layers

# Construct full deep learning model
def nn_model(input_shape,A_shape,n_layers,n_iter,n_CGiter,training):
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)
    A       = Input(A_shape)

    nrows = input_shape[0]
    ncols = input_shape[1]

    Flat_layer = Flatten(name='Flatten')
    Resh_layer = Reshape((nrows,ncols,1),name='Reshape')
    Cnn_layers = cnn(n_layers,training)
    ATb        = Flat_layer(X_input)
    ATA        = Lambda(lambda A : tf.matmul(A,A,transpose_a=True),name='ATA')(A)
    Normeq     = FormNormalEquations(name='NormEq')
    CG_init    = Conjugate_Gradient(100,name='CG_init')
    CG_layer   = Conjugate_Gradient(n_CGiter,name='CG')

    X = Lambda(lambda X : tf.zeros_like(X),name='Input')(X_input)

    # Ravel
    X = Flat_layer(X)
    # Create normal equations
    RHS = ATb
    M   = ATA
    # Call conjugate gradient to solve normal equations
    X = CG_init([X,M,RHS])

    for k in range(n_iter):
        # Form image
        Z = Resh_layer(X)
        # Calculate prior first
        for n in range(n_layers):
            Z = Cnn_layers['conv'+str(n)](Z)
            Z = Cnn_layers['bn'+str(n)](Z)
            if(n < n_layers-1):
                Z = Cnn_layers['a'+str(n)](Z)
        # Ravel
        Z = Flat_layer(Z)
        # Create normal equations
        RHS = Normeq([ATb,Z])
        M   = Lambda(lambda ATA : Normeq.getM(ATA),name='Get_M_'+str(k))(ATA)
        # Call conjudate gradient to solve normal equations
        X = CG_layer([X,M,RHS])

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = [X_input,A] , outputs = X, name='MoDL')

    return model
