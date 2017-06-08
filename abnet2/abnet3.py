from __future__ import division, print_function
import numpy as np
#import lasagne
#import theano
#import theano.tensor as T
#import lasagne.layers as layers
#import lasagne.nonlinearities as nl
#import lasagne.objectives
import time
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import torch.optim as optim

# TODO: save all distances calculated for the validation data
# so that this matrix can be re-used for computing auc/eer


epsilon = np.finfo(np.float32).eps

def iterate_minibatches_folder(inputs_folder, batchsize=1000, shuffle=False):
    """Generate mini bacthes from each file in the inputs folder
    """
    num_batch_files = len([name for name in os.listdir(inputs_folder) if os.path.isfile(name)])
    
    if shuffle:
        indices = np.arrange(len_data)
        np.random.shuffle(indices)
    if len_data < batch_size:
        print(" cndjncd") 



def iterate_minibatches(inputs, batchsize=1000, shuffle=False):
    """Generate mini batches from datasets
    """
    len_data = len(inputs[0])
    assert all([len(i) == len_data for i in inputs[1:]]), [len(i) for i in inputs]
    if shuffle:
        indices = np.arange(len_data)
        np.random.shuffle(indices)
    if len_data < batchsize:
        # input shorter than a single batch
        if shuffle:
            excerpt = indices
        else:
            excerpt = slice(len_data)
        yield [inp[excerpt] for inp in inputs]
    # for start_idx in range(0, len_data - batchsize + 1, batchsize):
    for start_idx in range(0, len_data, batchsize):
        if shuffle:
            excerpt = indices[start_idx:min(start_idx + batchsize, len_data)]
        else:
            excerpt = slice(start_idx, min(start_idx + batchsize, len_data))
        yield [inp[excerpt] for inp in inputs]


def train_iteration(data_train, data_val, train_fn, val_fn):
    """Generic train iteration: one full pass over the data"""
    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(data_train, 500, shuffle=True):
        train_err += train_fn(batch)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(data_val, 500, shuffle=False):
        aux = val_fn(batch)
        try:
            err, acc = aux
            val_acc += acc
        except:
            err = aux[0]
            val_acc = None
        val_err += err
        val_batches += 1

    return train_err / train_batches, val_err / val_batches

def train_iteration_by_batch(data_train, data_val, train_fn, val_fn):
    """Generic train iteration: one full pass over the data"""
    train_err = 0
    train_batches = 0
    for batch in data_train:
        train_err += train_fn(batch)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in data_val:
        aux = val_fn(batch)
        try:
            err, acc = aux
            val_acc += acc
        except:
            err = aux[0]
            val_acc = None
        val_err += err
        val_batches += 1

    return train_err / train_batches, val_err / val_batches


def train(data_train, data_val, train_fn, val_fn, network, max_epochs=100, patience=20, save_run=True, eval_fn=None):
    """Generic train strategy for neural networks
    (batch training, train/val sets, patience)

    Trains a neural network according to some data (list of inputs, targets)
    and a train function and an eval function on that data"""
    print("training...")

    run = []
    best_model = None
    best_epoch = None
    if patience <= 0:
        patience = max_epochs
    patience_val = 0
    best_val = None

    for epoch in range(max_epochs):
        start_time = time.time()
        train_err, val_err = train_iteration(data_train, data_val,
                                             train_fn, val_fn)

        run.append(layers.get_all_param_values(network))
        if np.isnan(val_err) or np.isnan(train_err):
            print("Train error or validation error is NaN, "
                  "stopping now.")
            break
        # Calculating patience
        if best_val == None or val_err < best_val:
            best_val = val_err
            patience_val = 0
            best_model = layers.get_all_param_values(network)
            best_epoch = epoch
        else:
            patience_val += 1
            if patience_val > patience:
                print("No improvements after {} iterations, "
                      "stopping now".format(patience))
                break

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        try:
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
        except:
            if eval_fn != None:
                acc = eval_fn(*data_val)
                print("  validation accuracy:\t\t{:.2f} %".format(acc))

    return best_model, best_epoch, run


def train_by_batch(train_pairs, validation_pairs,features, train_fn, val_fn, network,get_next_batch=None, max_epochs=100, patience=20, save_run=True, eval_fn=None):
    """Generic train strategy for neural networks
    (batch training, train/val sets, patience)
    Inject get_next_batch function that produces all the batches from folder
    Trains a neural network according to some data (list of inputs, targets)
    and a train function and an eval function on that data"""
    print("training...")

    run = []
    best_model = None
    best_epoch = None
    if patience <= 0:
        patience = max_epochs
    patience_val = 0
    best_val = None

    for epoch in range(max_epochs):
        start_time = time.time()
        data_train = get_next_batch(features,train_pairs)
        data_val = get_next_batch(features,validation_pairs)
        train_err, val_err = train_iteration_by_batch(data_train, data_val,
                                             train_fn, val_fn)

        run.append(layers.get_all_param_values(network))
        if np.isnan(val_err) or np.isnan(train_err):
            print("Train error or validation error is NaN, "
                  "stopping now.")
            break
        # Calculating patience
        if best_val == None or val_err < best_val:
            best_val = val_err
            patience_val = 0
            best_model = layers.get_all_param_values(network)
            best_epoch = epoch
        else:
            patience_val += 1
            if patience_val > patience:
                print("No improvements after {} iterations, "
                      "stopping now".format(patience))
                break

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        try:
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
        except:
            if eval_fn != None:
                acc = eval_fn(*data_val)
                print("  validation accuracy:\t\t{:.2f} %".format(acc))

    return best_model, best_epoch, run



class ABnet(nn.Module):
    """Siamese neural network written in Pytorch
    """
    def __init__(self, dims, nonlinearities=None, dropouts=None,
                 update_fn=None, batch_norm=True,
                 loss_type='cosine_margin', margin=0.8, lstm_layers=False):
        """Initialize a Siamese neural network

        Parameters:
        -----------
        update_fn: theano function with 2 arguments (loss, params)
            Update scheme, default to adadelta
        nonlinearities: Non Linearities between each layer, default : Only Sigmoid
        batch_norm: bool
            Do batch normalisation on first layer, default to false
        """
        super(ABnet, self).__init__()
        assert len(dims) >= 3, 'Not enough dimmensions'
        self.layers = []
        num_layers = len(dims)
        if nonlinearities == None:
           nonlinearities = [nn.Sigmoid()] * (len(dims) -1)
        for index_layer in range(1,num_layers):    
            self.layers.append(nn.Sequential(
                            nn.Linear(dims[index_layer-1],dims[layer])),
                            nonlinearities[index_layer]
                            )
        self.batch_norm = batch_norm
        self.loss_type = loss_type

    def forward(self, input1, input2):
        """ Forward function
        """
        output1 = self.layers[0](input1)
        output2 = self.layers[0](input2)
        for index_layer in range(1,num_layers-1):
            output1 = self.layers[index_layer](output1)
            output2 = self.layers[index_layer](output2)

        return output1, output2

def train_model(model,margin=0.85):
    """Train model function
    """
    # create your optimizer
    optimizer = optim.Adam(model.parameters, lr = 0.0001)
    optimizer.zero_grad()
    loss_fn = nn.CosineEmbeddingLoss(margin=margin, size_average=False)

#### TODO plug samper and iterator for batches here









    def change_update(self, update_fn=lasagne.updates.adadelta):
        self.updates = update_fn(self.loss, self.params)
        self.train_fn = theano.function(
            [self.input_var1, self.input_var2, self.target_var],
            self.loss, updates=self.updates)
        self.train_and_eval_fn = theano.function(
            [self.input_var1, self.input_var2, self.target_var],
            self.loss, updates=self.updates)
        self.val_fn = theano.function(
            [self.input_var1, self.input_var2, self.target_var],
            [self.test_loss])
        self.eval_fn = theano.function(
            [self.input_var1], self.test_prediction1)

    def train(self, X_train1, X_train2, y_train, X_val1, X_val2, y_val, max_epochs=500, patience=20):
        def train_batch(batch_data):
            return self.train_fn(*batch_data)
        def val_batch(batch_data):
            return self.val_fn(*batch_data)

        best_weights, best_epoch, run = train(
            [X_train1, X_train2, y_train], [X_val1, X_val2, y_val],
            train_batch, val_batch,
            self.network, max_epochs=max_epochs, patience=patience)
        layers.set_all_param_values(self.network, best_weights)
        return run, best_epoch

    def train_by_batch(self,train_pairs, validation_pairs,features,get_next_batch=None, max_epochs=500, patience=20):
        def train_batch(batch_data):
            return self.train_fn(*batch_data)
        def val_batch(batch_data):
            return self.val_fn(*batch_data)

        best_weights, best_epoch, run = train_by_batch(train_pairs, validation_pairs,features, train_batch, val_batch,
              self.network, get_next_batch=get_next_batch, max_epochs=max_epochs, patience=patience)
        layers.set_all_param_values(self.network, best_weights)
        return run, best_epoch


    def evaluate(self, X_test):
        embs = []
        for batch in iterate_minibatches([X_test], 500, shuffle=False):
            inputs = batch[0]
            emb = self.eval_fn(inputs)
            embs.append(emb)
        if len(embs) > 1:
            return np.concatenate(embs)
        else:
            return embs[0]

    def eer(self, X1, X2, y):
        """Estimation of the equal error rate
        Returns % of correctly classified data with optimal threshold
        """
        distances = self._distances(X1, X2)
        distances_same = distances[y==1]
        distances_diff = distances[y==0]
        # finding optimal threshold
        threshold = np.mean(distances)
        t_min = np.min(distances)
        t_max = np.max(distances)
        misclassified_diff = np.sum(distances_diff < threshold)
        misclassified_same = np.sum(distances_same > threshold)
        while misclassified_diff - misclassified_same > 1:
            misclassified_diff = np.sum(distances_diff < threshold)
            misclassified_same = np.sum(distances_same > threshold)
            if misclassified_diff > misclassified_same:
                t_max = threshold
                threshold = (threshold + t_min) / 2
            else:
                t_min = threshold
                threshold = (threshold + t_max) / 2

        error_rate = (misclassified_diff + misclassified_same) / y.shape[0]
        return (1 - error_rate) * 100

    def abx(self, X1, X2, y):
        """Discriminability estimation"""
        raise NotImplementedError

    def _distances(self, X1, X2):
        distances = []
        cos_theano = cos_sim(self.prediction1, self.prediction2)
        cos_fn = theano.function([self.input_var1, self.input_var2],
                                 cos_theano)
        for batch in iterate_minibatches([X1, X2]):
            distances.append((1 - cos_fn(*batch)) / 2)
        if len(distances) > 1:
            distances = np.concatenate(distances)
        else:
            distances = distances[0]
        return distances

    def auc(self, X1, X2, y):
        """Area under the ROC curve
        returns (auc * 2 - 1) * 100 (effective score in percent,
        chance at 0%)"""
        nbins = 100
        distances = self._distances(X1, X2)
        distances_same = distances[y==1]
        distances_diff = distances[y==0]
        thresholds = np.linspace(np.min(distances), np.max(distances), nbins)
        misclassified_diff, misclassified_same = np.empty((nbins,), dtype=float), np.empty((nbins,), dtype=float)
        for idx, threshold in enumerate(thresholds):
            misclassified_diff[idx] = float(np.sum(distances_diff < threshold)) / distances_diff.shape[0]
            misclassified_same[idx] = float(np.sum(distances_same > threshold)) / distances_same.shape[0]
        area = 0.
        for i in range(nbins - 1):
            x0 = misclassified_diff[i]
            x1 = misclassified_diff[i+1]
            y0 = misclassified_same[i]
            y1 = misclassified_same[i+1]
            area += (y0 + y1) * (x1 - x0) / 2
            # print(area)
        area = 1 - area
        return ((area * 2) - 1) * 100

    # def pretrain_AE(self, AE=1, nonlinearity=None):
    #     """Pretrain with autoencoder

    #     Parameters:
    #     ----------
    #     AE: str or int, number of additional hidden layers before output,
    #         if AE='mirror', it will use n_hid - 1 layers (n_hid being the
    #         number of hidden layers)
    #     nonlinearity: callable, if None, the non linearity of the penultimate
    #         layer will be used (unless 'mirror' is chosen, in which case the
    #         non linearities are chosen in mirror"""
    #     ae_network = self.network[0]
    #     if nonlinearity == None:
    #         #TODO get the correct layer everytime
    #         nonlinearity = ae_network.inputlayer.nonlinearity
    #     if AE isinstance(str) and AE == 'mirror':
    #         raise NotImplementedError
    #     else:
    #         for i in range(AE):
    #             raise NotImplementedError
    #             ae_network = layers.DenseLayer(
    #                 ae_network, num_units=dim,
    #                 W=lasagne.init.GlorotUniform(),
    #                 nonlinearity=nonlinearity)
    #     #dim = inputdim
    #     ae_network = layers.DenseLayer(
    #                 ae_network, num_units=dim,
    #                 W=lasagne.init.GlorotUniform(),
    #                 nonlinearity=nonlinearity)

    #     ae_params = layers.get_all_params(ae_network, trainable=True)
    #     ae_prediction = layers.get_output(ae_network)
    #     ae_loss = T.mean((ae_prediction - self.input_var1).norm(2, axis=-1))
    #     ae_updates = lasagne.updates.adadelta(ae_loss, ae_params)
    #     ae_train = theano.function(self.input_var1, ae_loss,
    #                                updates=ae_updates)
    #     ae_val = theano.function(self.input_var1, ae_loss)

    #     def train_batch(batch_data):
    #         return self.train_fn(*batch_data)
    #     def val_batch(batch_data):
    #         return self.val_fn(*batch_data)
    #     best_weights, run = train(
    #         [X_train], [X_val],
    #         train_batch, val_batch,
    #         ae_network, max_epochs=100, patience=10)
    #     layers.set_all_param_values(self.ae_network, best_weights)


#def loss_fn(prediction1, prediction2, targets, loss='cosine_margin', margin=0.15,
#            return_similarities=False):
#    if loss == 'cosine_margin':
        # part linear loss function:
#        x0 = 1 - margin
        # cos_sim_same = T.switch(cos_sim>0.9, 0, 1 - (cos_sim / x0))
#        cos_sim_diff = T.switch(cos_sim < x0, 0, (cos_sim - x0) / (1 - x0))
#        cos_sim_same = 1. - cos_sim
        # cos_sim_diff = T.switch(cos_sim<0.9, 0, cos_sim - x0)
#    else:
#        raise NotImplementedError 

#   if return_similarities:
#        return T.mean(T.switch(targets, cos_sim_same, cos_sim_diff)), cos_sim
#    else:
#        return T.mean(T.switch(targets, cos_sim_same, cos_sim_diff))
