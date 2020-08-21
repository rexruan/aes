#!/usr/bin/env python3
# encoding: utf-8

# script to train the model for aes

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from util import timestamp, Timer, stderr_print, loadbar, ConfusionMatrix, dictlist_to_csv
from ../data_preprocessing import datautil
from grader import Grader

torch.manual_seed(123)
data_set, fold, sent, text, block, unit, bi, num_layer = sys.argv[1:] 
#ann 1 1 0 all lstm 1 2

evaluate_on_test_data = True

feature_block = {
    'sent_morph':       ' '.join([str(i) for i in range(30)]),
    'sent_syntactic':   ' '.join([str(i) for i in range(30,41)]),
    'sent_lexical':     ' '.join([str(i) for i in range(41,51)]),
    'sent_general':     ' '.join([str(i) for i in range(51,63)]),
    'sent_all':         ' '.join([str(i) for i in range(63)]),
    'sent_k-best':      ' '.join([str(i) for i in '2 3 4 10 11 13 14 19 28 30 31 32 39 41 45 47 48 51 53 61 62'.split()]),
    
    'text_morph':       ' '.join([str(i) for i in range(30*3)]),     # 30 * 3 = 90
    'text_syntactic':   ' '.join([str(i) for i in range(30*3,41*3)]),# 11 * 3 = 33
    'text_lexical':     ' '.join([str(i) for i in range(41*3,51*3)]),# 10 * 3 = 30
    'text_general':     ' '.join([str(i) for i in range(51*3,191)]), # 12 * 3 + 2 = 38
    'text_all':         ' '.join([str(i) for i in range(191)]),       # 90 + 33 + 30 + 38 = 191
    'text_k-best':      ' '.join([str(int(i)*3+j) for i in '2 3 4 10 11 13 14 19 28 30 31 32 39 41 45 47 48 51 53 61 62'.split() for j in range(2)]) 

}
#hyperparameter dictionary for the models

hyperparams = {
    
    #experiment part:
    #(fold1-10, sent |& text, norm_std, feature block, neural network architecture: rnn units, bi|uni, layer_num )
    'experiment name':None,
    
    #'sent_num_features':feature_block['sent_morph'],
    #'sent_num_features':feature_block['sent_syntactic'],
    #'sent_num_features':feature_block['sent_lexical'],
    #'sent_num_features':feature_block['sent_general'],
    'sent_num_features':feature_block['sent_'+block],
    #'sent_num_features':feature_block['sent_k-best'],

    #text_feature: max_length 191 = 3 * 63  + 2
    #'text_num_features':feature_block['text_morph'],
    #'text_num_features':feature_block['text_syntactic'],
    #'text_num_features':feature_block['text_lexical'],
    #'text_num_features':feature_block['text_general'],
    'text_num_features':feature_block['text_'+block],
    #'text_num_features':feature_block['text_k-best'],
    
    #baseline
    'baseline':False,
    
    #sent_included
    'sent':sent,

    #text_included
    'text':text,
    
    #learning rate
    'learning_rate': 0.001,

    #numbers of the training epochs
    'number_of_epochs':200,

    #mini-batch size
    'batch_size': 20,

    #size of the hidden_layer for grade predicting
    #'hidden_nodes':100,
 
    #size of the hidden_layer
    'rnn_layer_size': 100,

    #number of hidden layers
    'rnn_layer_number': int(num_layer),

    #bidirectionality of rnn layer
    'bidirectional': bi,

    #RNN cell type:
    'cell_type':unit.lower(),
    #'cell_type':'LSTM',
    #'cell_type':'GRU',
    #'cell_type':'RNN',

    #dropout
    'dropout_rate': 0.5,

    #optimizer type
    'optimizer': optim.Adam,
    #'optimizer':optim.SGD,
    #'optimizer':optim.Adadelta,
    #'optimizer':optim.RMSprop,
    #'optimizer':optim.Adagrad,

    #loss function
    #'loss_function':nn.MSELoss,
    #'loss_function':nn.NLLLoss,
    'loss_function':nn.CrossEntropyLoss,

    #normalization:
    #'normalization':'original',
    'normalization':'std',

    #activation function
    #'activation':nn.Softmax
    #'activation':nn.LogSoftmax
    'activation':nn.ReLU()
    #'activation':nn.Tanh
    #'activation':nn.Sigmoid


}


dataparams = {
    #directory for output files: trained models, logs and stats
    'output_dir':'out',
    'save_log':True,
    'save_conf_matrix':True,
    'save_model':False,

    #data directory:
    'data_dir':'data',

    #data_set:
    'data_set': data_set,
    #'data_set':'ann',
    #'data_set':'ann_robert',

    #fold:
    'fold_n':fold,
    #'fold_n':'1',

    #
    'train_file':'train',
    'dev_file':'dev',
    'test_file':'test',

    # embedding file
    'emb_file':'embedding/emb.data.std.sig',
    #'emb_file':'embedding/emb.data.lower',
    #'emb_file':'embedding/emb.data.std',
    #'word_emb':'word.emb',


    #length of input text
    #shorter texts are padded to match this length
    #'text_length':50, #unsure if correct

    'grade_file':'grade_file'

}
hyperparams['sent'] = True if sent == '1' else False
hyperparams['text'] = True if text == '1' else False
hyperparams['bidirectional'] = True if bi== '1' else False
path = os.path.join(dataparams['data_dir'],dataparams['data_set'], dataparams['fold_n'])
dataparams['train_file'] = os.path.join(path, dataparams['train_file'])
dataparams['dev_file'] = os.path.join(path, dataparams['dev_file'])
dataparams['test_file'] = os.path.join(path, dataparams['test_file'])

dataparams['emb_file'] = dataparams['emb_file']



#TIMESTAMP = timestamp()
TIMESTAMP = '_'.join([data_set,fold,sent,text,block,unit,bi,num_layer])

def train(model, grade2i, train_loader, dev_loader, number_of_epochs, loss_function, optimizer, learning_rate, output_dir, conf_matrix, **kwargs):

    #Initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    #Set up variables for logging and timing
    n_train_batches = len(train_loader) # ._loader is a list of lists of text objects [[1,2,3],[4,5,6],[7]]
    n_dev_batches = len(dev_loader)
    total_iterations = number_of_epochs * (n_train_batches + n_dev_batches)

    training_log = []
    best_model = (None, None)

    train_confm = conf_matrix.copy()
    dev_confm = conf_matrix.copy()

    timer = Timer()
    timer.start()
    
    print('Training started.')
    print()
    print('epoch\ttr_loss\tva_loss\ttr_acc \tva_acc \ttr_f1  \tva_f1')
    
    for epoch in range(1, number_of_epochs + 1):

        #Switch model to training mode
        model = model.train()
        train_confm.reset()
        train_loss = 0

        #Training minibatch loop
        for batch_n, X in enumerate(train_loader):
            stderr_print('Epoch {:>3d}: Training    |{}| {}'.format(epoch,
                                                                    loadbar(batch_n / (n_train_batches - 1)),
                                                                    timer.remaining(total_iterations)), end='\r')
            model.zero_grad()
            #model.hidden = model.init_hidden(batch_size=len(X))


            #L is the lengths vector to indicate the text lengths
            Y_h = model(X)
            Y = torch.LongTensor([grade2i[t.grade] for t in X])
            #Y_label = torch.FloatTensor([[1 if i==(y-1) else 0 for i in range(len(grade2i))] for y in Y])
            
            #compute the loss and update the weights with gradient desecent
            loss = loss_function(Y_h, Y)
            #print(epoch, loss.item())
            #print(Y_h.max(dim=1)[1] == Y_label.max(dim=1)[1])
            loss.backward()
            optimizer.step()
            
            pred_grades = Y_h.max(dim=1)[1]
            train_confm.add(pred_grades, Y)
            train_loss += loss
            

        #train_loss /= len([_ for sub in train_loader for _ in sub])
        stderr_print('\x1b[2K', end='')

        #Switch model to evaluation mode
        model = model.eval()
        dev_confm.reset()

        #Validation minibatch loop
        #has the same flow as the training loop above, with the exception
        # of using torch.no_grad() to prevent modifying the weights
        dev_loss = batch_predict(model, grade2i, dev_loader, loss_function, dev_confm,
                                 loadtext='Evaluating',
                                 timer = timer,
                                 mean_loss = True)

        #Record the results
        results = {'epoch':epoch, 'train_loss':train_loss.item(), 'dev_loss':dev_loss.item(),
                   'train_acc':train_confm.accuracy().item(), 'dev_acc':dev_confm.accuracy().item(),
                   'train_f1':train_confm.f_score(mean=True).item(), 'dev_f1': dev_confm.f_score(mean=True).item()}
        training_log.append(results)

        print('{epoch:d}\t{train_loss:.3f}\t{dev_loss:.3f}\t{train_acc:.3f}\t'
              '{dev_acc:.3f}\t{train_f1:.3f}\t{dev_f1:.3f}'.format(**results))

        #Save the current model if it has the highest validation accuracy
        if best_model[0] is None or best_model[0] > results['dev_loss']:
            torch.save(model,f"{output_dir}/{TIMESTAMP}.check")
            best_model = (results['dev_loss'], epoch)

    print()
    print('Training finished in {:02d}:{:02d}:{:02d}.'.format(*timer.since_start()))


    # Load the best model
    if best_model[1] != epoch:
        print(f'Loading model with the highest validation accruacy (Epoch {best_model[1]}.')
        model = torch.load(f'{output_dir}/{TIMESTAMP}.check')

    #Clean up checkpoint file
    os.remove(f'{output_dir}/{TIMESTAMP}.check')

    return model, training_log


def batch_predict(model, grade2i, data_loader, loss_function, conf_matrix,
                  loadtext='Evaluating', timer=None, mean_loss=False, **kwargs):
    n_batches = len(data_loader)
    loss = 0

    model = model.eval()
    with torch.no_grad():
        for batch_n, X in enumerate(data_loader): #L is the vector of text lengths
            stderr_print('{} |{}|'.format(loadtext, loadbar(batch_n / (n_batches - 1))), end='\r')
            sys.stdout.flush()

            #model.init_hidden(batch_size=len(X))

            Y_h = model(X)
            Y = torch.LongTensor([grade2i[t.grade] for t in X])
            #Y_label = torch.FloatTensor([[1 if i==(y-1) else 0 for i in range(len(grade2i))] for y in Y])
            #compute the loss and update the weights with gradient desecent
            loss += loss_function(Y_h, Y)
            
            pred_grades = Y_h.max(dim=1)[1]
            #print(pred_grades,Y)
            conf_matrix.add(pred_grades, Y)

            if timer is not None:
                timer.tick()
    stderr_print('\x1b[2K',end='')

    if mean_loss:
        return loss / len([_ for sub in data_loader for _ in sub])
    return loss

##############################################

def main():
    #Prepare data

    #First to read the embedding file
    stderr_print('Loading embedding...', end='')
    sent2i, sent_embeddings, text2i, text_embeddings = datautil.load_embeddings(dataparams['emb_file'])
    #word2i, word_embeddings = datautil.load_embeddings(dataparams['word_emb'],word=True)
    stderr_print('DONE')

    #Load and index grade list
    stderr_print('Loading grade scale...', end='')
    grade2i, i2grade = datautil.load_grades(dataparams['grade_file'])
    hyperparams['grade_scale'] = len(grade2i)
    hyperparams['padding_id'] = 0
    stderr_print('DONE')

    #Read and index datasets, create tensors
    #Each dataset is a tuple: (input_tensor, targets_tensor, text_length_tensor)
    stderr_print('Loading datasets ... ', end='')


    train_data = datautil.prepare_data(dataparams['train_file'])
    dev_data = datautil.prepare_data(dataparams['dev_file'])
    test_data = datautil.prepare_data(dataparams['test_file'])


    #Create dataloaders, batches of data
    train_loader = datautil.data_loader(train_data, batch_size=hyperparams['batch_size'], shuffle=True)

    dev_loader = datautil.data_loader(dev_data,batch_size=hyperparams['batch_size'], shuffle=False)

    test_loader = datautil.data_loader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)

    stderr_print('DONE')

   


    # Set up the model
    hyperparams['loss_function']      = hyperparams['loss_function']() 
    model = Grader(text2ind=text2i, sent2ind=sent2i,
                   sent_embeddings=sent_embeddings, text_embeddings=text_embeddings,
                    
                   **hyperparams)
    print()
    print('Hyperparameters:')
    print('\n'.join([f'{k}:{v}' for k, v in hyperparams.items()]))
    print()
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))


    # Set up the confusion matrix to record out prediction
    conf_matrix = ConfusionMatrix(hyperparams['grade_scale'],
                                  ignore_index=None,
                                  class_dict=i2grade)
   

    # Train the model
    model, training_log = train(model,
                                grade2i=grade2i,
                                train_loader=train_loader,
                                dev_loader=dev_loader,
                                conf_matrix=conf_matrix,
                                **hyperparams, **dataparams)

    # Save model and training log
    if dataparams['save_model']:
        torch.save({'model':model,
                    'emb_file':dataparams['emb_file'],
                    'grade_file':dataparams['grade_file']},
                     f"{dataparams['output_dir']}/{TIMESTAMP}.model")

    if dataparams['save_log']:
        dictlist_to_csv(training_log, f"{dataparams['output_dir']}/{TIMESTAMP}-log.csv")
        dictlist_to_csv([hyperparams], f"{dataparams['output_dir']}/{TIMESTAMP}-params-log.csv")
        print(f"Training log saved to {dataparams['output_dir']}/{TIMESTAMP}-log.csv")

    #Evaluate model on dev data
    print()
    print('Evaluating on dev data:')
    conf_matrix.reset()
    loss = batch_predict(model,
                         data_loader=dev_loader,
                         conf_matrix=conf_matrix,
                         mean_loss=False,
                         grade2i=grade2i,
                         **hyperparams)
    conf_matrix.print_class_stats()
    print(f"Dev set accuracy: {conf_matrix.accuracy():.4f}")
    print(f"Dev set mean loss: {loss:8g}")
    print()

    if dataparams['save_conf_matrix']:
        conf_matrix.matrix_to_csv(f"{dataparams['output_dir']}/{TIMESTAMP}-confmat-dev.csv")
        print(f"Confusion matrix saved to {dataparams['output_dir']}/{TIMESTAMP}-confmat-dev.csv")

    #Evaluate model on test data
    if evaluate_on_test_data:
        print()
        print('Evaluating on test data:')
        conf_matrix.reset()
        loss = batch_predict(model,
                             data_loader=test_loader,
                             conf_matrix=conf_matrix,
                             mean_loss = False,
                             grade2i=grade2i,
                             **hyperparams)

        conf_matrix.print_class_stats()
        print(f"Test set accuracy: {conf_matrix.accuracy():.4f}")
        print(f"Test set mean loss: {loss:8g}")
        print()

    if dataparams['save_conf_matrix']:
        conf_matrix.matrix_to_csv(f"{dataparams['output_dir']}/{TIMESTAMP}-confmat-test.csv")
        print(f"Confusion matrix saved to {dataparams['output_dir']}/{TIMESTAMP}-confmat-test.csv")


if __name__ == '__main__':
    main()



