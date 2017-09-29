#hyperparameters 

val_n_epochs = 10000
val_learning_rate = 0.0003
val_weight_decay_rate =  0.00001
lambda_recon = 0.09 
lambda_adv = 0.5 
lambda_gradient_loss = 120
lambda_starting_gradient_loss = 40
adaptive_lambda_gradient_loss_grow_time_iter = 3200

size_batch = 72 
size_image = 128
size_overlap = 7
size_hiding = 32

gradientDelay = 0 #by iterations

config_trainset_testset_size_ratio = 0.9

sample_result_per_iters = 250 

hp_gradient_pow = 2

prepath_data = "../../ProjsStorage/GradientChannel/"
train_test_set_folder_name = "set" # put your data insidet this folder seperated by train and test folder

path_trainset_pickle = prepath_data + "data/"+train_test_set_folder_name+"/" +train_test_set_folder_name + "_trainset.pickle" 
path_testset_pickle  = prepath_data + "data/"+train_test_set_folder_name+"/" + train_test_set_folder_name + "_testset.pickle" 
path_trainset = prepath_data + "data/" + train_test_set_folder_name + "/train/" 
path_testset = prepath_data + "data/" + train_test_set_folder_name + "/test/" 
path_model = prepath_data + "models/" + train_test_set_folder_name + "/" 
path_result= prepath_data + "results/" + train_test_set_folder_name + "/" 
path_pretrained_model = prepath_data + "models/" + train_test_set_folder_name + "/model-0" 


#end of hyperparameters

class ModelConfig():

    def __init__(self):

        self.path_trainset = path_trainset
        self.path_testset  = path_testset
        self.path_trainset_pickle = path_trainset_pickle
        self.path_testset_pickle  = path_testset_pickle
        self.path_model = path_model
        self.path_result= path_result
        self.path_pretrained_model = path_pretrained_model
        self.size_image = size_image
        self.size_hiding = size_hiding
        self.size_batch = size_batch
        self.size_overlap = size_overlap

        self.val_n_epochs = val_n_epochs
        self.val_learning_rate = val_learning_rate
        self.val_weight_decay_rate = val_weight_decay_rate
        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_gradient_loss = lambda_gradient_loss
        self.lambda_starting_gradient_loss = lambda_starting_gradient_loss
        self.hp_gradient_pow = hp_gradient_pow
        self.gradientDelay = gradientDelay

        self.adaptive_lambda_gradient_loss_grow_time_iter = adaptive_lambda_gradient_loss_grow_time_iter

        self.config_trainset_testset_size_ratio = config_trainset_testset_size_ratio
        self.sample_result_per_iters = sample_result_per_iters



