import tensorflow as tf
import os
import shutil
import datetime

from model import *
from hyperparameters import *

from git import Repo




path_file_note = prepath_data + "note.txt"
path_scr_record = prepath_data + "src_record/"

sess = tf.InteractiveSession()

model_config = ModelConfig() 
model = Model(sess=sess, config=model_config)

def print_note(model_config, path_file_note):

	file_note = open(path_file_note,"w+") 

	file_note.write("Research Note - Hyperparameter\n\n")
	# file_note.write("")
	file_note.write("Git Branch: %s \n\n" % (str(Repo("./../").active_branch.name)))
	file_note.write("Date: %s \n\n" % (str(datetime.datetime.now())))
	file_note.write("Output Standard: %s \n\n" % (str("v1.03")))
	

	hyperparameters_str = str(vars(model_config))

	hyperparameters_str = hyperparameters_str.replace("{", "")
	hyperparameters_str = hyperparameters_str.replace("}", "")
	hyperparameters_str = hyperparameters_str.replace(", ", ",")
	hyperparameters_str = hyperparameters_str.split(',')
	hyperparameters = sorted(hyperparameters_str)

	hyperparameters_str = ""

	for i in range(len(hyperparameters)):
		hyperparameters_str += str(hyperparameters[i]) + "\n"

	file_note.write(str(hyperparameters_str))

	file_note.close()

def copy_scr_code(path_scr_record):

	src_files_to_copy = [f for f in os.listdir("./") if os.path.isfile(os.path.join("./", f))]

	if not os.path.exists(path_scr_record):
            os.makedirs( path_scr_record )

	for i in range(len(src_files_to_copy)):
		shutil.copyfile(src_files_to_copy[i], (path_scr_record + src_files_to_copy[i]))
	

copy_scr_code(path_scr_record=path_scr_record)
print_note(model_config=model_config, path_file_note=path_file_note)

model.Run()


