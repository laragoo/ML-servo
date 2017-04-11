#ML-servo

* USAGE
	`python -i DAC_out.py`
	runtime parameters specified at end of 'DAC_out.py' file and may be modified there

 *FILE DESCRIPTION
data/
	- pretraining data currently in use, using PID controller
data_old_heater/
	- pretraining data for old heater
data_schmitt/
	- pretraining data using schmitt controller
data_thurs/
	- more schmitt data
actormodel.h5
	- saved weights for actor model
actormodel.json
	- saved architecture for actor model
ActorNetwork.py
	- File specifying architecture and methods for actor network
criticmodel.h5
	- saved weights for critic model
criticmodel.json
	- saved architecture for critic model
CriticNetwork.py
	- File specifying architecture and methods for critic network
DAC_out.py
	- Main file for training and running model
OU.py
	- File implementing Ornstein-Uhlenbeck Process for use in training
ReplayBuffer.py
	- File implementing Replay Buffer for use in training
servo.py
	- File implementing data retrieval from file or from arduino
ws339_18.ino
	- File implementing communication to/from arduino
ws339_18.py
	- File implementing python-arduino interface
