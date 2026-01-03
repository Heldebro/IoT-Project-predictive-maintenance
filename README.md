# IoT-Project-predictive-maintenance

Output of the program at the moment is top 5 engines with the highest predicted probability to fail with the following attributes: 
* Unit_number: int
* cycle: int
* pred_proba: float
* rul: int
After top 5 engines the following is added: 
* failed [] List of history of failed engines. 
* just_failed[] List of just failed engines during this tick t. 

ex. 
{'unit_number': 8, 'cycle': 11, 'pred_proba': 0.32, 'rul': 139},{'unit_number': 9, 'cycle': 11, 'pred_proba': 0.32, 'rul': 139}, 'failed': [58, 39, 47], 'just_failed': [23]}
