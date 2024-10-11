import numpy as np

def clossness_metric(actual_val,Predicted_val,range_normalizer=None):
        '''
        it receives actual  and predicted and calculte the single prediction accuracy (or closeness)
        it need Y (the entire Y block to make sure there is not bias caused by the magnitude of th ecolomns)
        '''
        if range_normalizer is None:
            range_normalizer=actual_val
        pa=np.zeros_like(actual_val)
        for i in range(actual_val.shape[1]):
            base_value=np.min(range_normalizer[:,i])
            scaled_Y=range_normalizer[:,i]-base_value
            Y_avr=np.mean(scaled_Y)
            error=np.abs(actual_val[:,i]-Predicted_val[:,i])
            pa[:,i]=1-(error/Y_avr)
        Prediction_accuracy=np.mean(pa,axis=1)
        return Prediction_accuracy